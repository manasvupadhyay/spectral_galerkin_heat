"""GPU-based spectral method kernels for the heat equation (requires CuPy).

Public functions in this module mirror CPU kernels and are called by
SpectralSolver (CupyBackend).
"""

# Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique
#
# Author: Théo Andrieux
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import cupyx.scipy.ndimage as cupy_ndimage
import cupyx.scipy.fft as cupy_fft
from numba import cuda
import math

from fast_heat_solv.physics import spectral_state as _state

__all__ = [
    "SpectralSolverState",
    "SpectralGrid",
    "FineMeshState",
    "update_modes_etd1",
    "add_source_term_modes",
    "add_bottom_surface_source",
    "compute_latent_heat_source",
    "reconstruct_surface_temperature",
    "reconstruct_bottom_temperature",
]

# ======================================
# CUDA Kernels (Device Functions)
# ======================================

@cuda.jit
def _update_modes_etd1_kernel(aK, KK, Cp_broadcast, B_scaled, a_temp_out):
    """
    Update spectral coefficients for ETD1 scheme.
    Grid: 3D (nz, ny, nx)
    """
    z, y, x = cuda.grid(3)
    nz, ny, nx = aK.shape

    if z < nz and y < ny and x < nx:
        # Reconstruct KK_by_Cp factor on the fly: KK * Cp_broadcast
        factor = KK[z, y, x] * Cp_broadcast[z, 0, 0]
        a_temp_out[z, y, x] = aK[z, y, x] + factor * B_scaled[y, x]

@cuda.jit
def _add_source_term_modes_kernel(a_temp, KK, Q_modes):
    """
    Accumulate volumetric source term into temperature modes.
    Grid: 3D (nz, ny, nx)
    """
    z, y, x = cuda.grid(3)
    nz, ny, nx = a_temp.shape
    
    if z < nz and y < ny and x < nx:
        a_temp[z, y, x] += KK[z, y, x] * Q_modes[z, y, x]


@cuda.jit
def _add_bottom_surface_source_kernel(a_temp, KK, Cp_broadcast_bottom, B_scaled):
    """
    Accumulate a surface source at z=0 into temperature modes.
    a_temp[z,y,x] += KK[z,y,x] * Cp_bottom[z] * B_scaled[y,x]
    Grid: 3D (nz, ny, nx)
    """
    z, y, x = cuda.grid(3)
    nz, ny, nx = a_temp.shape

    if z < nz and y < ny and x < nx:
        a_temp[z, y, x] += KK[z, y, x] * Cp_broadcast_bottom[z, 0, 0] * B_scaled[y, x]



@cuda.jit
def _compute_evaporation_flux_kernel(T_surface, q_out, P0, T_boil, DeltaH_LV, R_v, T_liquidus):
    """
    Compute evaporative heat flux (Arrhenius law).
    Grid: 2D (ny, nx)
    """
    y, x = cuda.grid(2)
    ny, nx = T_surface.shape
    
    if y < ny and x < nx:
        T = T_surface[y, x]
        if T < T_liquidus:
            q_out[y, x] = 0.0
        else:
            # Constants pre-calculated or passed
            # factor1 = 0.82 * DeltaH_LV * P0 / sqrt(2 * pi * R_v)
            # Calculated inside or passed? Calculating inside for clarity, 
            # ideally pass as uniform to save registers.
            
            # Using math.sqrt/exp for scalar device functions
            factor1 = 0.82 * DeltaH_LV * P0 / math.sqrt(2.0 * math.pi * R_v)
            factor2 = DeltaH_LV / (R_v * T_boil)
            
            term = (1.0 / math.sqrt(T)) * math.exp(factor2 * (1.0 - T_boil / T))
            q_out[y, x] = factor1 * term

# ======================================
# Spectral Method GPU State Definition
# ======================================
#
# The state classes (SpectralGrid / FineMeshState / SolverBuffers /
# SpectralSolverState) and the propagator precompute are backend-agnostic and
# live in ``spectral_state``; this module only binds the CuPy array module.

SpectralGrid = _state.SpectralGrid
FineMeshState = _state.FineMeshState
SolverBuffers = _state.SolverBuffers


class SpectralSolverState(_state.SpectralSolverState):
    """GPU spectral state: :class:`spectral_state.SpectralSolverState` bound to CuPy."""

    def __init__(self, phys, geom, num):
        super().__init__(phys, geom, num, xp=cp)


# ======================================
# Wrapper Functions
# ======================================

def _launch_config(shape, tpb=None):
    """Compute (blockspergrid, threadsperblock) for a CUDA kernel launch."""
    if tpb is None:
        tpb = (8,) * len(shape)
    return tuple((d + t - 1) // t for d, t in zip(shape, tpb)), tpb


def update_modes_etd1(aK, KK, Cp_broadcast, B_scaled, a_temp_out):
    """Wrapper for ETD1 kernel."""
    blockspergrid, threadsperblock = _launch_config(aK.shape)
    _update_modes_etd1_kernel[blockspergrid, threadsperblock](aK, KK, Cp_broadcast, B_scaled, a_temp_out)


def add_source_term_modes(a_temp, KK, Q_modes):
    """Wrapper for Source Term Accumulation."""
    blockspergrid, threadsperblock = _launch_config(a_temp.shape)
    _add_source_term_modes_kernel[blockspergrid, threadsperblock](a_temp, KK, Q_modes)


def add_bottom_surface_source(a_temp, KK, Cp_broadcast_bottom, B_scaled):
    """Wrapper for bottom surface source accumulation (GPU)."""
    blockspergrid, threadsperblock = _launch_config(a_temp.shape)
    _add_bottom_surface_source_kernel[blockspergrid, threadsperblock](
        a_temp, KK, Cp_broadcast_bottom, B_scaled
    )


@cuda.jit
def compute_source_term_from_temperature(T_curr, T_prev, T_S, T_L, rho, L, dt, out):
    """
    Compute Q = - rho * L * (1 / (TL - TS)) * (dT/dt) * Indicator(TS <= T <= TL)
    Used for latent heat calculation. GPU version (CUDA kernel).
    """
    z, y, x = cuda.grid(3)
    nz, ny, nx = T_curr.shape
    if z < nz and y < ny and x < nx:
        T = T_curr[z, y, x]
        # Indicator function for mushy zone (inclusive)
        if T >= T_S and T <= T_L:
            T_p = T_prev[z, y, x]
            # Clamp T_prev to [T_S, T_L]
            lower = T_S
            upper = T_L
            if T_p < lower:
                T_p = lower
            elif T_p > upper:
                T_p = upper
            dT = T - T_p
            factor = -rho * L / ((T_L - T_S) * dt)
            out[z, y, x] = factor * dT
        else:
            out[z, y, x] = 0.0
def compute_evaporation_flux(T_surface, q_out, P0, T_boil, DeltaH_LV, R_v, T_liquidus):
    """Wrapper for Evaporation kernel."""
    blockspergrid, threadsperblock = _launch_config(T_surface.shape, (16, 16))
    _compute_evaporation_flux_kernel[blockspergrid, threadsperblock](
        T_surface, q_out, P0, T_boil, DeltaH_LV, R_v, T_liquidus
    )


def compute_gaussian_laser_flux(X, Y, laser_x, laser_y, laser_r, laser_coef):
    """Compute Gaussian flux on grid defined by 1D CuPy arrays X, Y.

    Mirrors the CPU implementation: form 2D mesh via broadcasting to avoid
    shape-broadcast errors when X and Y are 1D arrays of different lengths.
    """
    # Ensure X, Y are 1D arrays (cell-centered coordinates)
    # dx: shape (1, nx), dy: shape (ny, 1)
    dx = X[None, :] - laser_x
    dy = Y[:, None] - laser_y
    r_sq = dx ** 2 + dy ** 2
    return (laser_coef * cp.exp(-2.0 * r_sq / (laser_r ** 2))).astype(cp.float32)



def project_box_to_modes(field_box, SsState):
    """Project fine box field to global spectral modes."""
    if SsState.fine_mesh is None:
        raise RuntimeError("Fine mesh not initialized.")
    fm = SsState.fine_mesh
    modes = cp.einsum('zyx,Zz,Yy,Xx->ZYX', field_box, fm.B_fine[2], fm.B_fine[1], fm.B_fine[0], optimize=True)
    return modes * fm.dV_fine


def _reconstruct_temperature_box(a, SsState):
    """Reconstructs temperature in a small ROI around the laser."""
    if SsState.fine_mesh is None:
        raise RuntimeError("Fine mesh not initialized.")
    fm = SsState.fine_mesh
    return cp.einsum('ZYX,Zz,Yy,Xx->zyx', a, fm.B_fine[2], fm.B_fine[1], fm.B_fine[0], optimize=True)



def initialize_latent_heat_if_needed(SsState):
    """Initialize fine-mesh T_prev from current trial modes on the first time step."""
    fm = SsState.fine_mesh
    if fm is None or fm.T_prev is not None:
        return
    T_box = _reconstruct_temperature_box(SsState.buffers.a_temp, SsState)
    fm.T_prev = T_box.copy()
    if fm.Q_prev is None:
        fm.Q_prev = cp.zeros_like(SsState.buffers.Q_latent_buffer)


def shift_latent_heat_history(fm, laser_state, num):
    """Shift T_prev and Q_prev to align with the current laser position.

    Call once per time step, before the fixed-point iteration begins.
    """
    if fm is None or fm.T_prev is None:
        return
    shift_x = laser_state.v[0] * num.dt
    shift_y = laser_state.v[1] * num.dt
    shift_pixels = (0, -shift_y / fm.dy_fine, -shift_x / fm.dx_fine)
    T_prev_shifted = cp.empty_like(fm.T_prev)
    cupy_ndimage.shift(fm.T_prev, shift_pixels, output=T_prev_shifted, order=1, mode='nearest')
    fm.T_prev = T_prev_shifted
    if fm.Q_prev is not None:
        Q_prev_shifted = cp.empty_like(fm.Q_prev)
        cupy_ndimage.shift(fm.Q_prev, shift_pixels, output=Q_prev_shifted, order=1, mode='constant', cval=0.0)
        fm.Q_prev = Q_prev_shifted


def compute_latent_heat_source(Q_buffer, phys, num, SsState):
    """Compute volumetric latent heat source Q (W/m^3) on the fine mesh.

    Uses the current trial modes (buffers.a_temp) and the stored T_prev.
    Does not update T_prev; call update_latent_heat_history after convergence.
    """
    fm = SsState.fine_mesh
    if fm is None or fm.T_prev is None:
        Q_buffer.fill(0.0)
        return

    T_box = _reconstruct_temperature_box(SsState.buffers.a_temp, SsState)

    blockspergrid, threadsperblock = _launch_config(T_box.shape)
    compute_source_term_from_temperature[blockspergrid, threadsperblock](
        T_box, fm.T_prev,
        phys.T_solidus, phys.T_liquidus,
        phys.rho, phys.L_f, num.dt,
        Q_buffer
    )


def update_latent_heat_history(SsState):
    """Store the converged fine-mesh temperature as T_prev for the next step.

    Call once per time step, after the fixed-point iteration has converged.
    """
    fm = SsState.fine_mesh
    if fm is None:
        return
    T_box = _reconstruct_temperature_box(SsState.buffers.a_temp, SsState)
    fm.T_prev[:] = T_box[:]


def DCT_II(q):
    """Apply Discrete Cosine Transform Type II (Ortho) on GPU."""
    # Cupyx provides dctn in modern versions. 
    # If using older CuPy where dctn is missing, one must use FFT approach.
    # Assuming valid environment:
    return cupy_fft.dctn(q, type=2, norm='ortho', axes=None).astype(cp.float32)

def IDCT_II(a):
    """Apply Discrete Cosine Transform Type III (Inverse Ortho) on GPU."""
    return cupy_fft.dctn(a, type=3, norm='ortho', axes=None).astype(cp.float32)

def reconstruct_surface_temperature(a, SsState):
    """Reconstruct 2D temperature field at z=0 (GPU)."""
   # Sum over Z modes (weighted by Cp coefficients at z=0, which is just Cp/sqrt(1/L)?? No)
    # In helpers.py: A = (SsState.Cp32[:, None, None] * a).sum(axis=0)
    # This assumes cos(p*pi*z/Lz) at z=0 is 1.0. 
    # The reconstruction formula is T = sum(a * Bx * By * Bz).
    # Bz[p] at z=0 is Cp[p] * cos(p*pi*z/Lz) -> z top surface
    A = ( SsState.grid.Cp32_broadcast * a).sum(axis=0)
    
    # Use DCT-III (IDCT) for surface temperature
    dct_result = IDCT_II(A)
    return (SsState.grid.recon_scale * dct_result).astype(cp.float32)

def reconstruct_bottom_temperature(a, SsState):
    """Reconstruct 2D temperature field at z=0 (bottom surface, GPU).
    
    At z=0, cos(p*pi*0/Lz) = 1, so the weighting is just Cp (no sign alternation).
    """
    A = (SsState.grid.Cp32_broadcast_bottom * a).sum(axis=0)
    dct_result = IDCT_II(A)
    return (SsState.grid.recon_scale * dct_result).astype(cp.float32)



def shift_flux(field: cp.ndarray, shift: tuple, geom) -> cp.ndarray:
    """Translate a surface flux field by ``shift=(dx, dy)`` meters on GPU."""
    dx, dy = shift
    shift_pixels = (dy / geom.dy, dx / geom.dx)
    out = cp.empty_like(field)
    cupy_ndimage.shift(field, shift_pixels, order=1, mode='constant', cval=0.0, output=out)
    return out