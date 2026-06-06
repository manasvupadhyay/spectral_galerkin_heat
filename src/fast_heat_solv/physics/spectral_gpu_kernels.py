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
from fast_heat_solv.physics import spectral_ops as _ops

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
        # CuPy/CUDA primitives for the shared free functions in spectral_ops.
        self.hooks = _state.BackendHooks(
            idct=IDCT_II,
            ndshift=_ndshift,
            source_term=_source_term,
        )


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



# Backend-agnostic free functions live in ``spectral_ops``: the pure einsum/copy
# ones read the array module from ``SsState.xp``; the hook-using ones reach the
# FFT / ndimage / source-term primitives below through ``SsState.hooks``. They
# are re-exported here so callers keep using ``spectral_gpu_kernels.<fn>``;
# ``_reconstruct_temperature_box`` is used internally by ``_source_term`` callers.
project_box_to_modes = _ops.project_box_to_modes
_reconstruct_temperature_box = _ops.reconstruct_temperature_box
initialize_latent_heat_if_needed = _ops.initialize_latent_heat_if_needed
update_latent_heat_history = _ops.update_latent_heat_history
reconstruct_surface_temperature = _ops.reconstruct_surface_temperature
reconstruct_bottom_temperature = _ops.reconstruct_bottom_temperature
compute_latent_heat_source = _ops.compute_latent_heat_source
shift_latent_heat_history = _ops.shift_latent_heat_history


def _ndshift(field, shift_pixels, order, mode, cval):
    """ndimage shift primitive (GPU): cupyx.scipy.ndimage into a fresh buffer."""
    out = cp.empty_like(field)
    cupy_ndimage.shift(field, shift_pixels, order=order, mode=mode, cval=cval, output=out)
    return out


def _source_term(T_curr, T_prev, T_S, T_L, rho, L, dt, out):
    """Latent-heat source primitive (GPU): launch the CUDA source-term kernel."""
    blockspergrid, threadsperblock = _launch_config(T_curr.shape)
    compute_source_term_from_temperature[blockspergrid, threadsperblock](
        T_curr, T_prev, T_S, T_L, rho, L, dt, out
    )


def DCT_II(q):
    """Apply Discrete Cosine Transform Type II (Ortho) on GPU."""
    # Cupyx provides dctn in modern versions. 
    # If using older CuPy where dctn is missing, one must use FFT approach.
    # Assuming valid environment:
    return cupy_fft.dctn(q, type=2, norm='ortho', axes=None).astype(cp.float32)

def IDCT_II(a):
    """Apply Discrete Cosine Transform Type III (Inverse Ortho) on GPU."""
    return cupy_fft.dctn(a, type=3, norm='ortho', axes=None).astype(cp.float32)


def shift_flux(field: cp.ndarray, shift: tuple, geom) -> cp.ndarray:
    """Translate a surface flux field by ``shift=(dx, dy)`` meters on GPU."""
    dx, dy = shift
    shift_pixels = (dy / geom.dy, dx / geom.dx)
    return _ndshift(field, shift_pixels, order=1, mode='constant', cval=0.0)