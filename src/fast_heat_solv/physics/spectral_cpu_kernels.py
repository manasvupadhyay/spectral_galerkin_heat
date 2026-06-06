"""CPU-based spectral method kernels for the heat equation.

Public functions in this module are called by SpectralSolver (NumpyBackend):
- ``update_modes_etd1``: Time integration step
- ``compute_latent_heat_source``: Latent heat and evaporation effects
- ``reconstruct_surface_temperature``: Extract solution on top surface
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


import numpy as np
from numba import njit, prange
from scipy.ndimage import shift as scipy_shift
import pyfftw

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
# Spectral Method CPU State Definition
# ======================================
#
# The state classes (SpectralGrid / FineMeshState / SolverBuffers /
# SpectralSolverState) and the propagator precompute are backend-agnostic and
# live in ``spectral_state``; this module only binds the NumPy array module.

SpectralGrid = _state.SpectralGrid
FineMeshState = _state.FineMeshState
SolverBuffers = _state.SolverBuffers


class SpectralSolverState(_state.SpectralSolverState):
    """CPU spectral state: :class:`spectral_state.SpectralSolverState` bound to NumPy."""

    def __init__(self, phys, geom, num):
        super().__init__(phys, geom, num, xp=np)


# ======================================
# Spectral Method CPU Kernels
# ======================================


@njit(parallel=True, fastmath=True, cache=True)
def update_modes_etd1(aK, KK, Cp_broadcast, B_scaled, a_temp_out):
    """
    Update spectral coefficients for ETD1 scheme.
    Calculates: a_out = aK + (KK * Cp) * B_scaled
    """
    nz = aK.shape[0]
    for p in prange(nz):
        a_temp_out[p, :, :] = aK[p, :, :] + KK[p, :, :] * Cp_broadcast[p, 0, 0] * B_scaled

@njit(parallel=True, fastmath=True, cache=True)
def add_source_term_modes(a_temp, KK, Q_modes):
    """
    Accumulate volumetric source term into temperature modes.
    a_temp += KK * Q_modes
    """
    nz = a_temp.shape[0]
    for p in prange(nz):
        for i in range(a_temp.shape[1]):
            for j in range(a_temp.shape[2]):
                a_temp[p, i, j] += KK[p, i, j] * Q_modes[p, i, j]

@njit(parallel=True, fastmath=True, cache=True)
def add_bottom_surface_source(a_temp, KK, Cp_broadcast_bottom, B_scaled):
    """
    Accumulate a surface source at z=0 into temperature modes.
    a_temp[p,:,:] += KK[p,:,:] * Cp_bottom[p] * B_scaled[:,:]
    """
    nz = a_temp.shape[0]
    for p in prange(nz):
        a_temp[p, :, :] += KK[p, :, :] * Cp_broadcast_bottom[p, 0, 0] * B_scaled[:, :]

@njit(parallel=True, fastmath=True, cache=True)
def compute_source_term_from_temperature(T_curr, T_prev, T_S, T_L, rho, L, dt, out):
    """
    Compute Q = - rho * L * (1 / (TL - TS)) * (dT/dt) * Indicator(TS <= T <= TL)
    Used for latent heat calculation.
    """
    nz, ny, nx = T_curr.shape
    for k in prange(nz):
        for j in range(ny):
            for i in range(nx):
                T = T_curr[k, j, i]
                # Indicator function for mushy zone (inclusive)
                if T >= T_S and T <= T_L:
                    T_p = T_prev[k, j, i]
                    
                    # Clamp T_prev to [T_S, T_L]
                    if T_p < T_S:
                        T_p = T_S
                    elif T_p > T_L:
                        T_p = T_L

                    dT = T - T_p
                    factor = -rho * L / ((T_L - T_S) * dt)
                    out[k, j, i] = factor * dT
                else:
                    out[k, j, i] = 0.0



@njit(parallel=True, fastmath=True, cache=True)
def compute_evaporation_flux(T_surface, q_out, P0, T_boil, DeltaH_LV, R_v, T_liquidus):
    """
    Compute evaporative heat flux based on surface temperature using Arrhenius law.
    q_out is updated in-place.

    Signature matches ``spectral_gpu_kernels.compute_evaporation_flux`` so the
    unified solver can call it through either backend.
    """
    ny, nx = T_surface.shape
    factor1 = 0.82 * DeltaH_LV * P0 / np.sqrt(2 * np.pi * R_v)
    factor2 = DeltaH_LV / (R_v * T_boil)
    
    for j in prange(ny):
        for i in range(nx):
            T = T_surface[j, i]
            if T < T_liquidus:
                q_out[j, i] = 0.0
            else:
                # 1/sqrt(T) * exp(...)
                term = (1.0 / np.sqrt(T)) * np.exp(factor2 * (1.0 - T_boil / T))
                q_out[j, i] = factor1 * term


def compute_gaussian_laser_flux(x, y, laser_x, laser_y, laser_r, laser_coef):
    """Compute Gaussian flux on grid defined by 1D arrays x, y."""
    # (nx,) -> (1, nx)
    dx = x[None, :] - laser_x
    # (ny,) -> (ny, 1)
    dy = y[:, None] - laser_y
    r_sq = dx**2 + dy**2
    return (laser_coef * np.exp(-2.0 * r_sq / laser_r ** 2))


def project_box_to_modes(field_box, SsState):
    """Project fine box field to global spectral modes."""
    if SsState.fine_mesh is None:
        raise RuntimeError("Fine mesh not initialized.")
    fm = SsState.fine_mesh
    modes = np.einsum('zyx,Zz,Yy,Xx->ZYX', field_box, fm.B_fine[2], fm.B_fine[1], fm.B_fine[0], optimize=True)
    return modes * fm.dV_fine


def _reconstruct_temperature_box(a, SsState):
    """Reconstructs temperature in a small ROI around the laser."""
    if SsState.fine_mesh is None:
        raise RuntimeError("Fine mesh not initialized.")
    fm = SsState.fine_mesh
    return np.einsum('ZYX,Zz,Yy,Xx->zyx', a, fm.B_fine[2], fm.B_fine[1], fm.B_fine[0], optimize=True)

def initialize_latent_heat_if_needed(SsState):
    """Initialize fine-mesh T_prev from current trial modes on the first time step.

    Must be called after ``buffers.a_temp`` has been set to a reasonable
    estimate (e.g. the decayed modes).
    """
    fm = SsState.fine_mesh
    if fm is None or fm.T_prev is not None:
        return
    T_box = _reconstruct_temperature_box(SsState.buffers.a_temp, SsState)
    fm.T_prev = T_box.copy()
    if fm.Q_prev is None:
        fm.Q_prev = np.zeros_like(SsState.buffers.Q_latent_buffer)


def shift_latent_heat_history(fm, laser_state, num):
    """Shift T_prev and Q_prev to align with the current laser position.

    Call once per time step, before the fixed-point iteration begins.
    """
    if fm is None or fm.T_prev is None:
        return
    shift_x = laser_state.v[0] * num.dt
    shift_y = laser_state.v[1] * num.dt
    shift_pixels = (0, -shift_y / fm.dy_fine, -shift_x / fm.dx_fine)
    fm.T_prev = scipy_shift(fm.T_prev, shift_pixels, order=1, mode='nearest')
    if fm.Q_prev is not None:
        fm.Q_prev = scipy_shift(fm.Q_prev, shift_pixels, order=1, mode='constant', cval=0.0)


def compute_latent_heat_source(Q_buffer, phys, num, SsState):
    """Compute volumetric latent heat source Q (W/m^3) on the fine mesh.

    Uses the current trial modes (``buffers.a_temp``) and the stored
    ``T_prev``.  Does not update ``T_prev``; call
    :func:`update_latent_heat_history` after the iteration has converged.
    """
    fm = SsState.fine_mesh
    if fm is None or fm.T_prev is None:
        Q_buffer.fill(0.0)
        return

    T_box = _reconstruct_temperature_box(SsState.buffers.a_temp, SsState)

    compute_source_term_from_temperature(
        T_box, fm.T_prev,
        phys.T_solidus, phys.T_liquidus,
        phys.rho, phys.L_f, num.dt,
        Q_buffer
    )


def update_latent_heat_history(SsState):
    """Store the converged fine-mesh temperature as T_prev for the next step.

    Call once per time step, after the fixed-point iteration has converged
    and ``buffers.a_temp`` holds the final spectral modes.
    """
    fm = SsState.fine_mesh
    if fm is None:
        return
    T_box = _reconstruct_temperature_box(SsState.buffers.a_temp, SsState)
    fm.T_prev[:] = T_box[:]


def DCT_II(q):
    """Apply Discrete Cosine Transform Type II (Ortho)."""
    arr = np.ascontiguousarray(q, dtype=np.float32)
    return pyfftw.interfaces.scipy_fft.dctn(arr, type=2, norm='ortho', axes=tuple(range(arr.ndim)), workers=-1)

def IDCT_II(a):
    """Apply Discrete Cosine Transform Type II (Ortho)."""
    arr = np.ascontiguousarray(a, dtype=np.float32)
    return pyfftw.interfaces.scipy_fft.dctn(arr, type=3, norm='ortho', axes=tuple(range(arr.ndim)), workers=-1)

# Gain of few percent compared to scipy.fft.dctn(...) directly

def reconstruct_surface_temperature(a, SsState):
    """Reconstruct 2D temperature field at z=0."""
    # Sum over Z modes weighted by Cp evaluated at the top surface.
    # Use a preallocated 2D buffer when available to avoid allocating a full
    # temporary (nz, ny, nx) array. `np.einsum` with `out=` performs the
    # contraction
    A = np.einsum('p,pij->ij', SsState.grid.Cp32_broadcast[:, 0, 0], a, optimize=True)
    # Use DCT-II for surface temperature (mathematical definition)
    dct_result = IDCT_II(A)
    return (SsState.grid.recon_scale * dct_result)

def reconstruct_bottom_temperature(a, SsState):
    """Reconstruct 2D temperature field at z=0 (bottom surface).
    
    At z=0, cos(p*pi*0/Lz) = 1, so the weighting is just Cp (no sign alternation).
    """
    A = np.einsum('p,pij->ij', SsState.grid.Cp32_broadcast_bottom[:, 0, 0], a, optimize=True)
    dct_result = IDCT_II(A)
    return (SsState.grid.recon_scale * dct_result)

def shift_flux(field: np.ndarray, shift: tuple, geom) -> np.ndarray:
    """Translate a surface flux field by ``shift=(dx, dy)`` meters."""
    dx, dy = shift
    shift_pixels = (dy / geom.dy, dx / geom.dx)
    
    return scipy_shift(field, shift_pixels, order=1, mode='constant', cval=0.0)
