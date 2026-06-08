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

    def __init__(self, phys, geom, num, fine):
        super().__init__(phys, geom, num, fine, xp=np)
        # NumPy/Numba primitives for the shared free functions in spectral_ops.
        self.hooks = _state.BackendHooks(
            idct=IDCT_II,
            ndshift=_ndshift,
            source_term=compute_source_term_from_temperature,
        )


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


# Backend-agnostic free functions live in ``spectral_ops``; re-exported here so
# callers keep using ``spectral_cpu_kernels.<fn>``.
project_box_to_modes = _ops.project_box_to_modes
_reconstruct_temperature_box = _ops.reconstruct_temperature_box
initialize_latent_heat_if_needed = _ops.initialize_latent_heat_if_needed
update_latent_heat_history = _ops.update_latent_heat_history
reconstruct_surface_temperature = _ops.reconstruct_surface_temperature
reconstruct_bottom_temperature = _ops.reconstruct_bottom_temperature
compute_latent_heat_source = _ops.compute_latent_heat_source
shift_latent_heat_history = _ops.shift_latent_heat_history


def _ndshift(field, shift_pixels, order, mode, cval):
    """ndimage shift primitive (CPU): scipy.ndimage, returns a new array."""
    return scipy_shift(field, shift_pixels, order=order, mode=mode, cval=cval)


def DCT_II(q):
    """Apply Discrete Cosine Transform Type II (Ortho)."""
    arr = np.ascontiguousarray(q, dtype=np.float32)
    return pyfftw.interfaces.scipy_fft.dctn(arr, type=2, norm='ortho', axes=tuple(range(arr.ndim)), workers=-1)

def IDCT_II(a):
    """Apply Discrete Cosine Transform Type II (Ortho)."""
    arr = np.ascontiguousarray(a, dtype=np.float32)
    return pyfftw.interfaces.scipy_fft.dctn(arr, type=3, norm='ortho', axes=tuple(range(arr.ndim)), workers=-1)

# Gain of few percent compared to scipy.fft.dctn(...) directly


def shift_flux(field: np.ndarray, shift: tuple, geom) -> np.ndarray:
    """Translate a surface flux field by ``shift=(dx, dy)`` meters."""
    dx, dy = shift
    shift_pixels = (dy / geom.d.y, dx / geom.d.x)
    return _ndshift(field, shift_pixels, order=1, mode='constant', cval=0.0)
