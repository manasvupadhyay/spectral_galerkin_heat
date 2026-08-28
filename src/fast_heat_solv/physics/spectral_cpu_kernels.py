"""CPU-based spectral method kernels for the heat equation.

Public functions in this module are called by SpectralSolver (NumpyBackend):
- ``update_modes_etd1``: Time integration step
- ``compute_latent_heat_source``: Latent heat and evaporation effects
- ``reconstruct_surface_temperature``: Extract solution on top surface
"""

# Copyright 2026 Laboratoire de Mécanique des Solides (LMS),
# École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris,
# Route de Saclay, Palaiseau, 91128, France.
#
# Author: Théo Andrieux, Jules Dichamp, Manas V. Upadhyay
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
import scipy.fft
from numba import njit, prange
from scipy.ndimage import shift as scipy_shift

from fast_heat_solv.core.laser import super_gaussian_flux as _super_gaussian_flux
from fast_heat_solv.physics import spectral_ops as _ops
from fast_heat_solv.physics import spectral_state as _state

__all__ = [
    "FineMeshState",
    "SpectralGrid",
    "SpectralSolverState",
    "add_bottom_surface_source",
    "add_source_term_modes",
    "compute_latent_heat_source",
    "compute_latent_heat_source_grid",
    "reconstruct_bottom_temperature",
    "reconstruct_surface_temperature",
    "update_modes_etd1",
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
            dct=DCT_II,
            corr_source=_correction_source_cpu,
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
    Compute Q = - rho * L * (f_l(T_curr) - f_l(T_prev)) / dt, with the liquid
    fraction f_l clamped to [0, 1].
    Used for latent heat calculation.
    """
    nz, ny, nx = T_curr.shape
    inv_band = 1.0 / (T_L - T_S)
    factor = -rho * L / dt
    for k in prange(nz):
        for j in range(ny):
            for i in range(nx):
                f_c = (T_curr[k, j, i] - T_S) * inv_band
                if f_c < 0.0:
                    f_c = 0.0
                elif f_c > 1.0:
                    f_c = 1.0

                f_p = (T_prev[k, j, i] - T_S) * inv_band
                if f_p < 0.0:
                    f_p = 0.0
                elif f_p > 1.0:
                    f_p = 1.0

                out[k, j, i] = factor * (f_c - f_p)



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


@njit(parallel=True, fastmath=True, cache=True)
def _grad_kprime_cpu(T, kp, inv_dx, inv_dy, inv_dz, gx, gy, gz):
    """First pass of the divergence-form correction: ``g = k'(T) ∇T`` (CPU).

    Central differences in the interior, first-order one-sided at the faces:
    the ``numpy.gradient(.., edge_order=1)`` stencil, fused with the ``k'``
    multiply.
    """
    nz, ny, nx = T.shape
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                if x == 0:
                    dtx = (T[z, y, 1] - T[z, y, 0]) * inv_dx
                elif x == nx - 1:
                    dtx = (T[z, y, nx - 1] - T[z, y, nx - 2]) * inv_dx
                else:
                    dtx = (T[z, y, x + 1] - T[z, y, x - 1]) * (0.5 * inv_dx)
                if y == 0:
                    dty = (T[z, 1, x] - T[z, 0, x]) * inv_dy
                elif y == ny - 1:
                    dty = (T[z, ny - 1, x] - T[z, ny - 2, x]) * inv_dy
                else:
                    dty = (T[z, y + 1, x] - T[z, y - 1, x]) * (0.5 * inv_dy)
                if z == 0:
                    dtz = (T[1, y, x] - T[0, y, x]) * inv_dz
                elif z == nz - 1:
                    dtz = (T[nz - 1, y, x] - T[nz - 2, y, x]) * inv_dz
                else:
                    dtz = (T[z + 1, y, x] - T[z - 1, y, x]) * (0.5 * inv_dz)
                kv = kp[z, y, x]
                gx[z, y, x] = kv * dtx
                gy[z, y, x] = kv * dty
                gz[z, y, x] = kv * dtz


@njit(parallel=True, fastmath=True, cache=True)
def _div_minus_capacity_cpu(gx, gy, gz, ap, T, Tprev,
                            inv_dx, inv_dy, inv_dz, inv_dt, out):
    """Second pass: ``f = ∇·g - a'(T)(T - T_prev)/dt`` (CPU).

    Same ``numpy.gradient`` stencil applied to ``g``, capacity source fused in.
    Mirrors the GPU ``_div_minus_capacity_kernel``.
    """
    nz, ny, nx = T.shape
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                if x == 0:
                    dgx = (gx[z, y, 1] - gx[z, y, 0]) * inv_dx
                elif x == nx - 1:
                    dgx = (gx[z, y, nx - 1] - gx[z, y, nx - 2]) * inv_dx
                else:
                    dgx = (gx[z, y, x + 1] - gx[z, y, x - 1]) * (0.5 * inv_dx)
                if y == 0:
                    dgy = (gy[z, 1, x] - gy[z, 0, x]) * inv_dy
                elif y == ny - 1:
                    dgy = (gy[z, ny - 1, x] - gy[z, ny - 2, x]) * inv_dy
                else:
                    dgy = (gy[z, y + 1, x] - gy[z, y - 1, x]) * (0.5 * inv_dy)
                if z == 0:
                    dgz = (gz[1, y, x] - gz[0, y, x]) * inv_dz
                elif z == nz - 1:
                    dgz = (gz[nz - 1, y, x] - gz[nz - 2, y, x]) * inv_dz
                else:
                    dgz = (gz[z + 1, y, x] - gz[z - 1, y, x]) * (0.5 * inv_dz)
                dTdt = (T[z, y, x] - Tprev[z, y, x]) * inv_dt
                out[z, y, x] = dgx + dgy + dgz - ap[z, y, x] * dTdt


def _correction_source_cpu(T, k_prime, a_prime, T_prev, dt, dx, dy, dz):
    """Divergence-form correction forcing ``f = ∇·(k'∇T) - a'∂_tT`` (CPU).

    Two fused ``prange`` stencil passes.
    """
    gx = np.empty_like(T)
    gy = np.empty_like(T)
    gz = np.empty_like(T)
    out = np.empty_like(T)
    real_t = T.dtype.type
    inv_dx = real_t(1.0 / dx)
    inv_dy = real_t(1.0 / dy)
    inv_dz = real_t(1.0 / dz)
    inv_dt = real_t(1.0 / dt)
    _grad_kprime_cpu(T, k_prime, inv_dx, inv_dy, inv_dz, gx, gy, gz)
    _div_minus_capacity_cpu(gx, gy, gz, a_prime, T, T_prev,
                            inv_dx, inv_dy, inv_dz, inv_dt, out)
    return out


def compute_gaussian_laser_flux(x, y, laser_x, laser_y, laser_r, laser_coef):
    """Gaussian (order-2 super-Gaussian) flux on the 1-D grid ``(x, y)``.

    Thin backend wrapper over the shared, backend-agnostic
    :func:`~fast_heat_solv.core.laser.super_gaussian_flux` so the CPU and GPU
    paths use one definition.
    """
    return _super_gaussian_flux(np, x, y, laser_x, laser_y, laser_r, laser_r, 2.0, laser_coef)


# Backend-agnostic free functions live in ``spectral_ops``; re-exported here so
# callers keep using ``spectral_cpu_kernels.<fn>``.
project_box_to_modes = _ops.project_box_to_modes
_reconstruct_temperature_box = _ops.reconstruct_temperature_box
initialize_latent_heat_if_needed = _ops.initialize_latent_heat_if_needed
update_latent_heat_history = _ops.update_latent_heat_history
reconstruct_surface_temperature = _ops.reconstruct_surface_temperature
reconstruct_bottom_temperature = _ops.reconstruct_bottom_temperature
compute_latent_heat_source = _ops.compute_latent_heat_source
compute_latent_heat_source_grid = _ops.compute_latent_heat_source_grid
shift_latent_heat_history = _ops.shift_latent_heat_history
reconstruct_volume = _ops.reconstruct_volume
project_volume = _ops.project_volume
assemble_property_correction = _ops.assemble_property_correction


@njit(parallel=True, fastmath=True, cache=True)
def _shift_xy_2d(field, i0y, i1y, w0y, w1y, iny,
                 i0x, i1x, w0x, w1x, inx, nearest, cval, out):
    """Bilinear x/y translation of a 2-D field from precomputed axis tables."""
    ny, nx = field.shape
    for oy in prange(ny):
        a = i0y[oy]
        b = i1y[oy]
        wy0 = w0y[oy]
        wy1 = w1y[oy]
        yin = iny[oy]
        for ox in range(nx):
            c = i0x[ox]
            d = i1x[ox]
            val = (wy0 * (w0x[ox] * field[a, c] + w1x[ox] * field[a, d])
                   + wy1 * (w0x[ox] * field[b, c] + w1x[ox] * field[b, d]))
            if (not nearest) and not (yin and inx[ox]):
                val = cval
            out[oy, ox] = val


@njit(parallel=True, fastmath=True, cache=True)
def _shift_xy_3d(field, i0y, i1y, w0y, w1y, iny,
                 i0x, i1x, w0x, w1x, inx, nearest, cval, out):
    """Bilinear x/y translation of a 3-D volume (z-shift == 0), parallel over z.

    Each z-slice is translated by the same (y, x) offset, so the
    per-axis index/weight/in-range tables are shared across all slices.
    """
    nz, ny, nx = field.shape
    for z in prange(nz):
        for oy in range(ny):
            a = i0y[oy]
            b = i1y[oy]
            wy0 = w0y[oy]
            wy1 = w1y[oy]
            yin = iny[oy]
            for ox in range(nx):
                c = i0x[ox]
                d = i1x[ox]
                val = (wy0 * (w0x[ox] * field[z, a, c] + w1x[ox] * field[z, a, d])
                       + wy1 * (w0x[ox] * field[z, b, c] + w1x[ox] * field[z, b, d]))
                if (not nearest) and not (yin and inx[ox]):
                    val = cval
                out[z, oy, ox] = val


def _shift_axis_tables(n, shift, dtype):
    """Precompute clamped neighbour indices, bilinear weights and in-range mask.

    For output index ``o`` the source coordinate is ``p = o - shift`` (scipy's
    ``out[o] = in(o - shift)``). Returns clamped ``i0``/``i1`` (so the weight-0
    upper neighbour at the far edge never reads out of bounds), the linear
    weights, and a boolean ``in-range`` mask (``0 <= p <= n-1``) used only by
    'constant' mode. scipy fills the output with ``cval`` when ``p`` leaves the
    array, rather than blending against the boundary.
    """
    o = np.arange(n, dtype=np.float64)
    p = o - float(shift)
    i0 = np.floor(p).astype(np.int64)
    t = p - i0
    in_range = (p >= 0.0) & (p <= n - 1)
    i0c = np.clip(i0, 0, n - 1)
    i1c = np.clip(i0 + 1, 0, n - 1)
    w1 = t.astype(dtype)
    w0 = (1.0 - t).astype(dtype)
    return i0c, i1c, w0, w1, in_range


def _ndshift(field, shift_pixels, order, mode, cval):
    """Multi-threaded order-1 ndimage shift for the solver's x/y translations.

    2-D fields and 3-D volumes translated only in (y, x) (z-shift == 0), 
    ``order=1``, modes ``'nearest'``/``'constant'``. 
    Any other request falls back to ``scipy.ndimage.shift``.
    """
    if order == 1 and mode in ("nearest", "constant") and field.ndim in (2, 3):
        is_3d = field.ndim == 3
        if not is_3d or shift_pixels[0] == 0:
            sy, sx = (shift_pixels[1], shift_pixels[2]) if is_3d else \
                     (shift_pixels[0], shift_pixels[1])
            dt = field.dtype.type
            i0y, i1y, w0y, w1y, iny = _shift_axis_tables(field.shape[-2], sy, dt)
            i0x, i1x, w0x, w1x, inx = _shift_axis_tables(field.shape[-1], sx, dt)
            nearest = mode == "nearest"
            out = np.empty_like(field)
            kernel = _shift_xy_3d if is_3d else _shift_xy_2d
            kernel(field, i0y, i1y, w0y, w1y, iny,
                   i0x, i1x, w0x, w1x, inx, nearest, dt(cval), out)
            return out
    return scipy_shift(field, shift_pixels, order=order, mode=mode, cval=cval)


def DCT_II(q):
    """Apply Discrete Cosine Transform Type II (Ortho).

    Precision-transparent: the transform runs (and returns) in the input
    array's float dtype (float32 or float64).
    """
    arr = np.ascontiguousarray(q)
    return scipy.fft.dctn(arr, type=2, norm='ortho', axes=tuple(range(arr.ndim)), workers=-1)

def IDCT_II(a):
    """Apply Discrete Cosine Transform Type II (Ortho).

    Precision-transparent: runs in the input array's float dtype.
    """
    arr = np.ascontiguousarray(a)
    return scipy.fft.dctn(arr, type=3, norm='ortho', axes=tuple(range(arr.ndim)), workers=-1)


def shift_flux(field: np.ndarray, shift: tuple, geom) -> np.ndarray:
    """Translate a surface flux field by ``shift=(dx, dy)`` meters."""
    dx, dy = shift
    shift_pixels = (dy / geom.d.y, dx / geom.d.x)
    return _ndshift(field, shift_pixels, order=1, mode='constant', cval=0.0)
