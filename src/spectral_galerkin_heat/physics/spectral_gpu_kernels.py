"""GPU-based spectral method kernels for the heat equation (requires CuPy).

Public functions in this module mirror CPU kernels and are called by
SpectralSolver (CupyBackend).
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

import math

import cupy as cp
import cupyx.scipy.ndimage as cupy_ndimage
import numpy as np
from numba import cuda

from spectral_galerkin_heat.core.laser import super_gaussian_flux as _super_gaussian_flux
from spectral_galerkin_heat.physics import spectral_ops as _ops
from spectral_galerkin_heat.physics import spectral_state as _state

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
# CUDA Kernels (Device Functions)
# ======================================

# CuPy ``ElementwiseKernel``s rather than numba ``@cuda.jit``: launching a
# numba kernel on a CuPy array carries a per-call overhead these small kernels
# cannot amortise. CuPy broadcasts the operands (``Cp_broadcast`` is (nz,1,1),
# ``B_scaled`` is (ny,nx)) and each writes into a caller-owned buffer, so no
# full-grid temporary is created.

_ek_update_modes_etd1 = cp.ElementwiseKernel(
    "F aK, F KK, F Cp, F B", "F out",
    "out = aK + (KK * Cp) * B",
    "fhs_update_modes_etd1")

_ek_add_source_term_modes = cp.ElementwiseKernel(
    "F KK, F Q", "F a_temp",
    "a_temp += KK * Q",
    "fhs_add_source_term_modes")

_ek_add_bottom_surface_source = cp.ElementwiseKernel(
    "F KK, F Cp_bottom, F B", "F a_temp",
    "a_temp += KK * Cp_bottom * B",
    "fhs_add_bottom_surface_source")

_ek_evaporation_flux = cp.ElementwiseKernel(
    "F T_surface, float64 P0, float64 T_boil, float64 DeltaH_LV,"
    " float64 R_v, float64 T_liquidus",
    "F q_out",
    """
    double T = (double)T_surface;
    if (T < T_liquidus) {
        q_out = 0;
    } else {
        double factor1 = 0.82 * DeltaH_LV * P0 / sqrt(2.0 * M_PI * R_v);
        double factor2 = DeltaH_LV / (R_v * T_boil);
        double term = (1.0 / sqrt(T)) * exp(factor2 * (1.0 - T_boil / T));
        q_out = factor1 * term;
    }
    """, "fhs_evaporation_flux")

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

    def __init__(self, phys, geom, num, fine):
        super().__init__(phys, geom, num, fine, xp=cp)
        # CuPy/CUDA primitives for the shared free functions in spectral_ops.
        self.hooks = _state.BackendHooks(
            idct=IDCT_II,
            ndshift=_ndshift,
            source_term=_source_term,
            dct=DCT_II,
            corr_source=_correction_source,
        )


# ======================================
# Wrapper Functions
# ======================================

def _launch_config(shape, tpb=None):
    """Compute (blockspergrid, threadsperblock) for a CUDA kernel launch."""
    if tpb is None:
        tpb = (8,) * len(shape)
    return tuple((d + t - 1) // t for d, t in zip(shape, tpb, strict=True)), tpb


def update_modes_etd1(aK, KK, Cp_broadcast, B_scaled, a_temp_out):
    """Wrapper for ETD1 kernel."""
    _ek_update_modes_etd1(aK, KK, Cp_broadcast, B_scaled, a_temp_out)


def add_source_term_modes(a_temp, KK, Q_modes):
    """Wrapper for Source Term Accumulation."""
    _ek_add_source_term_modes(KK, Q_modes, a_temp)


def add_bottom_surface_source(a_temp, KK, Cp_broadcast_bottom, B_scaled):
    """Wrapper for bottom surface source accumulation (GPU)."""
    _ek_add_bottom_surface_source(KK, Cp_broadcast_bottom, B_scaled, a_temp)


@cuda.jit
def _compute_source_term_from_temperature_kernel(T_curr, T_prev, T_S, T_L, rho, L, dt, out):
    """
    Compute Q = - rho * L * (f_l(T_curr) - f_l(T_prev)) / dt, with the liquid
    fraction f_l clamped to [0, 1].
    Used for latent heat calculation. GPU version (CUDA kernel).
    """
    z, y, x = cuda.grid(3)
    nz, ny, nx = T_curr.shape
    if z < nz and y < ny and x < nx:
        inv_band = 1.0 / (T_L - T_S)

        f_c = (T_curr[z, y, x] - T_S) * inv_band
        if f_c < 0.0:
            f_c = 0.0
        elif f_c > 1.0:
            f_c = 1.0

        f_p = (T_prev[z, y, x] - T_S) * inv_band
        if f_p < 0.0:
            f_p = 0.0
        elif f_p > 1.0:
            f_p = 1.0

        out[z, y, x] = (-rho * L / dt) * (f_c - f_p)


def compute_source_term_from_temperature(T_curr, T_prev, T_S, T_L, rho, L, dt, out):
    """Wrapper for the latent-heat source-term kernel (host-callable)."""
    blockspergrid, threadsperblock = _launch_config(T_curr.shape)
    _compute_source_term_from_temperature_kernel[blockspergrid, threadsperblock](
        T_curr, T_prev, T_S, T_L, rho, L, dt, out
    )


def compute_evaporation_flux(T_surface, q_out, P0, T_boil, DeltaH_LV, R_v, T_liquidus):
    """Wrapper for Evaporation kernel."""
    _ek_evaporation_flux(
        T_surface, float(P0), float(T_boil), float(DeltaH_LV),
        float(R_v), float(T_liquidus), q_out,
    )


def compute_gaussian_laser_flux(X, Y, laser_x, laser_y, laser_r, laser_coef):
    """Gaussian (order-2 super-Gaussian) flux on the 1-D CuPy grid ``(X, Y)``.

    Thin backend wrapper over the shared, backend-agnostic
    :func:`~spectral_galerkin_heat.core.laser.super_gaussian_flux` (precision-transparent:
    it follows ``X.dtype``). The spectral solver calls the profile directly; this
    is kept for the linear solver's Gaussian-only path.
    """
    return _super_gaussian_flux(cp, X, Y, laser_x, laser_y, laser_r, laser_r, 2.0, laser_coef)



# Backend-agnostic free functions live in ``spectral_ops``; re-exported here so
# callers keep using ``spectral_gpu_kernels.<fn>``.
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


def _ndshift(field, shift_pixels, order, mode, cval):
    """ndimage shift primitive (GPU): cupyx.scipy.ndimage into a fresh buffer."""
    out = cp.empty_like(field)
    cupy_ndimage.shift(field, shift_pixels, order=order, mode=mode, cval=cval, output=out)
    return out


def _source_term(T_curr, T_prev, T_S, T_L, rho, L, dt, out):
    """Latent-heat source primitive (GPU): launch the CUDA source-term kernel."""
    blockspergrid, threadsperblock = _launch_config(T_curr.shape)
    _compute_source_term_from_temperature_kernel[blockspergrid, threadsperblock](
        T_curr, T_prev, T_S, T_L, rho, L, dt, out
    )


# ``f = ∇·(k'∇T) - a'(T)(T - T_prev)/dt`` in a single pass.


_CORR_PREAMBLE = r"""
template <typename F>
__device__ __forceinline__ F g_x_at(const F* T, const F* kp,
        int z, int y, int x, int nz, int ny, int nx, F inv_dx) {
    const int base = (z * ny + y) * nx;
    F dtx;
    if (x == 0)            dtx = (T[base + 1] - T[base + 0]) * inv_dx;
    else if (x == nx - 1)  dtx = (T[base + nx - 1] - T[base + nx - 2]) * inv_dx;
    else                   dtx = (T[base + x + 1] - T[base + x - 1]) * (F(0.5) * inv_dx);
    return kp[base + x] * dtx;
}
template <typename F>
__device__ __forceinline__ F g_y_at(const F* T, const F* kp,
        int z, int y, int x, int nz, int ny, int nx, F inv_dy) {
    const int zb = z * ny * nx;
    F dty;
    if (y == 0)            dty = (T[zb + 1 * nx + x] - T[zb + 0 * nx + x]) * inv_dy;
    else if (y == ny - 1)  dty = (T[zb + (ny - 1) * nx + x] - T[zb + (ny - 2) * nx + x]) * inv_dy;
    else                   dty = (T[zb + (y + 1) * nx + x] - T[zb + (y - 1) * nx + x]) * (F(0.5) * inv_dy);
    return kp[zb + y * nx + x] * dty;
}
template <typename F>
__device__ __forceinline__ F g_z_at(const F* T, const F* kp,
        int z, int y, int x, int nz, int ny, int nx, F inv_dz) {
    const int p = y * nx + x, s = ny * nx;
    F dtz;
    if (z == 0)            dtz = (T[1 * s + p] - T[0 * s + p]) * inv_dz;
    else if (z == nz - 1)  dtz = (T[(nz - 1) * s + p] - T[(nz - 2) * s + p]) * inv_dz;
    else                   dtz = (T[(z + 1) * s + p] - T[(z - 1) * s + p]) * (F(0.5) * inv_dz);
    return kp[z * s + p] * dtz;
}
"""

_ek_corr_source = cp.ElementwiseKernel(
    "raw F T, raw F kp, raw F ap, raw F Tprev,"
    " F inv_dx, F inv_dy, F inv_dz, F inv_dt,"
    " int32 nz, int32 ny, int32 nx",
    "F out",
    """
    const int x = i % nx;
    const int y = (i / nx) % ny;
    const int z = i / (nx * ny);
    F dgx, dgy, dgz;
    if (x == 0)
        dgx = (g_x_at(&T[0], &kp[0], z, y, 1, nz, ny, nx, inv_dx)
             - g_x_at(&T[0], &kp[0], z, y, 0, nz, ny, nx, inv_dx)) * inv_dx;
    else if (x == nx - 1)
        dgx = (g_x_at(&T[0], &kp[0], z, y, nx - 1, nz, ny, nx, inv_dx)
             - g_x_at(&T[0], &kp[0], z, y, nx - 2, nz, ny, nx, inv_dx)) * inv_dx;
    else
        dgx = (g_x_at(&T[0], &kp[0], z, y, x + 1, nz, ny, nx, inv_dx)
             - g_x_at(&T[0], &kp[0], z, y, x - 1, nz, ny, nx, inv_dx)) * (F(0.5) * inv_dx);
    if (y == 0)
        dgy = (g_y_at(&T[0], &kp[0], z, 1, x, nz, ny, nx, inv_dy)
             - g_y_at(&T[0], &kp[0], z, 0, x, nz, ny, nx, inv_dy)) * inv_dy;
    else if (y == ny - 1)
        dgy = (g_y_at(&T[0], &kp[0], z, ny - 1, x, nz, ny, nx, inv_dy)
             - g_y_at(&T[0], &kp[0], z, ny - 2, x, nz, ny, nx, inv_dy)) * inv_dy;
    else
        dgy = (g_y_at(&T[0], &kp[0], z, y + 1, x, nz, ny, nx, inv_dy)
             - g_y_at(&T[0], &kp[0], z, y - 1, x, nz, ny, nx, inv_dy)) * (F(0.5) * inv_dy);
    if (z == 0)
        dgz = (g_z_at(&T[0], &kp[0], 1, y, x, nz, ny, nx, inv_dz)
             - g_z_at(&T[0], &kp[0], 0, y, x, nz, ny, nx, inv_dz)) * inv_dz;
    else if (z == nz - 1)
        dgz = (g_z_at(&T[0], &kp[0], nz - 1, y, x, nz, ny, nx, inv_dz)
             - g_z_at(&T[0], &kp[0], nz - 2, y, x, nz, ny, nx, inv_dz)) * inv_dz;
    else
        dgz = (g_z_at(&T[0], &kp[0], z + 1, y, x, nz, ny, nx, inv_dz)
             - g_z_at(&T[0], &kp[0], z - 1, y, x, nz, ny, nx, inv_dz)) * (F(0.5) * inv_dz);
    const F dTdt = (T[i] - Tprev[i]) * inv_dt;
    out = dgx + dgy + dgz - ap[i] * dTdt;
    """, "fhs_corr_source", preamble=_CORR_PREAMBLE)


def _correction_source(T, k_prime, a_prime, T_prev, dt, dx, dy, dz):
    """Divergence-form correction forcing ``f = ∇·(k'∇T) - a'∂_tT`` (GPU)."""
    nz, ny, nx = T.shape
    out = cp.empty_like(T)
    real_t = T.dtype.type
    _ek_corr_source(
        T, k_prime, a_prime, T_prev,
        real_t(1.0 / dx), real_t(1.0 / dy), real_t(1.0 / dz),
        real_t(1.0 / dt),
        cp.int32(nz), cp.int32(ny), cp.int32(nx), out,
    )
    return out


# ---------------------------------------------------------------------------
# Real-FFT-based DCT-II / DCT-III on the contiguous last axis.
# ---------------------------------------------------------------------------

# Keyed on (N, real dtype): the factors are always built in double and cast
# down, so a float32 and a float64 transform of the same length need separate
# entries.
_COMPLEX_OF = {np.dtype(np.float32): np.complex64,
               np.dtype(np.float64): np.complex128}

_DCT2_TWIDDLE = {}   # (N, dtype) -> (C, S) ortho recombine factors, length N
_DCT3_TWIDDLE = {}   # (N, dtype) -> (PA, QA, PB, QB) factors, length N//2+1


def _dct2_twiddle(N, real_t):
    t = _DCT2_TWIDDLE.get((N, real_t))
    if t is None:
        k = cp.arange(N, dtype=cp.float64)
        ang = (math.pi / (2.0 * N)) * k
        w = cp.full(N, math.sqrt(2.0 / N))
        w[0] = math.sqrt(1.0 / N)
        t = ((w * cp.cos(ang)).astype(real_t),
             (w * cp.sin(ang)).astype(real_t))
        _DCT2_TWIDDLE[(N, real_t)] = t
    return t


def _dct3_twiddle(N, real_t):
    t = _DCT3_TWIDDLE.get((N, real_t))
    if t is None:
        L = N // 2 + 1
        k = cp.arange(L, dtype=cp.float64)
        ang = (math.pi / (2.0 * N)) * k
        cosk = cp.cos(ang)
        sink = cp.sin(ang)
        g = cp.full(L, math.sqrt(2.0 * N))
        g[0] = math.sqrt(4.0 * N)
        gnk = cp.full(L, math.sqrt(2.0 * N))
        PA = 0.5 * cosk * g
        QA = 0.5 * sink * gnk
        PB = 0.5 * sink * g
        QB = -0.5 * cosk * gnk
        QA[0] = 0.0          # k=0 has no conjugate partner
        QB[0] = 0.0
        t = tuple(z.astype(real_t) for z in (PA, QA, PB, QB))
        _DCT3_TWIDDLE[(N, real_t)] = t
    return t


_ek_dct2_reorder = cp.ElementwiseKernel(
    "raw F x, int32 N", "F v",
    """
    int r = i / N, m = i % N;
    int src = (m < (N + 1) / 2) ? (2 * m) : (2 * N - 2 * m - 1);
    v = x[r * N + src];
    """, "fhs_dct2_reorder")

_ek_dct2_recombine = cp.ElementwiseKernel(
    "raw C V, raw F Cf, raw F S, int32 N, int32 L", "F X",
    """
    int r = i / N, k = i % N, M = N / 2;
    C z; F ar, ai;
    if (k <= M) { z = V[r * L + k];       ar = z.real(); ai =  z.imag(); }
    else        { z = V[r * L + (N - k)]; ar = z.real(); ai = -z.imag(); }
    X = ar * Cf[k] + ai * S[k];
    """, "fhs_dct2_recombine")

_ek_dct3_prerecombine = cp.ElementwiseKernel(
    "raw F X, raw F PA, raw F QA, raw F PB, raw F QB,"
    " int32 N, int32 L", "C V",
    """
    int r = i / L, k = i % L;
    int nk = (k == 0) ? 0 : (N - k);
    F xk = X[r * N + k], xnk = X[r * N + nk];
    V = C(PA[k] * xk + QA[k] * xnk, PB[k] * xk + QB[k] * xnk);
    """, "fhs_dct3_prerecombine")

_ek_dct3_unreorder = cp.ElementwiseKernel(
    "raw F v, int32 N", "F x",
    """
    int r = i / N, n = i % N;
    int src = (n % 2 == 0) ? (n / 2) : (N - (n + 1) / 2);
    x = v[r * N + src];
    """, "fhs_dct3_unreorder")


def _dct2_last(x2):
    """DCT-II (ortho) along the last axis of a contiguous (R, N) real array."""
    N = x2.shape[1]
    v = cp.empty_like(x2)
    _ek_dct2_reorder(x2, cp.int32(N), v)
    V = cp.fft.rfft(v, axis=-1)
    C, S = _dct2_twiddle(N, x2.dtype)
    X = cp.empty_like(x2)
    _ek_dct2_recombine(V, C, S, cp.int32(N), cp.int32(N // 2 + 1), X)
    return X


def _dct3_last(x2):
    """DCT-III (ortho, inverse of II) along the last axis of a real (R, N) array."""
    R, N = x2.shape
    L = N // 2 + 1
    real_t = x2.dtype
    PA, QA, PB, QB = _dct3_twiddle(N, real_t)
    V = cp.empty((R, L), _COMPLEX_OF[real_t])
    _ek_dct3_prerecombine(x2, PA, QA, PB, QB, cp.int32(N), cp.int32(L), V)
    v = cp.fft.irfft(V, n=N, axis=-1)        # complex -> contiguous real
    out = cp.empty((R, N), real_t)
    _ek_dct3_unreorder(v.astype(real_t, copy=False), cp.int32(N), out)
    return out


def _dctn_contig(x, dct_type):
    """Separable n-D DCT done one axis at a time, each on the contiguous layout.
    """
    last = _dct2_last if dct_type == 2 else _dct3_last
    for _ in range(x.ndim):
        x = cp.ascontiguousarray(cp.moveaxis(x, 0, -1))
        shp = x.shape
        x = last(x.reshape(-1, shp[-1])).reshape(shp)
    return x


def DCT_II(q):
    """Apply Discrete Cosine Transform Type II (Ortho) on GPU."""
    return _dctn_contig(q, 2)

def IDCT_II(a):
    """Apply Discrete Cosine Transform Type III (Inverse Ortho) on GPU."""
    return _dctn_contig(a, 3)


def shift_flux(field: cp.ndarray, shift: tuple, geom) -> cp.ndarray:
    """Translate a surface flux field by ``shift=(dx, dy)`` meters on GPU."""
    dx, dy = shift
    shift_pixels = (dy / geom.d.y, dx / geom.d.x)
    return _ndshift(field, shift_pixels, order=1, mode='constant', cval=0.0)