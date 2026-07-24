"""GPU-based spectral method kernels for the heat equation (requires CuPy).

Public functions in this module mirror CPU kernels and are called by
SpectralSolver (CupyBackend).
"""

# Copyright 2026 Laboratoire de Mécanique des Solides (LMS), 
# École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris, 
# Route de Saclay, Palaiseau, 91128, France.
# 
# Author: Théo Andrieux, Jules Dichamp, Manas V. Upadhyay, Jules Dichamp, Manas V. Upadhyay
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
from numba import cuda
import math

from fast_heat_solv.physics import spectral_state as _state
from fast_heat_solv.physics import spectral_ops as _ops
from fast_heat_solv.core.laser import super_gaussian_flux as _super_gaussian_flux

__all__ = [
    "SpectralSolverState",
    "SpectralGrid",
    "FineMeshState",
    "update_modes_etd1",
    "add_source_term_modes",
    "add_bottom_surface_source",
    "compute_latent_heat_source",
    "compute_latent_heat_source_grid",
    "reconstruct_surface_temperature",
    "reconstruct_bottom_temperature",
]

# ======================================
# CUDA Kernels (Device Functions)
# ======================================

# These are CuPy ``ElementwiseKernel``s rather than numba ``@cuda.jit`` kernels:
# see the note above the DCT stages for why a numba launch handed a CuPy array
# costs ~0.85 ms. CuPy broadcasts the operands (``Cp_broadcast`` is (nz,1,1),
# ``B_scaled`` is (ny,nx)) the same way the explicit indexing did, and each
# writes into a caller-owned buffer, so no full-grid temporary is created.
# Operand grouping is kept exactly as the numba kernels had it, so results are
# bit-identical.

_ek_update_modes_etd1 = cp.ElementwiseKernel(
    "float32 aK, float32 KK, float32 Cp, float32 B", "float32 out",
    "out = aK + (KK * Cp) * B",
    "fhs_update_modes_etd1")

_ek_add_source_term_modes = cp.ElementwiseKernel(
    "float32 KK, float32 Q", "float32 a_temp",
    "a_temp += KK * Q",
    "fhs_add_source_term_modes")

_ek_add_bottom_surface_source = cp.ElementwiseKernel(
    "float32 KK, float32 Cp_bottom, float32 B", "float32 a_temp",
    "a_temp += KK * Cp_bottom * B",
    "fhs_add_bottom_surface_source")

# Evaporation: the numba kernel promoted to double via ``math.sqrt``/``math.exp``
# on float64 scalar args and stored back to float32, so the arithmetic is done in
# double here too.
_ek_evaporation_flux = cp.ElementwiseKernel(
    "float32 T_surface, float64 P0, float64 T_boil, float64 DeltaH_LV,"
    " float64 R_v, float64 T_liquidus",
    "float32 q_out",
    """
    double T = (double)T_surface;
    if (T < T_liquidus) {
        q_out = 0.0f;
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
    return tuple((d + t - 1) // t for d, t in zip(shape, tpb)), tpb


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
    :func:`~fast_heat_solv.core.laser.super_gaussian_flux` (precision-transparent:
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
#
# An ElementwiseKernel over the output (`i` is its flat index, decomposed back to
# z,y,x) rather than a numba `@cuda.jit` stencil: the numba launch spent ~0.72 ms
# of its 3.11 ms per call just adopting the CuPy arrays (see the note above the
# DCT stages). The `g_*_at` helpers are the same one-sided/central branches the
# numba device functions had, in the same order, so the result is bit-identical.
# The flat index keeps the contiguous x axis on the fastest-varying thread index,
# so the stencil's ±1 reads still coalesce.

_CORR_PREAMBLE = r"""
__device__ __forceinline__ float g_x_at(const float* T, const float* kp,
        int z, int y, int x, int nz, int ny, int nx, float inv_dx) {
    const int base = (z * ny + y) * nx;
    float dtx;
    if (x == 0)            dtx = (T[base + 1] - T[base + 0]) * inv_dx;
    else if (x == nx - 1)  dtx = (T[base + nx - 1] - T[base + nx - 2]) * inv_dx;
    else                   dtx = (T[base + x + 1] - T[base + x - 1]) * (0.5f * inv_dx);
    return kp[base + x] * dtx;
}
__device__ __forceinline__ float g_y_at(const float* T, const float* kp,
        int z, int y, int x, int nz, int ny, int nx, float inv_dy) {
    const int zb = z * ny * nx;
    float dty;
    if (y == 0)            dty = (T[zb + 1 * nx + x] - T[zb + 0 * nx + x]) * inv_dy;
    else if (y == ny - 1)  dty = (T[zb + (ny - 1) * nx + x] - T[zb + (ny - 2) * nx + x]) * inv_dy;
    else                   dty = (T[zb + (y + 1) * nx + x] - T[zb + (y - 1) * nx + x]) * (0.5f * inv_dy);
    return kp[zb + y * nx + x] * dty;
}
__device__ __forceinline__ float g_z_at(const float* T, const float* kp,
        int z, int y, int x, int nz, int ny, int nx, float inv_dz) {
    const int p = y * nx + x, s = ny * nx;
    float dtz;
    if (z == 0)            dtz = (T[1 * s + p] - T[0 * s + p]) * inv_dz;
    else if (z == nz - 1)  dtz = (T[(nz - 1) * s + p] - T[(nz - 2) * s + p]) * inv_dz;
    else                   dtz = (T[(z + 1) * s + p] - T[(z - 1) * s + p]) * (0.5f * inv_dz);
    return kp[z * s + p] * dtz;
}
"""

_ek_corr_source = cp.ElementwiseKernel(
    "raw float32 T, raw float32 kp, raw float32 ap, raw float32 Tprev,"
    " float32 inv_dx, float32 inv_dy, float32 inv_dz, float32 inv_dt,"
    " int32 nz, int32 ny, int32 nx",
    "float32 out",
    """
    const int x = i % nx;
    const int y = (i / nx) % ny;
    const int z = i / (nx * ny);
    float dgx, dgy, dgz;
    if (x == 0)
        dgx = (g_x_at(&T[0], &kp[0], z, y, 1, nz, ny, nx, inv_dx)
             - g_x_at(&T[0], &kp[0], z, y, 0, nz, ny, nx, inv_dx)) * inv_dx;
    else if (x == nx - 1)
        dgx = (g_x_at(&T[0], &kp[0], z, y, nx - 1, nz, ny, nx, inv_dx)
             - g_x_at(&T[0], &kp[0], z, y, nx - 2, nz, ny, nx, inv_dx)) * inv_dx;
    else
        dgx = (g_x_at(&T[0], &kp[0], z, y, x + 1, nz, ny, nx, inv_dx)
             - g_x_at(&T[0], &kp[0], z, y, x - 1, nz, ny, nx, inv_dx)) * (0.5f * inv_dx);
    if (y == 0)
        dgy = (g_y_at(&T[0], &kp[0], z, 1, x, nz, ny, nx, inv_dy)
             - g_y_at(&T[0], &kp[0], z, 0, x, nz, ny, nx, inv_dy)) * inv_dy;
    else if (y == ny - 1)
        dgy = (g_y_at(&T[0], &kp[0], z, ny - 1, x, nz, ny, nx, inv_dy)
             - g_y_at(&T[0], &kp[0], z, ny - 2, x, nz, ny, nx, inv_dy)) * inv_dy;
    else
        dgy = (g_y_at(&T[0], &kp[0], z, y + 1, x, nz, ny, nx, inv_dy)
             - g_y_at(&T[0], &kp[0], z, y - 1, x, nz, ny, nx, inv_dy)) * (0.5f * inv_dy);
    if (z == 0)
        dgz = (g_z_at(&T[0], &kp[0], 1, y, x, nz, ny, nx, inv_dz)
             - g_z_at(&T[0], &kp[0], 0, y, x, nz, ny, nx, inv_dz)) * inv_dz;
    else if (z == nz - 1)
        dgz = (g_z_at(&T[0], &kp[0], nz - 1, y, x, nz, ny, nx, inv_dz)
             - g_z_at(&T[0], &kp[0], nz - 2, y, x, nz, ny, nx, inv_dz)) * inv_dz;
    else
        dgz = (g_z_at(&T[0], &kp[0], z + 1, y, x, nz, ny, nx, inv_dz)
             - g_z_at(&T[0], &kp[0], z - 1, y, x, nz, ny, nx, inv_dz)) * (0.5f * inv_dz);
    const float dTdt = (T[i] - Tprev[i]) * inv_dt;
    out = dgx + dgy + dgz - ap[i] * dTdt;
    """, "fhs_corr_source", preamble=_CORR_PREAMBLE)


def _correction_source(T, k_prime, a_prime, T_prev, dt, dx, dy, dz):
    """Divergence-form correction forcing ``f = ∇·(k'∇T) - a'∂_tT`` (GPU)."""
    nz, ny, nx = T.shape
    out = cp.empty_like(T)
    _ek_corr_source(
        T, k_prime, a_prime, T_prev,
        cp.float32(1.0 / dx), cp.float32(1.0 / dy), cp.float32(1.0 / dz),
        cp.float32(1.0 / dt),
        cp.int32(nz), cp.int32(ny), cp.int32(nx), out,
    )
    return out


# ---------------------------------------------------------------------------
# Real-FFT-based DCT-II / DCT-III on the contiguous last axis.
#
# cupyx's ``dct`` is ~6× heavier than the underlying real FFT (9.2 ms vs 1.4 ms
# for a 512-long contiguous axis) — it does the pre/post-processing in generic
# strided passes. The approach here computes an N-point DCT from one N-point
# real FFT plus an even/odd reorder and a twiddle recombine; doing those two
# steps in fused, coalesced kernels (with the twiddle factors cached per length)
# beats cupyx's ``dct`` by ~30 % for both the forward (II) and inverse (III).
# This real-FFT-based construction is the standard way to build real-to-real
# transforms since cuFFT has no native DCT.
# ---------------------------------------------------------------------------

_DCT2_TWIDDLE = {}   # N -> (C, S) ortho recombine factors, length N
_DCT3_TWIDDLE = {}   # N -> (PA, QA, PB, QB) inverse factors, length N//2+1


def _dct2_twiddle(N):
    t = _DCT2_TWIDDLE.get(N)
    if t is None:
        k = cp.arange(N, dtype=cp.float64)
        ang = (math.pi / (2.0 * N)) * k
        w = cp.full(N, math.sqrt(2.0 / N)); w[0] = math.sqrt(1.0 / N)
        t = ((w * cp.cos(ang)).astype(cp.float32),
             (w * cp.sin(ang)).astype(cp.float32))
        _DCT2_TWIDDLE[N] = t
    return t


def _dct3_twiddle(N):
    t = _DCT3_TWIDDLE.get(N)
    if t is None:
        L = N // 2 + 1
        k = cp.arange(L, dtype=cp.float64)
        ang = (math.pi / (2.0 * N)) * k
        cosk = cp.cos(ang); sink = cp.sin(ang)
        g = cp.full(L, math.sqrt(2.0 * N)); g[0] = math.sqrt(4.0 * N)
        gnk = cp.full(L, math.sqrt(2.0 * N))
        PA = 0.5 * cosk * g;  QA = 0.5 * sink * gnk
        PB = 0.5 * sink * g;  QB = -0.5 * cosk * gnk
        QA[0] = 0.0; QB[0] = 0.0          # k=0 has no conjugate partner
        t = tuple(z.astype(cp.float32) for z in (PA, QA, PB, QB))
        _DCT3_TWIDDLE[N] = t
    return t


# The DCT stages are CuPy ``ElementwiseKernel``s rather than numba ``@cuda.jit``
# kernels. Numba re-adopts a CuPy array through ``__cuda_array_interface__`` on
# every launch, and CuPy exports CAI v3 with ``stream=1``, whose semantics oblige
# numba to synchronise as it adopts the array. That serialises the pipeline and
# costs ~0.85 ms per launch — against ~35 us for the same launch from CuPy, and
# ~45 us of real work for a 2-D surface transform. The stages below are plain
# gathers/recombines, so they port to ElementwiseKernel bit-exactly (`i` is the
# flat index of the non-raw output, which sets the launch range).

_ek_dct2_reorder = cp.ElementwiseKernel(
    "raw float32 x, int32 N", "float32 v",
    """
    int r = i / N, m = i % N;
    int src = (m < (N + 1) / 2) ? (2 * m) : (2 * N - 2 * m - 1);
    v = x[r * N + src];
    """, "fhs_dct2_reorder")

_ek_dct2_recombine = cp.ElementwiseKernel(
    "raw complex64 V, raw float32 C, raw float32 S, int32 N, int32 L", "float32 X",
    """
    int r = i / N, k = i % N, M = N / 2;
    complex<float> z; float ar, ai;
    if (k <= M) { z = V[r * L + k];       ar = z.real(); ai =  z.imag(); }
    else        { z = V[r * L + (N - k)]; ar = z.real(); ai = -z.imag(); }
    X = ar * C[k] + ai * S[k];
    """, "fhs_dct2_recombine")

_ek_dct3_prerecombine = cp.ElementwiseKernel(
    "raw float32 X, raw float32 PA, raw float32 QA, raw float32 PB, raw float32 QB,"
    " int32 N, int32 L", "complex64 V",
    """
    int r = i / L, k = i % L;
    int nk = (k == 0) ? 0 : (N - k);
    float xk = X[r * N + k], xnk = X[r * N + nk];
    V = complex<float>(PA[k] * xk + QA[k] * xnk, PB[k] * xk + QB[k] * xnk);
    """, "fhs_dct3_prerecombine")

_ek_dct3_unreorder = cp.ElementwiseKernel(
    "raw float32 v, int32 N", "float32 x",
    """
    int r = i / N, n = i % N;
    int src = (n % 2 == 0) ? (n / 2) : (N - (n + 1) / 2);
    x = v[r * N + src];
    """, "fhs_dct3_unreorder")


def _dct2_last(x2):
    """DCT-II (ortho) along the last axis of a contiguous (R, N) float32 array."""
    R, N = x2.shape
    v = cp.empty_like(x2)
    _ek_dct2_reorder(x2, cp.int32(N), v)
    V = cp.fft.rfft(v, axis=-1)
    C, S = _dct2_twiddle(N)
    X = cp.empty_like(x2)
    _ek_dct2_recombine(V, C, S, cp.int32(N), cp.int32(N // 2 + 1), X)
    return X


def _dct3_last(x2):
    """DCT-III (ortho, inverse of II) along the last axis of (R, N) float32."""
    R, N = x2.shape; L = N // 2 + 1
    PA, QA, PB, QB = _dct3_twiddle(N)
    V = cp.empty((R, L), cp.complex64)
    _ek_dct3_prerecombine(x2, PA, QA, PB, QB, cp.int32(N), cp.int32(L), V)
    v = cp.fft.irfft(V, n=N, axis=-1)        # complex64 -> contiguous float32
    out = cp.empty((R, N), cp.float32)
    _ek_dct3_unreorder(v.astype(cp.float32, copy=False), cp.int32(N), out)
    return out


def _dctn_contig(x, dct_type):
    """Separable n-D DCT done one axis at a time, each on the contiguous layout.

    cupyx's ``dctn`` transforms strided axes in place: the outermost axis of a
    512×256×450 volume costs ~100 ms vs ~9 ms contiguous, so the strided 3-D call
    is ~2.3× slower than necessary. The DCT is a tensor product, so moving each
    axis to the last (contiguous) position and transforming is bit-identical and
    far faster — the cost is then the transpose copies plus the FFT.

    The transposes are minimised by *cycling* rather than moving each axis back:
    ``moveaxis(x, 0, -1)`` rolls the axis labels, so applying it ``ndim`` times
    transforms every axis on the contiguous last position and returns the array
    to its original layout — 3 transpose copies instead of 6 (move-out +
    move-back per axis). The contiguous-axis transform itself is the fused
    real-FFT-based DCT (``_dct2_last`` / ``_dct3_last``), ~30 % faster than cupyx's.
    """
    last = _dct2_last if dct_type == 2 else _dct3_last
    for _ in range(x.ndim):
        x = cp.ascontiguousarray(cp.moveaxis(x, 0, -1))
        shp = x.shape
        x = last(x.reshape(-1, shp[-1])).reshape(shp)
    return x.astype(cp.float32, copy=False)


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