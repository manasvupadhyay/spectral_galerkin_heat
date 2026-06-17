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

    def __init__(self, phys, geom, num, fine):
        super().__init__(phys, geom, num, fine, xp=cp)
        # CuPy/CUDA primitives for the shared free functions in spectral_ops.
        self.hooks = _state.BackendHooks(
            idct=IDCT_II,
            ndshift=_ndshift,
            source_term=_source_term,
            dct=DCT_II,
            dct_axis=_dct_axis,
            dst_axis=_dst_axis,
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
    blockspergrid, threadsperblock = _launch_config(T_surface.shape, (16, 16))
    _compute_evaporation_flux_kernel[blockspergrid, threadsperblock](
        T_surface, q_out, P0, T_boil, DeltaH_LV, R_v, T_liquidus
    )


def compute_gaussian_laser_flux(X, Y, laser_x, laser_y, laser_r, laser_coef):
    """Compute Gaussian flux on grid defined by 1D CuPy arrays X, Y."""
    # dx: shape (1, nx), dy: shape (ny, 1)
    dx = X[None, :] - laser_x
    dy = Y[:, None] - laser_y
    r_sq = dx ** 2 + dy ** 2
    return (laser_coef * cp.exp(-2.0 * r_sq / (laser_r ** 2))).astype(cp.float32, copy=False)



# Backend-agnostic free functions live in ``spectral_ops``; re-exported here so
# callers keep using ``spectral_gpu_kernels.<fn>``.
project_box_to_modes = _ops.project_box_to_modes
_reconstruct_temperature_box = _ops.reconstruct_temperature_box
initialize_latent_heat_if_needed = _ops.initialize_latent_heat_if_needed
update_latent_heat_history = _ops.update_latent_heat_history
reconstruct_surface_temperature = _ops.reconstruct_surface_temperature
reconstruct_bottom_temperature = _ops.reconstruct_bottom_temperature
compute_latent_heat_source = _ops.compute_latent_heat_source
shift_latent_heat_history = _ops.shift_latent_heat_history
reconstruct_volume = _ops.reconstruct_volume
project_volume = _ops.project_volume
conductivity_correction_modes = _ops.conductivity_correction_modes
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


@cuda.jit
def _grad_kprime_kernel(T, kp, inv_dx, inv_dy, inv_dz, gx, gy, gz):
    """First pass of the divergence-form correction: ``g = k'(T) ∇T``.

    Central differences in the interior, first-order one-sided at the faces —
    bit-for-bit the ``numpy.gradient(.., edge_order=1)`` stencil used by the CPU
    fallback, fused with the ``k'`` multiply so ``∇T`` is never materialised.

    Thread layout maps the fastest thread index to the contiguous ``x`` axis so
    the ±1 neighbour reads coalesce (see ``_correction_source`` launch config).
    """
    x, y, z = cuda.grid(3)
    nz, ny, nx = T.shape
    if z < nz and y < ny and x < nx:
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
        k = kp[z, y, x]
        gx[z, y, x] = k * dtx
        gy[z, y, x] = k * dty
        gz[z, y, x] = k * dtz


@cuda.jit
def _div_minus_capacity_kernel(gx, gy, gz, ap, T, Tprev,
                               inv_dx, inv_dy, inv_dz, inv_dt, out):
    """Second pass: ``f = ∇·g - a'(T) (T - T_prev)/dt``.

    Same ``numpy.gradient`` stencil applied to ``g``, with the capacity source
    fused in so the whole real-space forcing is one extra read/write pass.
    """
    x, y, z = cuda.grid(3)
    nz, ny, nx = T.shape
    if z < nz and y < ny and x < nx:
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


def _correction_source(T, k_prime, a_prime, T_prev, dt, dx, dy, dz):
    """Divergence-form correction forcing ``f = ∇·(k'∇T) - a'∂_tT`` (GPU).

    Two fused stencil passes replacing the six ``xp.gradient`` calls of the
    backend-agnostic path: ``g = k'∇T`` then ``∇·g`` minus the capacity source.
    Bit-for-bit the ``numpy.gradient(edge_order=1)`` discretisation, ~5× cheaper
    (one read/write pass each instead of ``xp.gradient``'s strided slicing).
    """
    gx = cp.empty_like(T)
    gy = cp.empty_like(T)
    gz = cp.empty_like(T)
    out = cp.empty_like(T)
    # Map the fastest thread index (cuda.grid(3)[0]) to the contiguous x axis so
    # the stencil's ±1 reads coalesce; grid is sized (nx, ny, nz) accordingly.
    nz, ny, nx = T.shape
    tpb = (32, 8, 1)
    bpg = ((nx + tpb[0] - 1) // tpb[0],
           (ny + tpb[1] - 1) // tpb[1],
           (nz + tpb[2] - 1) // tpb[2])
    _grad_kprime_kernel[bpg, tpb](
        T, k_prime, 1.0 / dx, 1.0 / dy, 1.0 / dz, gx, gy, gz)
    _div_minus_capacity_kernel[bpg, tpb](
        gx, gy, gz, a_prime, T, T_prev,
        1.0 / dx, 1.0 / dy, 1.0 / dz, 1.0 / dt, out)
    return out


# ---------------------------------------------------------------------------
# Makhoul DCT-II / DCT-III on the contiguous last axis.
#
# cupyx's ``dct`` is ~6× heavier than the underlying real FFT (9.2 ms vs 1.4 ms
# for a 512-long contiguous axis) — it does the pre/post-processing in generic
# strided passes. Makhoul's algorithm computes an N-point DCT from one N-point
# real FFT plus an even/odd reorder and a twiddle recombine; doing those two
# steps in fused, coalesced kernels (with the twiddle factors cached per length)
# beats cupyx's ``dct`` by ~30 % for both the forward (II) and inverse (III).
# Refs: Makhoul 1980; GPU spectral solvers (CaNS, arXiv:2001.05234) build their
# real-to-real transforms the same way since cuFFT has no native DCT.
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


@cuda.jit
def _dct2_reorder_kernel(x, v, N):
    """Even/odd reorder v = [x0,x2,..,  x(odd reversed)] over rows (R, N)."""
    idx = cuda.grid(1); R = x.shape[0]
    if idx < R * N:
        r = idx // N; m = idx % N
        if m < (N + 1) // 2:
            v[r, m] = x[r, 2 * m]
        else:
            v[r, m] = x[r, 2 * N - 2 * m - 1]


@cuda.jit
def _dct2_recombine_kernel(Vr, Vi, C, S, X, N):
    """X[k] = Re(W^k V[k])·2·norm, V[k>N/2] via conjugate symmetry."""
    idx = cuda.grid(1); R = X.shape[0]
    if idx < R * N:
        r = idx // N; k = idx % N; M = N // 2
        if k <= M:
            ar = Vr[r, k];     ai = Vi[r, k]
        else:
            ar = Vr[r, N - k]; ai = -Vi[r, N - k]
        X[r, k] = ar * C[k] + ai * S[k]


@cuda.jit
def _dct3_prerecombine_kernel(X, PA, QA, PB, QB, V, N):
    """Rebuild the half-spectrum V[k]=a+ib, k=0..N/2, from the (k, N-k) pair."""
    idx = cuda.grid(1); R = X.shape[0]; L = N // 2 + 1
    if idx < R * L:
        r = idx // L; k = idx % L
        nk = 0 if k == 0 else N - k
        xk = X[r, k]; xnk = X[r, nk]
        V[r, k] = complex(PA[k] * xk + QA[k] * xnk,
                          PB[k] * xk + QB[k] * xnk)


@cuda.jit
def _dct3_unreorder_kernel(v, x, N):
    """Inverse of the even/odd reorder: scatter v back to natural order."""
    idx = cuda.grid(1); R = x.shape[0]
    if idx < R * N:
        r = idx // N; n = idx % N
        if n % 2 == 0:
            x[r, n] = v[r, n // 2]
        else:
            x[r, n] = v[r, N - (n + 1) // 2]


_DCT_TPB = 256


def _dct_grid(total):
    return (total + _DCT_TPB - 1) // _DCT_TPB, _DCT_TPB


def _dct2_last(x2):
    """DCT-II (ortho) along the last axis of a contiguous (R, N) float32 array."""
    R, N = x2.shape
    v = cp.empty_like(x2)
    bpg, tpb = _dct_grid(R * N)
    _dct2_reorder_kernel[bpg, tpb](x2, v, N)
    V = cp.fft.rfft(v, axis=-1)
    Vr = cp.ascontiguousarray(V.real); Vi = cp.ascontiguousarray(V.imag)
    C, S = _dct2_twiddle(N)
    X = cp.empty_like(x2)
    _dct2_recombine_kernel[bpg, tpb](Vr, Vi, C, S, X, N)
    return X


def _dct3_last(x2):
    """DCT-III (ortho, inverse of II) along the last axis of (R, N) float32."""
    R, N = x2.shape; L = N // 2 + 1
    PA, QA, PB, QB = _dct3_twiddle(N)
    V = cp.empty((R, L), cp.complex64)
    bl, tpb = _dct_grid(R * L)
    _dct3_prerecombine_kernel[bl, tpb](x2, PA, QA, PB, QB, V, N)
    v = cp.fft.irfft(V, n=N, axis=-1)        # complex64 -> contiguous float32
    out = cp.empty((R, N), cp.float32)
    bpg, _ = _dct_grid(R * N)
    _dct3_unreorder_kernel[bpg, tpb](v, out, N)
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
    Makhoul DCT (``_dct2_last`` / ``_dct3_last``), ~30 % faster than cupyx's.
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


def _dct_axis(arr, axis):
    """One-axis forward DCT-II (ortho) on GPU — property-correction mixed transform.

    ``cupyx.scipy.fft`` mirrors the ``scipy.fft`` API, so this is the same code
    path as the CPU binding (property_correction.tex §6.3).
    """
    return cupy_fft.dct(arr, type=2, axis=axis, norm='ortho').astype(cp.float32, copy=False)


def _dst_axis(arr, axis):
    """One-axis forward DST-II (ortho) on GPU — differentiated axis of C^k."""
    return cupy_fft.dst(arr, type=2, axis=axis, norm='ortho').astype(cp.float32, copy=False)


def shift_flux(field: cp.ndarray, shift: tuple, geom) -> cp.ndarray:
    """Translate a surface flux field by ``shift=(dx, dy)`` meters on GPU."""
    dx, dy = shift
    shift_pixels = (dy / geom.d.y, dx / geom.d.x)
    return _ndshift(field, shift_pixels, order=1, mode='constant', cval=0.0)