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

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cupy_ndimage
import cupyx.scipy.fft as cupy_fft
from numba import cuda
import math
from dataclasses import dataclass
from fast_heat_solv.physics import spectral_helpers as spec_hp

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


@dataclass
class SpectralGrid:
    """Immutable grid definitions and reconstruction bases."""
    # Global coordinates (Cell-Centered)
    x: cp.ndarray = None
    y: cp.ndarray = None
    z: cp.ndarray = None
    
    # Derived 2D grids (X, Y are redundant but kept if heavily used, though we should prefer 1D)
    # Removing X, Y as per plan to reduce memory if they are just meshgrids of x, y

    # Reconstruction constants
    recon_scale: float = 0.0
    Cp32_broadcast: cp.ndarray = None
    Cp32_broadcast_bottom: cp.ndarray = None
    dct_scale: float = 0.0
    
    # Normalization coefficients (indexed by axis: 0=x, 1=y, 2=z)
    C: tuple = None

    # Lazy-loaded full reconstruction bases and node-centered coords (indexed by axis)
    B_recon: list = None
    coords_rec: list = None
    
    def __init__(self, geom):
        # Global mesh coordinates (Cell-Centered)
        dx, dy, dz = geom.dx, geom.dy, geom.dz
        nx, ny, nz = geom.nx, geom.ny, geom.nz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
        
        self.x, self.y, self.z = [((cp.arange(n) + 0.5) * d).astype(cp.float32)
                                   for n, d in zip((nx, ny, nz), (dx, dy, dz))]
        
        # Normalization coefficients (C[0]=x, C[1]=y, C[2]=z)
        self.C = tuple(cp.asarray(spec_hp._C_coef(n, L), dtype=cp.float32)
                       for n, L in zip((nx, ny, nz), (Lx, Ly, Lz)))

        # Scaling factors
        self.dct_scale = cp.float32((dx * dy) * np.sqrt((nx * ny) / (Lx * Ly)))
        self.recon_scale = cp.float32(np.sqrt(nx * ny) / np.sqrt(Lx * Ly))

        # Compute top-surface weighting for projection
        sign = cp.power(-1.0, cp.arange(nz, dtype=cp.float32)).astype(cp.float32)
        self.Cp32_broadcast = (self.C[2].astype(cp.float32) * sign)[:, None, None]

        # Bottom-surface weighting: cos(p*pi*0/Lz) = 1, so no sign alternation
        self.Cp32_broadcast_bottom = self.C[2].astype(cp.float32)[:, None, None]

    def prepare_full_reconstruction(self, geom):
        """Compute node-centered grids and full-domain reconstruction bases on demand.
        Kept on CPU since only used for final output and avoids GPU memory usage if not needed."""
        if self.B_recon is not None:
            return

        nx, ny, nz = geom.nx, geom.ny, geom.nz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
        dx, dy, dz = geom.dx, geom.dy, geom.dz

        # CPU-side numpy arrays (reconstruction only used for output)
        self.coords_rec = [((np.arange(n + 1)) * d).astype(np.float32)
                           for n, d in zip((nx, ny, nz), (dx, dy, dz))]
        C_cpu = [cp.asnumpy(self.C[a]) for a in range(3)]
        self.B_recon = [(C_cpu[a][:, None] * np.cos(np.pi * np.arange(n)[:, None] * self.coords_rec[a][None, :] / L)).astype(np.float32)
                        for a, (n, L) in enumerate(zip((nx, ny, nz), (Lx, Ly, Lz)))]

@dataclass
class FineMeshState:
    """Handles the moving fine mesh for latent heat/nonlinearities."""
    refinement: int = 4
    nx_box: int = 0
    ny_box: int = 0
    nz_box: int = 0
    n_fine_totals: list = None  # [nx_total, ny_total, nz_total]

    coords_fine: list = None    # [x_fine, y_fine, z_fine]
    dx_fine: float = 0.0
    dy_fine: float = 0.0
    dz_fine: float = 0.0
    dV_fine: float = 0.0

    B_fine_full: list = None    # cosine bases indexed by axis; z uses domain-shifted coords
    B_fine: list = None         # active window bases; z does not slide

    T_prev: cp.ndarray = None
    Q_prev: cp.ndarray = None

    def __init__(self, geom, grid: SpectralGrid):
        import logging
        logger = logging.getLogger(__name__)

        dx, dy, dz = geom.dx, geom.dy, geom.dz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz

        self.refinement = 4
        Lx_box, Ly_box, Lz_box = 0.9e-3, 0.9e-3, 0.04e-3

        self.dx_fine, self.dy_fine, self.dz_fine = dx/self.refinement, dy/self.refinement, dz/self.refinement
        d_fine = [self.dx_fine, self.dy_fine, self.dz_fine]

        self.n_fine_totals = [int(np.ceil(L / d)) for L, d in zip([Lx, Ly, Lz_box], d_fine)]
        self.coords_fine = [((cp.arange(n) + 0.5) * d).astype(cp.float32)
                            for n, d in zip(self.n_fine_totals, d_fine)]

        z_fine_global = (Lz - Lz_box) + self.coords_fine[2]

        logger.info("Precomputing fine cosine bases...")
        mode_counts = [geom.nx, geom.ny, geom.nz]
        L_domain = [Lx, Ly, Lz]
        fine_coords = [self.coords_fine[0], self.coords_fine[1], z_fine_global]
        self.B_fine_full = [
            (grid.C[a][:, None] * cp.cos(np.pi * cp.arange(mode_counts[a])[:, None] * fine_coords[a][None, :] / L_domain[a])).astype(cp.float32)
            for a in range(3)
        ]

        self.nx_box = min(int(np.ceil(Lx_box / self.dx_fine)), self.n_fine_totals[0])
        self.ny_box = min(int(np.ceil(Ly_box / self.dy_fine)), self.n_fine_totals[1])
        self.nz_box = min(int(np.ceil(Lz_box / self.dz_fine)), self.n_fine_totals[2])

        self.B_fine = [
            cp.zeros((geom.nx, self.nx_box), dtype=cp.float32),
            cp.zeros((geom.ny, self.ny_box), dtype=cp.float32),
            self.B_fine_full[2][:, :self.nz_box],  # z: static full-depth slice
        ]
        self.dV_fine = self.dx_fine * self.dy_fine * self.dz_fine

    def update(self, laser_state):
        """Update fine mesh basis subsets for x and y axes."""
        for a, (pos, d, n_box) in enumerate([
            (laser_state.x, self.dx_fine, self.nx_box),
            (laser_state.y, self.dy_fine, self.ny_box),
        ]):
            i_start, i_end, _ = _calculate_subgrid_indices(pos, d, self.n_fine_totals[a], n_box)
            self.B_fine[a][:, :] = self.B_fine_full[a][:, i_start:i_end]
        # z does not slide


@dataclass
class SolverBuffers:
    """Reusable working arrays."""
    a_temp: cp.ndarray = None      # (nz, ny, nx)
    q_evap_old: cp.ndarray = None  # (ny, nx)
    q_evap_buffer: cp.ndarray = None
    q_diff: cp.ndarray = None
    B_buffer: cp.ndarray = None
    Q_latent_buffer: cp.ndarray = None

    def __init__(self, num, fine_mesh: FineMeshState):
        nx , ny, nz = num.nx, num.ny, num.nz
        self.q_diff = cp.empty((ny, nx), dtype=cp.float32)
        self.B_buffer = cp.empty((ny, nx), dtype=cp.float32)
        self.a_temp = cp.empty((nz, ny, nx), dtype=cp.float32)
        self.q_evap_old = cp.zeros((ny, nx), dtype=cp.float32)
        self.q_evap_buffer = cp.zeros((ny, nx), dtype=cp.float32)
        if fine_mesh:
             self.Q_latent_buffer = cp.zeros((fine_mesh.nz_box, fine_mesh.ny_box, fine_mesh.nx_box), dtype=cp.float32)

@dataclass
class SpectralSolverState:
    """
    Coordinator class for the spectral method state.
    """
    # 1. Components
    grid: SpectralGrid = None
    buffers: SolverBuffers = None
    fine_mesh: FineMeshState = None
    
    # 2. Primary State
    a: cp.ndarray = None       # Current temperature modes (nz, ny, nx)
    
    # 3. Spectral Propagators
    K: cp.ndarray = None
    KK: cp.ndarray = None

    def __init__(self, phys, geom, num):
        # Initialize sub-components
        self.grid = SpectralGrid(geom)
        self.fine_mesh = FineMeshState(geom, self.grid)
        self.buffers = SolverBuffers(num, self.fine_mesh)
        
        # Precompute propagators
        self.K, self.KK = _precompute_K_KK(phys, num, geom)


def _precompute_K_KK(phys, num, geom):
    """
    Compute spectral Propagators (K, KK) based on grid and time step.
    K = exp(-alpha * k^2 * dt) for ETD1 (Exact integration of linear part)
    KK = phi_1 / (rho * Cp), where phi_1(z) = (exp(z) - 1) / z, z = -alpha * k^2 * dt
    """
    k = [np.pi * cp.arange(n) / L for n, L in zip((num.nz, num.ny, num.nx), (geom.Lz, geom.Ly, geom.Lx))]
    k_grids = cp.meshgrid(*k, indexing='ij')  # shape (nz, ny, nx) each
    denom = phys.k / (phys.rho * phys.Cp) * sum(kg**2 for kg in k_grids)
    K = cp.exp(-denom * num.dt).astype(cp.float32)
    mask_zero = (denom == 0)
    # Avoid div by zero
    denom[mask_zero] = 1.0

    phi_1 = (K - 1.0) / (-denom)
    phi_1[mask_zero] = num.dt

    KK = (phi_1 / (phys.rho * phys.Cp)).astype(cp.float32)
    return K, KK

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



def _calculate_subgrid_indices(pos, dx, n_total_fine, n_box):
    """
    Calculate start/end indices to center a box of size n_box around a physical position.
    
    Args:
        pos (float): Physical position (laser center).
        dx (float): Grid spacing.
        n_total_fine (int): Total number of points in fine grid.
        n_box (int): Number of points in the active box.
        
    Returns:
        tuple: (ix_start, ix_end, ix_relative_center)
    """
    # Find nearest global index for the center position
    idx_global = int(round(max(0.0, min(pos / dx, n_total_fine - 1))))
    
    target_center_offset = n_box // 2
    idx_start = idx_global - target_center_offset
    
    max_start = max(0, n_total_fine - n_box)
    idx_start = max(0, min(idx_start, max_start))
    idx_end = idx_start + n_box
    
    idx_relative = idx_global - idx_start
    idx_relative = max(0, min(idx_relative, n_box - 1))
    
    return idx_start, idx_end, idx_relative


# keep in helpers

def shift_flux(field: cp.ndarray, shift: tuple, geom) -> cp.ndarray:
    """Translate a surface flux field by ``shift=(dx, dy)`` meters on GPU."""
    dx, dy = shift
    shift_pixels = (dy / geom.dy, dx / geom.dx)
    out = cp.empty_like(field)
    cupy_ndimage.shift(field, shift_pixels, order=1, mode='constant', cval=0.0, output=out)
    return out