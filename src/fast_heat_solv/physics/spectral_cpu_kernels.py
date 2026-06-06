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
from fast_heat_solv.physics import spectral_helpers as spec_hp
import pyfftw
from dataclasses import dataclass

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

@dataclass
class SpectralGrid:
    """Immutable grid definitions and reconstruction bases."""
    # Global coordinates (Cell-Centered)
    x: np.ndarray = None
    y: np.ndarray = None
    z: np.ndarray = None

    # Reconstruction constants
    recon_scale: float = 0.0
    Cp32_broadcast: np.ndarray = None
    dct_scale: float = 0.0
    
    # Normalization coefficients (indexed by axis: 0=x, 1=y, 2=z)
    C: tuple = None

    # Lazy-loaded full reconstruction bases and node-centered coords (indexed by axis)
    B_recon: list = None
    coords_rec: list = None
    
    def __init__(self, geom):
        # Global mesh coordinates (Cell-Centered). Coordinate arrays stay
        # per-axis; Vec3 groups only the scalar triples (n, d, size) since a
        # dataclass cannot enter the numba kernels downstream.
        self.x, self.y, self.z = [((np.arange(n) + 0.5) * d).astype(np.float32)
                                   for n, d in zip(geom.n, geom.d)]

        # Normalization coefficients (C[0]=x, C[1]=y, C[2]=z)
        self.C = tuple(spec_hp._C_coef(n, L) for n, L in zip(geom.n, geom.size))

        # Scaling factors
        dx, dy = geom.dx, geom.dy
        nx, ny = geom.nx, geom.ny
        Lx, Ly = geom.Lx, geom.Ly
        self.dct_scale = np.float32((dx * dy) * np.sqrt((nx * ny) / (Lx * Ly)))
        self.recon_scale = np.float32(np.sqrt(nx * ny) / np.sqrt(Lx * Ly))

        # Compute top-surface weighting for projection
        sign = np.power(-1.0, np.arange(geom.nz, dtype=np.float32)).astype(np.float32)
        self.Cp32_broadcast = (self.C[2].astype(np.float32) * sign)[:, None, None]

        # Bottom-surface weighting: cos(p*pi*0/Lz) = 1, so no sign alternation
        self.Cp32_broadcast_bottom = self.C[2].astype(np.float32)[:, None, None]

    def prepare_full_reconstruction(self, geom):
        """Compute node-centered grids and full-domain reconstruction bases on demand."""
        if self.B_recon is not None:
            return

        nx, ny, nz = geom.nx, geom.ny, geom.nz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
        dx, dy, dz = geom.dx, geom.dy, geom.dz

        self.coords_rec = [((np.arange(n + 1)) * d).astype(np.float32)
                           for n, d in zip((nx, ny, nz), (dx, dy, dz))]
        self.B_recon = [(self.C[a][:, None] * np.cos(np.pi * np.arange(n)[:, None] * self.coords_rec[a][None, :] / L)).astype(np.float32)
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

    T_prev: np.ndarray = None
    Q_prev: np.ndarray = None

    def __init__(self, geom, grid: SpectralGrid):
        import logging
        logger = logging.getLogger(__name__)

        dx, dy, dz = geom.dx, geom.dy, geom.dz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz

        self.refinement = 4
        # TODO: these fine-mesh box extents are hard-coded; they should become
        # configurable parameters (or be sized dynamically from the domain /
        # laser footprint) rather than baked-in constants.
        Lx_box, Ly_box, Lz_box = 0.9e-3, 0.9e-3, 0.04e-3

        self.dx_fine, self.dy_fine, self.dz_fine = dx/self.refinement, dy/self.refinement, dz/self.refinement
        d_fine = [self.dx_fine, self.dy_fine, self.dz_fine]

        self.n_fine_totals = [int(np.ceil(L / d)) for L, d in zip([Lx, Ly, Lz_box], d_fine)]
        self.coords_fine = [((np.arange(n) + 0.5) * d).astype(np.float32)
                            for n, d in zip(self.n_fine_totals, d_fine)]

        z_fine_global = (Lz - Lz_box) + self.coords_fine[2]

        logger.info("Precomputing fine cosine bases...")
        mode_counts = [geom.nx, geom.ny, geom.nz]
        L_domain = [Lx, Ly, Lz]
        fine_coords = [self.coords_fine[0], self.coords_fine[1], z_fine_global]
        self.B_fine_full = [
            (grid.C[a][:, None] * np.cos(np.pi * np.arange(mode_counts[a])[:, None] * fine_coords[a][None, :] / L_domain[a])).astype(np.float32)
            for a in range(3)
        ]

        self.nx_box = min(int(np.ceil(Lx_box / self.dx_fine)), self.n_fine_totals[0])
        self.ny_box = min(int(np.ceil(Ly_box / self.dy_fine)), self.n_fine_totals[1])
        self.nz_box = min(int(np.ceil(Lz_box / self.dz_fine)), self.n_fine_totals[2])

        self.B_fine = [
            np.zeros((geom.nx, self.nx_box), dtype=np.float32),
            np.zeros((geom.ny, self.ny_box), dtype=np.float32),
            self.B_fine_full[2][:, :self.nz_box],  # z: static full-depth slice
        ]
        self.dV_fine = self.dx_fine * self.dy_fine * self.dz_fine

    def update(self, laser_state):
        """Update fine mesh basis subsets for x and y axes."""
        for a, (pos, d, n_box) in enumerate([
            (laser_state.x, self.dx_fine, self.nx_box),
            (laser_state.y, self.dy_fine, self.ny_box),
        ]):
            i_start, i_end, _ = spec_hp._calculate_subgrid_indices(pos, d, self.n_fine_totals[a], n_box)
            self.B_fine[a][:, :] = self.B_fine_full[a][:, i_start:i_end]
        # z does not slide


@dataclass
class SolverBuffers:
    """Reusable working arrays."""
    a_temp: np.ndarray = None      # (nz, ny, nx)
    q_evap_old: np.ndarray = None  # (ny, nx)
    q_evap_buffer: np.ndarray = None
    Q_latent_buffer: np.ndarray = None

    def __init__(self, num, fine_mesh: FineMeshState):
        nx , ny, nz = num.nx, num.ny, num.nz
        self.a_temp = np.empty((nz, ny, nx), dtype=np.float32)
        self.q_evap_old = np.zeros((ny, nx), dtype=np.float32)
        self.q_evap_buffer = np.zeros((ny, nx), dtype=np.float32)
        if fine_mesh:
             self.Q_latent_buffer = np.zeros((fine_mesh.nz_box, fine_mesh.ny_box, fine_mesh.nx_box), dtype=np.float32)


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
    a: np.ndarray = None       # Current temperature modes (nz, ny, nx)
    
    # 3. Spectral Propagators
    K: np.ndarray = None
    KK: np.ndarray = None

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
    # Per-axis wavenumbers in array-index order [z, y, x]; .zyx() makes the
    # reversal from (x, y, z) explicit instead of an implicit reversed literal.
    k = [np.pi * np.arange(n) / L for n, L in zip(geom.n.zyx(), geom.size.zyx())]
    k_grids = np.meshgrid(*k, indexing='ij')  # shape (nz, ny, nx) each
    denom = phys.k / (phys.rho * phys.Cp) * sum(kg**2 for kg in k_grids)
    K = np.exp(-denom * num.dt).astype(np.float32)
    mask_zero = (denom == 0)

    denom[mask_zero] = 1.0

    phi_1 = (K - 1.0) / (-denom)
    phi_1[mask_zero] = num.dt

    KK = (phi_1 / (phys.rho * phys.Cp)).astype(np.float32)
    return K, KK



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
