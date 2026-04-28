"""
CPU-based spectral method kernels for the heat equation.

Public functions in this module are called by SpectralSolverCPU:
- ``update_modes_etd1``: Time integration step
- ``compute_latent_heat_source``: Latent heat and evaporation effects
- ``reconstruct_surface_temperature``: Extract solution on top surface
"""

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
    
    # Derived 2D grids (X, Y are redundant but kept if heavily used, though we should prefer 1D)
    # Removing X, Y as per plan to reduce memory if they are just meshgrids of x, y

    # Reconstruction constants
    recon_scale: float = 0.0
    Cp32_broadcast: np.ndarray = None
    dct_scale: float = 0.0
    
    # Normalization coefficients
    Cm: np.ndarray = None
    Cn: np.ndarray = None
    Cp: np.ndarray = None

    # Precomputed cosine bases for reconstruction
    cos_mx: np.ndarray = None
    cos_ny: np.ndarray = None
    cos_pz: np.ndarray = None

    # Lazy-loaded full reconstruction bases
    Bx_recon: np.ndarray = None
    By_recon: np.ndarray = None
    Bz_recon: np.ndarray = None
    # Node-centered coords for reconstruction
    x_rec: np.ndarray = None
    y_rec: np.ndarray = None
    z_rec: np.ndarray = None
    
    def __init__(self, geom):
        # Global mesh coordinates (Cell-Centered)
        dx, dy, dz = geom.dx, geom.dy, geom.dz
        nx, ny, nz = geom.nx, geom.ny, geom.nz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
        
        self.x = ((np.arange(nx) + 0.5) * dx).astype(np.float32)
        self.y = ((np.arange(ny) + 0.5) * dy).astype(np.float32)
        self.z = ((np.arange(nz) + 0.5) * dz).astype(np.float32)
        
        # Normalization coefficients
        self.Cm = spec_hp._C_coef(nx, Lx)
        self.Cn = spec_hp._C_coef(ny, Ly)
        self.Cp = spec_hp._C_coef(nz, Lz)
        
        # Scaling factors
        self.dct_scale = np.float32((dx * dy) * np.sqrt((nx * ny) / (Lx * Ly)))
        self.recon_scale = np.float32(np.sqrt(nx * ny) / np.sqrt(Lx * Ly))
        
        # Precomputed cosine bases
        x_np, y_np, z_np = self.x, self.y, self.z
        self.cos_mx = np.cos(np.pi * np.arange(nx)[:, None] * x_np[None, :] / Lx).astype(np.float32)
        self.cos_ny = np.cos(np.pi * np.arange(ny)[:, None] * y_np[None, :] / Ly).astype(np.float32)
        self.cos_pz = np.cos(np.pi * np.arange(nz)[:, None] * z_np[None, :] / Lz).astype(np.float32)
        
        # Compute top-surface weighting for projection
        sign = np.power(-1.0, np.arange(nz, dtype=np.float32)).astype(np.float32)
        Cp_top = (self.Cp.astype(np.float32) * sign)
        self.Cp32_broadcast = Cp_top[:, None, None]
        
        # Bottom-surface weighting: cos(p*pi*0/Lz) = 1, so no sign alternation
        self.Cp32_broadcast_bottom = self.Cp.astype(np.float32)[:, None, None]

    def prepare_full_reconstruction(self, geom):
        """Compute node-centered grids and full-domain reconstruction bases on demand."""
        if self.Bx_recon is not None:
            return

        dx, dy, dz = geom.dx, geom.dy, geom.dz
        nx, ny, nz = geom.nx, geom.ny, geom.nz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz

        self.x_rec = ((np.arange(nx+1)) * dx).astype(np.float32)
        self.y_rec = ((np.arange(ny+1)) * dy).astype(np.float32)
        self.z_rec = ((np.arange(nz+1)) * dz).astype(np.float32)

        m, n, p = np.arange(nx), np.arange(ny), np.arange(nz)
        self.Bx_recon = (self.Cm[:, None] * np.cos(np.pi * m[:, None] * self.x_rec[None, :] / Lx)).astype(np.float32)
        self.By_recon = (self.Cn[:, None] * np.cos(np.pi * n[:, None] * self.y_rec[None, :] / Ly)).astype(np.float32)
        self.Bz_recon = (self.Cp[:, None] * np.cos(np.pi * p[:, None] * self.z_rec[None, :] / Lz)).astype(np.float32)

@dataclass
class FineMeshState:
    """Handles the moving fine mesh for latent heat/nonlinearities."""
    # Logic configuration
    refinement: int = 4
    nx_box: int = 0
    ny_box: int = 0
    nz_box: int = 0
    nx_fine_total: int = 0
    ny_fine_total: int = 0
    nz_fine_total: int = 0
    
    # Grid coordinates (Static full fine grid)
    x_fine: np.ndarray = None
    y_fine: np.ndarray = None
    z_fine: np.ndarray = None
    
    # Box Coordinate Arrays (Active Window)
    box_x: np.ndarray = None
    box_y: np.ndarray = None
    box_z: np.ndarray = None
    
    dx_fine: float = 0.0
    dy_fine: float = 0.0
    dz_fine: float = 0.0
    dV_fine: float = 0.0

    # Precomputed Full Fine Bases
    Bx_fine_full: np.ndarray = None
    By_fine_full: np.ndarray = None
    Bz_fine_full: np.ndarray = None
    
    # Active Box Basis Subsets (Changing every step)
    Bx_fine: np.ndarray = None 
    By_fine: np.ndarray = None
    Bz_fine: np.ndarray = None
    
    # History
    T_prev: np.ndarray = None 
    Q_prev: np.ndarray = None
    
    def __init__(self, geom, grid: SpectralGrid):
        import logging
        logger = logging.getLogger(__name__)
        
        dx, dy, dz = geom.dx, geom.dy, geom.dz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
        
        self.refinement = 4
        Lx_box, Ly_box, Lz_box = 0.9e-3, 0.9e-3, 0.04e-3
        
        self.dx_fine, self.dy_fine, self.dz_fine = dx/self.refinement, dy/self.refinement, dz/self.refinement
        
        self.nx_fine_total = int(np.ceil(Lx / self.dx_fine))
        self.ny_fine_total = int(np.ceil(Ly / self.dy_fine))
        self.nz_fine_total = int(np.ceil(Lz_box / self.dz_fine))
        
        self.x_fine = ((np.arange(self.nx_fine_total) + 0.5) * self.dx_fine).astype(np.float32)
        self.y_fine = ((np.arange(self.ny_fine_total) + 0.5) * self.dy_fine).astype(np.float32)
        self.z_fine = ((np.arange(self.nz_fine_total) + 0.5) * self.dz_fine).astype(np.float32)
        
        # Shift z_fine to top of the domain
        z_fine_global = (Lz - Lz_box) + self.z_fine

        logger.info("Precomputing fine cosine bases...")
        m, n, p = np.arange(geom.nx), np.arange(geom.ny), np.arange(geom.nz)
        self.Bx_fine_full = (grid.Cm[:, None] * np.cos(np.pi * m[:, None] * self.x_fine[None, :] / Lx)).astype(np.float32)
        self.By_fine_full = (grid.Cn[:, None] * np.cos(np.pi * n[:, None] * self.y_fine[None, :] / Ly)).astype(np.float32)
        self.Bz_fine_full = (grid.Cp[:, None] * np.cos(np.pi * p[:, None] * z_fine_global[None, :] / Lz)).astype(np.float32)
        
        # Box dimensions
        self.nx_box = int(np.ceil(Lx_box / self.dx_fine))
        self.ny_box = int(np.ceil(Ly_box / self.dy_fine))
        self.nz_box = self.nz_fine_total
        
        # Allocation for active window
        self.Bx_fine = np.zeros((geom.nx, self.nx_box), dtype=np.float32)
        self.By_fine = np.zeros((geom.ny, self.ny_box), dtype=np.float32)
        self.Bz_fine = self.Bz_fine_full[:, :self.nz_box]
        
        self.box_x = np.zeros(self.nx_box, dtype=np.float32)
        self.box_y = np.zeros(self.ny_box, dtype=np.float32)
        self.box_z = z_fine_global
        
        self.dV_fine = self.dx_fine * self.dy_fine * self.dz_fine

    def update(self, laser_state):
        """Update fine mesh box coordinates and basis subsets."""
        # Update X-Axis
        ix_start, ix_end, _ = _calculate_subgrid_indices(
            laser_state.x, self.dx_fine, self.nx_fine_total, self.nx_box
        )
        self.box_x[:] = self.x_fine[ix_start:ix_end]
        self.Bx_fine[:, :] = self.Bx_fine_full[:, ix_start:ix_end]
        
        # Update Y-Axis
        iy_start, iy_end, _ = _calculate_subgrid_indices(
            laser_state.y, self.dy_fine, self.ny_fine_total, self.ny_box
        )
        self.box_y[:] = self.y_fine[iy_start:iy_end]
        self.By_fine[:, :] = self.By_fine_full[:, iy_start:iy_end]


@dataclass
class SolverBuffers:
    """Reusable working arrays."""
    a_temp: np.ndarray = None      # (nz, ny, nx)
    q_evap_old: np.ndarray = None  # (ny, nx)
    q_evap_buffer: np.ndarray = None
    q_diff: np.ndarray = None
    B_buffer: np.ndarray = None
    Q_latent_buffer: np.ndarray = None

    def __init__(self, num, fine_mesh: FineMeshState):
        nx , ny, nz = num.nx, num.ny, num.nz
        self.q_diff = np.empty((ny, nx), dtype=np.float32)
        self.B_buffer = np.empty((ny, nx), dtype=np.float32)
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
    xp = np # Default to numpy for precalc

    kx = (np.pi * xp.arange(num.nx) / geom.Lx)
    ky = (np.pi * xp.arange(num.ny) / geom.Ly)
    kz = (np.pi * xp.arange(num.nz) / geom.Lz)

    denom = phys.k / (phys.rho * phys.Cp) *(kx[None, None, :]**2 + ky[None, :, None]**2 + kz[:, None, None]**2)
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


@njit(parallel=True, fastmath=True)
def update_modes_etd1(aK, KK, Cp_broadcast, B_scaled, a_temp_out):
    """
    Update spectral coefficients for ETD1 scheme.
    Calculates: a_out = aK + (KK * Cp) * B_scaled
    """
    nz = aK.shape[0]
    for p in prange(nz):
        a_temp_out[p, :, :] = aK[p, :, :] + KK[p, :, :] * Cp_broadcast[p, 0, 0] * B_scaled

@njit(parallel=True, fastmath=True)
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

@njit(parallel=True, fastmath=True)
def add_bottom_surface_source(a_temp, KK, Cp_broadcast_bottom, B_scaled):
    """
    Accumulate a surface source at z=0 into temperature modes.
    a_temp[p,:,:] += KK[p,:,:] * Cp_bottom[p] * B_scaled[:,:]
    """
    nz = a_temp.shape[0]
    for p in prange(nz):
        a_temp[p, :, :] += KK[p, :, :] * Cp_broadcast_bottom[p, 0, 0] * B_scaled[:, :]

@njit(parallel=True, fastmath=True)
def compute_source_term_from_temperature(T_curr, T_prev, T_S, T_L, rho, L, dt, out):
    """
    Compute Q = - rho * L * (1 / (TL - TS)) * (dT/dt) * Indicator(TS <= T <= TL)
    Used for latent heat calculation.
    """
    factor = - rho * L / ( (T_L - T_S) * dt )
    nz, ny, nx = T_curr.shape
    for k in prange(nz):
        for j in range(ny):
            for i in range(nx):
                T = T_curr[k, j, i]
                # Indicator function for mushy zone (inclusive)
                if T >= T_S and T <= T_L:
                    T_p = T_prev[k, j, i]
                    
                    # Fix T_prev to the boundaries [T_S, T_L] if it was outside.
                    # This ensures we calculate Delta(f_liquid) = (T - T_p_clamped)/(T_L - T_S),
                    # correctly separating latent heat from sensible heat.
                    if T_p < T_S-(T_L - T_S): 
                        T_p = T_S-(T_L - T_S)
                    elif T_p > T_L+(T_L - T_S):
                        T_p = T_L+(T_L - T_S)
                    
                    dT = T - T_p
                    # Note: Since both T and T_p are in [T_S, T_L], |dT| <= (T_L - T_S),
                    # so the energy bound is naturally satisfied.
                    
                    factor = -rho * L / ((T_L - T_S) * dt)
                    out[k, j, i] = factor * dT
                else:
                    out[k, j, i] = 0.0



@njit(parallel=True, fastmath=True)
def compute_evaporation_flux(T_surface, q_out, P0, R, T_boil, DeltaH_LV, R_v, T_liquidus):
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


def project_box_to_modes(field_box, SsState):
    """Project fine box field to global spectral modes."""
    if SsState.fine_mesh is None:
        raise RuntimeError("Fine mesh not initialized.")
    
    fm = SsState.fine_mesh
    # 1. Contract Z_box: (nz_box, ny_box, nx_box) . (nz, nz_box) -> (ny_box, nx_box, nz)
    temp1 = np.tensordot(field_box, fm.Bz_fine, axes=(0, 1))
    # 2. Contract Y_box: (ny_box, nx_box, nz) . (ny, ny_box) -> (nx_box, nz, ny)
    temp2 = np.tensordot(temp1, fm.By_fine, axes=(0, 1))
    # 3. Contract X_box: (nx_box, nz, ny) . (nx, nx_box) -> (nz, ny, nx)
    modes = np.tensordot(temp2, fm.Bx_fine, axes=(0, 1))
    
    modes *= fm.dV_fine
    return modes


def _reconstruct_temperature_box(a, SsState):
    """
    Reconstructs temperature in a small ROI around the laser.
    Pure Python function using tensordot (optimized in numpy).
    """
    if SsState.fine_mesh is None:
        raise RuntimeError("Fine mesh not initialized.")
    
    fm = SsState.fine_mesh
    # Tensor Contraction: Modes -> Physical Space
    # T(x,y,z) = sum_p sum_n sum_m  a[p,n,m] * Bz[p,z] * By[n,y] * Bx[m,x]
    
    T_step1 = np.tensordot(a, fm.Bz_fine, axes=(0, 0)) # Contraction over Z
    T_step2 = np.tensordot(T_step1, fm.By_fine, axes=(0, 0)) # Contraction over Y
    T_box = np.tensordot(T_step2, fm.Bx_fine, axes=(0, 0)) # Contraction over X
    
    return T_box, (fm.box_x, fm.box_y, fm.box_z)

def compute_latent_heat_source(Q_buffer, phys, laser_state, num, SsState, alpha=0.2):
    """
    Compute volumetric latent heat source Q (W/m^3).
    HIGH-LEVEL ORCHESTRATOR (Runs in Python, calls Kernels).
    """
    if SsState.fine_mesh is None:
        return

    fm = SsState.fine_mesh

    # 1. Reconstruct Temperature on Fine Mesh
    T_box, _ = _reconstruct_temperature_box(SsState.buffers.a_temp, SsState)

    # 2. Initialize/Retrieve State buffers
    if fm.T_prev is None:
        fm.T_prev = np.zeros_like(T_box)
        fm.T_prev[:] = T_box[:]
        fm.Q_prev = np.zeros_like(Q_buffer)
        Q_buffer.fill(0.0)
        return

    # 3. Shift Previous Fields to Current Frame
    shift_x = laser_state.v[0] * num.dt
    shift_y = laser_state.v[1] * num.dt

    shift_pixels = (0, -shift_y / fm.dy_fine, -shift_x / fm.dx_fine)
 
    # Order=1 (Linear) usually sufficient for smooth fields like T
    T_prev_aligned = scipy_shift(fm.T_prev, shift_pixels, order=1, mode='nearest')
    Q_prev_aligned = scipy_shift(fm.Q_prev, shift_pixels, order=1, mode='constant', cval=0.0)
    
    # 4. Compute Source Term (Calls Numba Kernel)
    compute_source_term_from_temperature(T_box, T_prev_aligned,
        phys.T_solidus, phys.T_liquidus,
        phys.rho, phys.L_f, num.dt,
        Q_buffer
    )

    # 5. Apply Relaxation
    if alpha < 1.0:
        Q_buffer[:] = alpha * Q_buffer + (1.0 - alpha) * Q_prev_aligned

    # 6. Update History
    fm.T_prev[:] = T_box[:]
    fm.Q_prev[:] = Q_buffer[:]

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
    idx_global = int(round(np.clip(pos / dx, 0.0, n_total_fine - 1)))
    
    # Calculate desired start index to center the box
    target_center_offset = n_box // 2
    idx_start = idx_global - target_center_offset
    
    # Clamp start index to valid range [0, max_start]
    max_start = max(0, n_total_fine - n_box)
    idx_start = max(0, min(idx_start, max_start))
    idx_end = idx_start + n_box
    
    # Calculate clamped relative center (index of laser within the box)
    idx_relative = idx_global - idx_start
    idx_relative = max(0, min(idx_relative, n_box - 1))
    
    return idx_start, idx_end, idx_relative


# keep in helpers
def shift_flux(field: np.ndarray, shift: tuple, geom) -> np.ndarray:
    """Translate a surface flux field by ``shift=(dx, dy)`` meters."""
    dx, dy = shift
    shift_pixels = (dy / geom.dy, dx / geom.dx)
    
    return scipy_shift(field, shift_pixels, order=1, mode='constant', cval=0.0)
