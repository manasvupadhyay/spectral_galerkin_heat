import numpy as np
from numba import njit, prange
from scipy.ndimage import shift as scipy_shift
import utils.spectral_helpers as spec_hp
import pyfftw
from dataclasses import dataclass

# ======================================
# Spectral Method CPU State Definition
# ======================================


@dataclass
class SpectralSolverState:
    """
    Encapsulates solver-specific buffers and precomputed grids needed for the spectral method.
    Keeps global GeomParams/NumParams clean from implementation details.
    """
    # Spectral Propagators
    K: np.ndarray = None
    KK: np.ndarray = None
    KK_by_Cp: np.ndarray = None
    
    # State Arrays
    a: np.ndarray = None       # Current temperature modes (nz, ny, nx)
    aK: np.ndarray = None      # Decayed temperature modes (nz, ny, nx)
    a_temp: np.ndarray = None  # Temporary working array (nz, ny, nx)
    
    # Buffers
    q_evap_old: np.ndarray = None
    q_evap_buffer: np.ndarray = None
    q_diff: np.ndarray = None
    B_buffer: np.ndarray = None
    
    # Latent Heat Specifics
    Q_latent_buffer: np.ndarray = None
    isotherm_cache: list = None
    
        
    # Fine Grid / Aliasing structures
    # Box Coordinate Arrays (Changing every step)
    box_x: np.ndarray = None
    box_y: np.ndarray = None
    box_z: np.ndarray = None
    
    # Active Box Basis Subsets (Changing every step)
    Bx_fine: np.ndarray = None 
    By_fine: np.ndarray = None
    Bz_fine: np.ndarray = None
    
    # Precomputed Full Fine Bases (Static, but needed for slicing)
    Bx_fine_full: np.ndarray = None
    By_fine_full: np.ndarray = None
    Bz_fine_full: np.ndarray = None
    
    # Box State Descriptors
    ix_laser_box: int = 0
    nx_box: int = 0
    ny_box: int = 0
    nz_box: int = 0
    
    fine_mesh_initialized: bool = False

    # Grid coordinates
    x: np.ndarray = None
    y: np.ndarray = None
    z: np.ndarray = None
    X: np.ndarray = None
    Y: np.ndarray = None
    
    # Helper coefficients
    Cp32_broadcast: np.ndarray = None
    recon_scale: float = 0.0

    # Fine grid coords
    x_fine: np.ndarray = None
    y_fine: np.ndarray = None
    z_fine: np.ndarray = None
    dx_fine: float = 0.0
    dy_fine: float = 0.0
    dz_fine: float = 0.0
    nx_fine_total: int = 0
    ny_fine_total: int = 0
    nz_fine_total: int = 0

    # Additional history for Latent Heat
    T_prev: np.ndarray = None
    Q_prev: np.ndarray = None
    laser_x_prev: float = None
    laser_y_prev: float = None
    dV_fine: float = 0.0

    def prepare_reconstruction_basis(self, geom):
        # Global mesh coordinates (Cell-Centered to match the definition of DCT-II)
        dx, dy, dz = geom.dx, geom.dy, geom.dz
        nx, ny, nz = geom.nx, geom.ny, geom.nz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
        self.x = ((np.arange(nx) + 0.5) * dx).astype(np.float32)
        self.y = ((np.arange(ny) + 0.5) * dy).astype(np.float32)
        self.z = ((np.arange(nz) + 0.5) * dz).astype(np.float32)
        x_np, y_np, z_np = self.x, self.y, self.z
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        self.X, self.Y = self.X.astype(np.float32), self.Y.astype(np.float32)
        """Precompute reconstruction bases and normalization coefficients."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Precomputing reconstruction bases...")
        # Normalization coefficients ([FIX] Use spec_hp.C_coef)
        self.Cm = spec_hp.C_coef(nx, Lx)
        self.Cn = spec_hp.C_coef(ny, Ly)
        self.Cp = spec_hp.C_coef(nz, Lz)
        self.Cp32_broadcast = self.Cp.astype(np.float32)[:, None, None]
        
        # Scaling factors for DCT/IDCT
        self.dct_scale = np.float32((dx * dy) * np.sqrt((nx * ny) / (Lx * Ly)))
        self.recon_scale = np.float32(np.sqrt(nx * ny) / np.sqrt(Lx * Ly))

        # Precomputed cosine bases for reconstruction
        self.cos_mx = np.cos(np.pi * np.arange(nx)[:, None] * x_np[None, :] / Lx).astype(np.float32)
        self.cos_ny = np.cos(np.pi * np.arange(ny)[:, None] * y_np[None, :] / Ly).astype(np.float32)
        self.cos_pz = np.cos(np.pi * np.arange(nz)[:, None] * z_np[None, :] / Lz).astype(np.float32)

        # Fine mesh setup for latent heat correction
        self.refinement = 4
        self.Lx_box, self.Ly_box, self.Lz_box = 0.7e-3, 0.2e-3, 0.04e-3
        self.dx_fine, self.dy_fine, self.dz_fine = dx/self.refinement, dy/self.refinement, dz/self.refinement
        
        self.nx_fine_total = int(np.ceil(Lx / self.dx_fine))
        self.ny_fine_total = int(np.ceil(Ly / self.dy_fine))
        self.nz_fine_total = int(np.ceil(self.Lz_box / self.dz_fine))
        
        x_fine = ((np.arange(self.nx_fine_total) + 0.5) * self.dx_fine).astype(np.float32)
        y_fine = ((np.arange(self.ny_fine_total) + 0.5) * self.dy_fine).astype(np.float32)
        z_fine = ((np.arange(self.nz_fine_total) + 0.5) * self.dz_fine).astype(np.float32)
        self.x_fine = x_fine
        self.y_fine = y_fine
        self.z_fine = z_fine
        
        logger.info("Precomputing fine cosine bases...")
        m, n, p = np.arange(nx), np.arange(ny), np.arange(nz)
        self.Bx_fine_full = (self.Cm[:, None] * np.cos(np.pi * m[:, None] * x_fine[None, :] / Lx)).astype(np.float32)
        self.By_fine_full = (self.Cn[:, None] * np.cos(np.pi * n[:, None] * y_fine[None, :] / Ly)).astype(np.float32)
        self.Bz_fine_full = (self.Cp[:, None] * np.cos(np.pi * p[:, None] * z_fine[None, :] / Lz)).astype(np.float32)
        
        # Box dimensions in fine grid points
        self.nx_box = int(np.ceil(self.Lx_box / self.dx_fine))
        self.ny_box = int(np.ceil(self.Ly_box / self.dy_fine))
        self.nz_box = self.nz_fine_total
        
        # Preallocated arrays for fine mesh box
        self.Bx_fine = np.zeros((nx, self.nx_box), dtype=np.float32)
        self.By_fine = np.zeros((ny, self.ny_box), dtype=np.float32)
        self.Bz_fine = self.Bz_fine_full[:, :self.nz_box]
        self.box_x = np.zeros(self.nx_box, dtype=np.float32)
        self.box_y = np.zeros(self.ny_box, dtype=np.float32)
        self.box_z = np.linspace(0.0, self.Lz_box, self.nz_box, dtype=np.float32)
        self.fine_mesh_initialized = False
        self.ix_laser_box = max(0, min(self.nx_box - 1, int(round(0.5 * (self.nx_box - 1)))))
        
        self.dV_fine = self.dx_fine * self.dy_fine * self.dz_fine

    def prepare_K_buffers(self, phys, geom, num):
        """Precompute spectral propagators and allocate buffers.
        """

        import logging
        logger = logging.getLogger(__name__)
        logger.info("Precomputing K, KK... ")
        # [FIX] Use spec_hp.precompute_K_KK
        self.K, self.KK = precompute_K_KK(phys, num, geom)
        self.KK_by_Cp = (self.KK * self.Cp[:, None, None]).astype(np.float32) # Projected on x,y plane
        nx, ny, nz = num.nx, num.ny, num.nz
        # Allocate working arrays
        self.q_diff = np.empty((ny, nx), dtype=np.float32)
        self.B_buffer = np.empty((ny, nx), dtype=np.float32)
        self.a_temp = np.empty((nz, ny, nx), dtype=np.float32)
        self.aK = np.empty((nz, ny, nx), dtype=np.float32)
        self.q_evap_old = np.zeros((ny, nx), dtype=np.float32)
        self.q_evap_buffer = np.zeros((ny, nx), dtype=np.float32)
        # ZYX layout for contiguous X-scanning
        self.Q_latent_buffer = np.zeros((self.nz_box, self.ny_box, self.nx_box), dtype=np.float32)


def precompute_K_KK(phys, num, geom):
    """
    Compute spectral Propagators (K, KK) based on grid and time step.
    K = exp(-alpha * k^2 * dt) for ETD1 (Exact integration of linear part)
    KK = phi_1 / (rho * Cp), where phi_1(z) = (exp(z) - 1) / z, z = -alpha * k^2 * dt
    """
    xp = np # Default to numpy for precalc

    kx = (np.pi * xp.arange(num.nx) / geom.Lx)
    ky = (np.pi * xp.arange(num.ny) / geom.Ly)
    kz = (np.pi * xp.arange(num.nz) / geom.Lz)

    KX, KY, KZ = xp.meshgrid(kx, ky, kz, indexing='ij')
    k2 = (kx[None, None, :]**2 + ky[None, :, None]**2 + kz[:, None, None]**2)

    alpha = phys.k / (phys.rho * phys.Cp)
    K = xp.exp(-alpha * k2 * num.dt).astype(np.float32)

    denom = alpha * k2
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
def update_modes_etd1(aK, KK_by_Cp, B_scaled, a_temp_out):
    """
    Update spectral coefficients for ETD1 scheme.
    Calculates: a_out = aK + (KK/Cp) * B_scaled
    """
    nz = aK.shape[0]
    for p in prange(nz):
        a_temp_out[p, :, :] = aK[p, :, :] + KK_by_Cp[p, :, :] * B_scaled

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
                    dT = T - T_prev[k, j, i]
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


def compute_gaussian_laser_flux(X, Y, laser_x, laser_y, laser_r, laser_coef):
    """Compute Gaussian flux on grid X,Y."""
    r_sq = (X - laser_x) ** 2 + (Y - laser_y) ** 2
    return (laser_coef * np.exp(-2.0 * r_sq / laser_r ** 2)).astype(np.float32)


def project_box_to_modes(field_box, SsState):
    """Project fine box field to global spectral modes."""
    if not SsState.fine_mesh_initialized:
        raise RuntimeError("Fine mesh not initialized.")
    
    # 1. Contract Z_box: (nz_box, ny_box, nx_box) . (nz, nz_box) -> (ny_box, nx_box, nz)
    temp1 = np.tensordot(field_box, SsState.Bz_fine, axes=(0, 1))
    # 2. Contract Y_box: (ny_box, nx_box, nz) . (ny, ny_box) -> (nx_box, nz, ny)
    temp2 = np.tensordot(temp1, SsState.By_fine, axes=(0, 1))
    # 3. Contract X_box: (nx_box, nz, ny) . (nx, nx_box) -> (nz, ny, nx)
    modes = np.tensordot(temp2, SsState.Bx_fine, axes=(0, 1))
    
    modes *= SsState.dV_fine
    return modes.astype(np.float32)


def reconstruct_temperature_box(a, SsState):
    """
    Reconstructs temperature in a small ROI around the laser.
    Pure Python function using tensordot (optimized in numpy).
    """
    if not SsState.fine_mesh_initialized:
        raise RuntimeError("Fine mesh not initialized.")
    
    # Tensor Contraction: Modes -> Physical Space
    # T(x,y,z) = sum_p sum_n sum_m  a[p,n,m] * Bz[p,z] * By[n,y] * Bx[m,x]
    
    T_step1 = np.tensordot(a, SsState.Bz_fine, axes=(0, 0)) # Contraction over Z
    T_step2 = np.tensordot(T_step1, SsState.By_fine, axes=(0, 0)) # Contraction over Y
    T_box = np.tensordot(T_step2, SsState.Bx_fine, axes=(0, 0)) # Contraction over X
    
    return T_box.astype(np.float32), (SsState.box_x, SsState.box_y, SsState.box_z)

def compute_latent_heat_source(Q_buffer, phys, x_laser, y_laser, num, SsState, alpha=0.4):
    """
    Compute volumetric latent heat source Q (W/m^3).
    HIGH-LEVEL ORCHESTRATOR (Runs in Python, calls Kernels).
    """
    # 1. Reconstruct Temperature on Fine Mesh
    T_box, _ = reconstruct_temperature_box(SsState.a_temp, SsState)
    
    # 2. Initialize/Retrieve State buffers
    if not hasattr(SsState, 'T_prev') or SsState.T_prev is None:
        SsState.T_prev = np.zeros_like(T_box)
        SsState.T_prev[:] = T_box[:] 
        SsState.Q_prev = np.zeros_like(Q_buffer)
        SsState.laser_x_prev = x_laser
        SsState.laser_y_prev = y_laser
        Q_buffer.fill(0.0)
        return

    # 3. Shift Previous Fields to Current Frame
    shift_x = x_laser - SsState.laser_x_prev
    shift_y = y_laser - SsState.laser_y_prev
    
    shift_pixels = (0, -shift_y / SsState.dy_fine, -shift_x / SsState.dx_fine)
    
    # Order=1 (Linear) usually sufficient for smooth fields like T
    T_prev_aligned = scipy_shift(SsState.T_prev, shift_pixels, order=1, mode='nearest')
    Q_prev_aligned = scipy_shift(SsState.Q_prev, shift_pixels, order=1, mode='constant', cval=0.0)
    
    # 4. Compute Source Term (Calls Numba Kernel)
    compute_source_term_from_temperature(T_box, T_prev_aligned, 
                                          phys.T_solidus, phys.T_liquidus, 
                                          phys.rho, phys.L_f, num.dt, 
                                          Q_buffer)
    
    # 5. Apply Relaxation
    if alpha < 1.0:
        Q_buffer[:] = alpha * Q_buffer + (1.0 - alpha) * Q_prev_aligned
    
    # 6. Update History
    SsState.T_prev[:] = T_box[:]
    SsState.Q_prev[:] = Q_buffer[:] 
    SsState.laser_x_prev = x_laser
    SsState.laser_y_prev = y_laser


def DCT_II(q):
    """Apply Discrete Cosine Transform Type II (Ortho)."""
    arr = np.ascontiguousarray(q, dtype=np.float32)
    return pyfftw.interfaces.scipy_fft.dctn(arr, type=2, norm='ortho', axes=tuple(range(arr.ndim)), workers=-1).astype(np.float32, copy=False)

def IDCT_II(a):
    """Apply Discrete Cosine Transform Type II (Ortho)."""
    arr = np.ascontiguousarray(a, dtype=np.float32)
    return pyfftw.interfaces.scipy_fft.dctn(arr, type=3, norm='ortho', axes=tuple(range(arr.ndim)), workers=-1).astype(np.float32, copy=False)

# Gain of few percent compared to scipy.fft.dctn(...) directly

def reconstruct_surface_temperature(a, SsState):
    """Reconstruct 2D temperature field at z=0."""
    # Sum over Z modes (weighted by Cp coefficients at z=0, which is just Cp/sqrt(1/L)?? No)
    # In helpers.py: A = (SsState.Cp32[:, None, None] * a).sum(axis=0)
    # This assumes cos(p*pi*z/Lz) at z=0 is 1.0. 
    # The reconstruction formula is T = sum(a * Bx * By * Bz).
    # Bz[p] at z=0 is Cp[p] * cos(0) = Cp[p].
    A = (SsState.Cp32_broadcast * a).sum(axis=0)
    # Use DCT-II for surface temperature (mathematical definition)
    dct_result = IDCT_II(A)
    return (SsState.recon_scale * dct_result).astype(np.float32, copy=False)

def reconstruct_temperature_xz(a, num, geom, SsState, laser, y0=None):
    """
    Optimized reconstruction of X-Z temperature slice using FFTW/DCT.
    Returns (x_vals, z_vals, T_xz).
    """
    y0 = laser.y0
    nx, ny, nz = num.nx, num.ny, num.nz
    # Evaluate cosine basis at specific y0 (ny,)
    cos_y = SsState.Cn * np.cos(np.pi * np.arange(ny) * y0 / geom.Ly)
    # Contract Y axis: (nz, ny, nx) dot (ny,)
    A_xz = np.tensordot(a, cos_y, axes=(1, 0)) 
    # Reconstruct X-Z field using 2D IDCT (Type 3)
    scale_xz = np.sqrt(nx * nz / (geom.Lx * geom.Lz))
    T_xz = scale_xz * dctn(A_xz, type=3, norm='ortho', axes=(0, 1))
    
    return geom.x, geom.z, T_xz.astype(np.float32)


def calculate_subgrid_indices(pos, dx, n_total_fine, n_box):
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

# goes to spectral_helpers.py
def update_fine_mesh(SsState, x_laser, y_laser):
    """
    Update fine mesh box coordinates and basis subsets so the laser remains centered.
    """
    # 1. Update X-Axis (Scanning Direction)
    ix_start, ix_end, ix_laser_rel = calculate_subgrid_indices(
        x_laser, SsState.dx_fine, SsState.nx_fine_total, SsState.nx_box
    )
    
    SsState.box_x[:] = SsState.x_fine[ix_start:ix_end]
    SsState.ix_laser_box = ix_laser_rel

    # Helper to slice and copy bases safely (handles non-contiguous memory)
    # Bx_fine shape: (nx_global, nx_box)
    SsState.Bx_fine[:, :] = SsState.Bx_fine_full[:, ix_start:ix_end]
    
    # 2. Update Y-Axis (Transverse Direction)
    iy_start, iy_end, _ = calculate_subgrid_indices(
        y_laser, SsState.dy_fine, SsState.ny_fine_total, SsState.ny_box
    )
    
    SsState.box_y[:] = SsState.y_fine[iy_start:iy_end]
    # By_fine shape: (ny_global, ny_box)
    SsState.By_fine[:, :] = SsState.By_fine_full[:, iy_start:iy_end]

    # 3. Update Volume Metric & Status
    SsState.dV_fine = SsState.dx_fine * SsState.dy_fine * SsState.dz_fine
    SsState.fine_mesh_initialized = True


# keep in helpers
def shift_flux(field: np.ndarray, shift: tuple, geom) -> np.ndarray:
    """Translate a surface flux field by ``shift=(dx, dy)`` meters."""
    if field is None or field.size == 0:
        return np.zeros((geom.ny, geom.nx), dtype=np.float32)
    dx, dy = shift
    shift_pixels = (dy / geom.dy, dx / geom.dx)
    
    return scipy_shift(field, shift_pixels, order=1, mode='constant', cval=0.0).astype(np.float32) 
