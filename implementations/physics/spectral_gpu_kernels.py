import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cupy_ndimage
import cupyx.scipy.fft as cupy_fft
from numba import cuda
import math
from dataclasses import dataclass
import utils.spectral_helpers as spec_hp  # Assuming this contains only scalar logic or is ported elsewhere

# ======================================
# CUDA Kernels (Device Functions)
# ======================================

@cuda.jit
def update_modes_etd1_kernel(aK, KK, Cp_broadcast, B_scaled, a_temp_out):
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
def add_source_term_modes_kernel(a_temp, KK, Q_modes):
    """
    Accumulate volumetric source term into temperature modes.
    Grid: 3D (nz, ny, nx)
    """
    z, y, x = cuda.grid(3)
    nz, ny, nx = a_temp.shape
    
    if z < nz and y < ny and x < nx:
        a_temp[z, y, x] += KK[z, y, x] * Q_modes[z, y, x]

@cuda.jit
def compute_source_term_kernel(T_curr, T_prev, T_S, T_L, rho, L, dt, out):
    """
    Compute latent heat source term Q.
    Grid: 3D (nz, ny, nx)
    """
    z, y, x = cuda.grid(3)
    nz, ny, nx = T_curr.shape
    
    if z < nz and y < ny and x < nx:
        T = T_curr[z, y, x]
        # Indicator function for mushy zone (inclusive)
        if T >= T_S and T <= T_L:
            dT = T - T_prev[z, y, x]
            factor = -rho * L / ((T_L - T_S) * dt)
            out[z, y, x] = factor * dT
        else:
            out[z, y, x] = 0.0

@cuda.jit
def compute_evaporation_flux_kernel(T_surface, q_out, P0, R_gas, T_boil, DeltaH_LV, R_v, T_liquidus):
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
class SpectralSolverState:
    """
    Encapsulates solver-specific buffers and precomputed grids needed for the spectral method.
    ALL ARRAYS ARE CUPY ARRAYS.
    """
    # Spectral Propagators
    K: cp.ndarray = None
    KK: cp.ndarray = None
    
    # State Arrays
    # aK: cp.ndarray = None       # REMOVED (aliased to 'a' in solver)
    a: cp.ndarray = None       # Current temperature modes (nz, ny, nx)
    a_temp: cp.ndarray = None  # Temporary working array (nz, ny, nx)
    
    # Buffers
    q_evap_old: cp.ndarray = None
    q_evap_buffer: cp.ndarray = None
    q_diff: cp.ndarray = None
    B_buffer: cp.ndarray = None
    
    # Latent Heat Specifics
    Q_latent_buffer: cp.ndarray = None
    isotherm_cache: list = None
    
    # Fine Grid / Aliasing structures
    # Box Coordinate Arrays (Changing every step)
    box_x: cp.ndarray = None
    box_y: cp.ndarray = None
    box_z: cp.ndarray = None
    
    # Active Box Basis Subsets (Changing every step)
    Bx_fine: cp.ndarray = None 
    By_fine: cp.ndarray = None
    Bz_fine: cp.ndarray = None
    
    # Precomputed Full Fine Bases (Static, but needed for slicing)
    Bx_fine_full: cp.ndarray = None
    By_fine_full: cp.ndarray = None
    Bz_fine_full: cp.ndarray = None
    
    # Box State Descriptors
    ix_laser_box: int = 0
    nx_box: int = 0
    ny_box: int = 0
    nz_box: int = 0
    
    fine_mesh_initialized: bool = False
    
    # Grid coordinates
    x: cp.ndarray = None
    y: cp.ndarray = None
    z: cp.ndarray = None
    X: cp.ndarray = None
    Y: cp.ndarray = None
    
    # Helper coefficients
    Cp32_broadcast: cp.ndarray = None
    recon_scale: float = 0.0

    # Fine grid coords
    x_fine: cp.ndarray = None
    y_fine: cp.ndarray = None
    z_fine: cp.ndarray = None
    
    # Lazy full domain
    x_rec: cp.ndarray = None
    y_rec: cp.ndarray = None
    z_rec: cp.ndarray = None
    Bx_recon: cp.ndarray = None
    By_recon: cp.ndarray = None
    Bz_recon: cp.ndarray = None
    full_recon_initialized: bool = False
    dx_fine: float = 0.0
    dy_fine: float = 0.0
    dz_fine: float = 0.0
    nx_fine_total: int = 0
    ny_fine_total: int = 0
    nz_fine_total: int = 0

    # Additional history for Latent Heat
    T_prev: cp.ndarray = None
    Q_prev: cp.ndarray = None
    laser_x_prev: float = None
    laser_y_prev: float = None
    dV_fine: float = 0.0

    def prepare_reconstruction_basis(self, geom):
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Precomputing reconstruction bases on GPU...")

        dx, dy, dz = geom.dx, geom.dy, geom.dz
        nx, ny, nz = geom.nx, geom.ny, geom.nz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz

        # Create grids on GPU
        self.x = ((cp.arange(nx) + 0.5) * dx).astype(cp.float32)
        self.y = ((cp.arange(ny) + 0.5) * dy).astype(cp.float32)
        self.z = ((cp.arange(nz) + 0.5) * dz).astype(cp.float32)
        
        # Meshgrid on GPU
        self.X, self.Y = cp.meshgrid(self.x, self.y, indexing='xy')
        self.X, self.Y = self.X.astype(cp.float32), self.Y.astype(cp.float32)
        """Precompute reconstruction bases and normalization coefficients."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Precomputing reconstruction bases...")
        # Normalization coefficients ([FIX] Use spec_hp.C_coef)
        self.Cm = cp.asarray(spec_hp.C_coef(nx, Lx), dtype=cp.float32)
        self.Cn = cp.asarray(spec_hp.C_coef(ny, Ly), dtype=cp.float32)
        self.Cp = cp.asarray(spec_hp.C_coef(nz, Lz), dtype=cp.float32)
       
        # Scaling factors for DCT/IDCT
        self.dct_scale = cp.asarray((dx * dy) * cp.sqrt((nx * ny) / (Lx * Ly)), dtype=cp.float32)
        self.recon_scale = cp.asarray(cp.sqrt(nx * ny) / cp.sqrt(Lx * Ly), dtype=cp.float32)
        
        # Precomputed cosine bases for reconstruction (cell-centered points)
        x_np, y_np, z_np = self.x, self.y, self.z
        self.cos_mx = cp.cos(cp.pi * cp.arange(nx)[:, None] * x_np[None, :] / Lx).astype(cp.float32)
        self.cos_ny = cp.cos(cp.pi * cp.arange(ny)[:, None] * y_np[None, :] / Ly).astype(cp.float32)
        self.cos_pz = cp.cos(cp.pi * cp.arange(nz)[:, None] * z_np[None, :] / Lz).astype(cp.float32)

        # Full-domain bases are allocated lazily via prepare_full_reconstruction()
        self.full_recon_initialized = False
        
        # Fine mesh setup for latent heat correction
        self.refinement = 3
        self.Lx_box, self.Ly_box, self.Lz_box = 0.9e-3, 0.2e-3, 0.04e-3
        self.dx_fine, self.dy_fine, self.dz_fine = dx/self.refinement, dy/self.refinement, dz/self.refinement
        
        self.nx_fine_total = int(cp.ceil(Lx / self.dx_fine))
        self.ny_fine_total = int(cp.ceil(Ly / self.dy_fine))
        self.nz_fine_total = int(cp.ceil(self.Lz_box / self.dz_fine))
        
        x_fine = ((cp.arange(self.nx_fine_total) + 0.5) * self.dx_fine).astype(cp.float32)
        y_fine = ((cp.arange(self.ny_fine_total) + 0.5) * self.dy_fine).astype(cp.float32)
        z_fine = ((cp.arange(self.nz_fine_total) + 0.5) * self.dz_fine).astype(cp.float32)
        self.x_fine = x_fine
        self.y_fine = y_fine
        self.z_fine = z_fine

        # Shift z_fine to top of the domain for basis evaluation (fine mesh is at the top)
        z_fine_global = (Lz - self.Lz_box) + z_fine

        
        logger.info("Precomputing fine cosine bases on GPU...")
        m, n, p = cp.arange(nx), cp.arange(ny), cp.arange(nz)
        
        # Broadcasting for basis computation
        # (nx, 1) * (1, nx_fine) -> (nx, nx_fine)
        self.Bx_fine_full = (self.Cm[:, None] * cp.cos(cp.pi * m[:, None] * x_fine[None, :] / Lx)).astype(cp.float32)
        self.By_fine_full = (self.Cn[:, None] * cp.cos(cp.pi * n[:, None] * y_fine[None, :] / Ly)).astype(cp.float32)
        self.Bz_fine_full = (self.Cp[:, None] * cp.cos(cp.pi * p[:, None] * z_fine_global[None, :] / Lz)).astype(cp.float32)
        
        # Box dimensions in fine grid points
        self.nx_box = int(cp.ceil(self.Lx_box / self.dx_fine))
        self.ny_box = int(cp.ceil(self.Ly_box / self.dy_fine))
        self.nz_box = self.nz_fine_total
        
        # Preallocated arrays for fine mesh box (on GPU)
        self.Bx_fine = cp.zeros((nx, self.nx_box), dtype=cp.float32)
        self.By_fine = cp.zeros((ny, self.ny_box), dtype=cp.float32)
        self.Bz_fine = self.Bz_fine_full[:, :self.nz_box] # Slicing works on GPU
        self.box_x = cp.zeros(self.nx_box, dtype=cp.float32)
        self.box_y = cp.zeros(self.ny_box, dtype=cp.float32)
        self.box_z = z_fine_global # Use actual cell centers
        
        self.fine_mesh_initialized = False
        self.ix_laser_box = max(0, min(self.nx_box - 1, int(round(0.5 * (self.nx_box - 1)))))
        
        self.dV_fine = self.dx_fine * self.dy_fine * self.dz_fine

    def prepare_K_buffers(self, phys, geom, num):
        """Precompute spectral propagators and allocate buffers on GPU."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Precomputing K, KK on GPU... ")
        
        self.K, self.KK = precompute_K_KK(phys, num, geom)
        # Project KK onto top-surface weighted Cp for source projection
        nx, ny, nz = num.nx, num.ny, num.nz
        sign = cp.power(-1.0, cp.arange(nz, dtype=cp.float32)).astype(cp.float32)
        Cp_top = (self.Cp.astype(cp.float32) * sign)
        # Broadcasted coefficient array for fast contraction with modal arrays (shape: (nz,1,1))
        self.Cp32_broadcast = Cp_top[:, None, None]
        # Allocate working arrays on GPU
        self.q_diff = cp.empty((ny, nx), dtype=cp.float32)
        self.B_buffer = cp.empty((ny, nx), dtype=cp.float32)
        self.a_temp = cp.empty((nz, ny, nx), dtype=cp.float32)
        self.q_evap_old = cp.zeros((ny, nx), dtype=cp.float32)
        self.q_evap_buffer = cp.zeros((ny, nx), dtype=cp.float32)
        # ZYX layout for latent heat source
        self.Q_latent_buffer = cp.zeros((self.nz_box, self.ny_box, self.nx_box), dtype=cp.float32)
        
    def full_reconstruction(self, geom):
        """Prepare full-domain bases on demand."""
        if getattr(self, 'full_recon_initialized', False):
            return
            
        dx, dy, dz = geom.dx, geom.dy, geom.dz
        nx, ny, nz = geom.nx, geom.ny, geom.nz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
        
        dx_rec, dy_rec, dz_rec = dx, dy, dz
        
        self.x_rec = ((cp.arange(nx+1)) * dx_rec).astype(cp.float32)
        self.y_rec = ((cp.arange(ny+1)) * dy_rec).astype(cp.float32)
        self.z_rec = ((cp.arange(nz+1)) * dz_rec).astype(cp.float32)

        m = cp.arange(nx)
        n = cp.arange(ny)
        p = cp.arange(nz)

        self.Bx_recon = (self.Cm[:, None] * cp.cos(cp.pi * m[:, None] * self.x_rec[None, :] / Lx)).astype(cp.float32)
        self.By_recon = (self.Cn[:, None] * cp.cos(cp.pi * n[:, None] * self.y_rec[None, :] / Ly)).astype(cp.float32)
        self.Bz_recon = (self.Cp[:, None] * cp.cos(cp.pi * p[:, None] * self.z_rec[None, :] / Lz)).astype(cp.float32)
        
        self.full_recon_initialized = True
        self.q_evap_old = cp.zeros((ny, nx), dtype=cp.float32)
        self.q_evap_buffer = cp.zeros((ny, nx), dtype=cp.float32)
        # ZYX layout
        self.Q_latent_buffer = cp.zeros((self.nz_box, self.ny_box, self.nx_box), dtype=cp.float32)


def precompute_K_KK(phys, num, geom):
    """
    Compute spectral Propagators (K, KK) on GPU.
    """
    # Use CuPy
    kx = (np.pi * cp.arange(num.nx) / geom.Lx)
    ky = (np.pi * cp.arange(num.ny) / geom.Ly)
    kz = (np.pi * cp.arange(num.nz) / geom.Lz)

    # meshgrid(..., indexing='ij')
    denom = phys.k / (phys.rho * phys.Cp) *(kx[None, None, :]**2 + ky[None, :, None]**2 + kz[:, None, None]**2)
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

def update_modes_etd1(aK, KK, Cp_broadcast, B_scaled, a_temp_out):
    """Wrapper for ETD1 kernel."""
    nz, ny, nx = aK.shape
    # Block dims
    threadsperblock = (8, 8, 8)
    blockspergrid = (
        (nz + threadsperblock[0] - 1) // threadsperblock[0],
        (ny + threadsperblock[1] - 1) // threadsperblock[1],
        (nx + threadsperblock[2] - 1) // threadsperblock[2]
    )
    update_modes_etd1_kernel[blockspergrid, threadsperblock](aK, KK, Cp_broadcast, B_scaled, a_temp_out)



def add_source_term_modes(a_temp, KK, Q_modes):
    """Wrapper for Source Term Accumulation."""
    nz, ny, nx = a_temp.shape
    threadsperblock = (8, 8, 8)
    blockspergrid = (
        (nz + threadsperblock[0] - 1) // threadsperblock[0],
        (ny + threadsperblock[1] - 1) // threadsperblock[1],
        (nx + threadsperblock[2] - 1) // threadsperblock[2]
    )
    add_source_term_modes_kernel[blockspergrid, threadsperblock](a_temp, KK, Q_modes)


def compute_evaporation_flux(T_surface, q_out, P0, R_gas, T_boil, DeltaH_LV, R_v, T_liquidus):
    """Wrapper for Evaporation kernel."""
    ny, nx = T_surface.shape
    threadsperblock = (16, 16)
    blockspergrid = (
        (ny + threadsperblock[0] - 1) // threadsperblock[0],
        (nx + threadsperblock[1] - 1) // threadsperblock[1]
    )
    compute_evaporation_flux_kernel[blockspergrid, threadsperblock](
        T_surface, q_out, P0, R_gas, T_boil, DeltaH_LV, R_v, T_liquidus
    )


def compute_gaussian_laser_flux(X, Y, laser_x, laser_y, laser_r, laser_coef):
    """Compute Gaussian flux on GPU grid X,Y."""
    # CuPy handles element-wise operations automatically
    r_sq = (X - laser_x) ** 2 + (Y - laser_y) ** 2
    return (laser_coef * cp.exp(-2.0 * r_sq / laser_r ** 2)).astype(cp.float32)


def project_box_to_modes(field_box, SsState):
    """Project fine box field to global spectral modes (GPU)."""
    if not SsState.fine_mesh_initialized:
        raise RuntimeError("Fine mesh not initialized.")
    
    # cp.tensordot is highly optimized
    # 1. Contract Z_box: (nz_box, ny_box, nx_box) . (nz, nz_box) -> (ny_box, nx_box, nz)
    temp1 = cp.tensordot(field_box, SsState.Bz_fine, axes=(0, 1))
    # 2. Contract Y_box: (ny_box, nx_box, nz) . (ny, ny_box) -> (nx_box, nz, ny)
    temp2 = cp.tensordot(temp1, SsState.By_fine, axes=(0, 1))
    # 3. Contract X_box: (nx_box, nz, ny) . (nx, nx_box) -> (nz, ny, nx)
    modes = cp.tensordot(temp2, SsState.Bx_fine, axes=(0, 1))
    
    modes *= SsState.dV_fine
    return modes.astype(cp.float32)


def reconstruct_temperature_box(a, SsState):
    """
    Reconstructs temperature in a small ROI around the laser (GPU).
    """
    if not SsState.fine_mesh_initialized:
        raise RuntimeError("Fine mesh not initialized.")
    
    # Tensor Contraction: Modes -> Physical Space
    T_step1 = cp.tensordot(a, SsState.Bz_fine, axes=(0, 0)) # Contraction over Z
    T_step2 = cp.tensordot(T_step1, SsState.By_fine, axes=(0, 0)) # Contraction over Y
    T_box = cp.tensordot(T_step2, SsState.Bx_fine, axes=(0, 0)) # Contraction over X
    
    return T_box.astype(cp.float32), (SsState.box_x, SsState.box_y, SsState.box_z)


def compute_latent_heat_source(Q_buffer, phys, x_laser, y_laser, num, SsState, alpha=0.4):
    """
    Compute volumetric latent heat source Q (W/m^3) on GPU.
    """
    # 1. Reconstruct Temperature on Fine Mesh
    T_box, _ = reconstruct_temperature_box(SsState.a_temp, SsState)
    
    # 2. Initialize/Retrieve State buffers
    if not hasattr(SsState, 'T_prev') or SsState.T_prev is None:
        SsState.T_prev = cp.zeros_like(T_box)
        SsState.T_prev[:] = T_box[:] 
        SsState.Q_prev = cp.zeros_like(Q_buffer)
        SsState.laser_x_prev = x_laser
        SsState.laser_y_prev = y_laser
        Q_buffer.fill(0.0)
        return

    # 3. Shift Previous Fields to Current Frame
    shift_x = x_laser - SsState.laser_x_prev
    shift_y = y_laser - SsState.laser_y_prev
    
    shift_pixels = (0, -shift_y / SsState.dy_fine, -shift_x / SsState.dx_fine)
    
    # Use cupyx.scipy.ndimage.shift
    T_prev_aligned = cupy_ndimage.shift(SsState.T_prev, shift_pixels, order=1, mode='nearest')
    Q_prev_aligned = cupy_ndimage.shift(SsState.Q_prev, shift_pixels, order=1, mode='constant', cval=0.0)
    
    # 4. Compute Source Term (Calls CUDA Kernel)
    nz_box, ny_box, nx_box = T_box.shape
    threadsperblock = (8, 8, 8)
    blockspergrid = (
        (nz_box + threadsperblock[0] - 1) // threadsperblock[0],
        (ny_box + threadsperblock[1] - 1) // threadsperblock[1],
        (nx_box + threadsperblock[2] - 1) // threadsperblock[2]
    )
    
    compute_source_term_kernel[blockspergrid, threadsperblock](
        T_box, T_prev_aligned, 
        phys.T_solidus, phys.T_liquidus, 
        phys.rho, phys.L_f, num.dt, 
        Q_buffer
    )
    
    # 5. Apply Relaxation
    if alpha < 1.0:
        Q_buffer[:] = alpha * Q_buffer + (1.0 - alpha) * Q_prev_aligned
    
    # 6. Update History
    SsState.T_prev[:] = T_box[:]
    SsState.Q_prev[:] = Q_buffer[:] 
    SsState.laser_x_prev = x_laser
    SsState.laser_y_prev = y_laser


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
    A = (SsState.Cp32_broadcast * a).sum(axis=0)
    
    # Use DCT-III (IDCT) for surface temperature
    dct_result = IDCT_II(A)
    return (SsState.recon_scale * dct_result).astype(cp.float32)


def reconstruct_temperature_xz(a, num, geom, SsState, laser, y0=None):
    """
    Optimized reconstruction of X-Z temperature slice using CuPy.
    """
    if y0 is None:
        y0 = laser.y0
        
    nx, ny, nz = num.nx, num.ny, num.nz
    
    # Evaluate cosine basis at specific y0 (ny,)
    # Warning: cp.arange returns array, ensure types match
    y_indices = cp.arange(ny, dtype=cp.float32)
    cos_y = SsState.Cn * cp.cos(np.pi * y_indices * y0 / geom.Ly)
    
    # Contract Y axis: (nz, ny, nx) dot (ny,) -> (nz, nx)
    A_xz = cp.tensordot(a, cos_y, axes=(1, 0)) 
    
    # Reconstruct X-Z field using 2D IDCT (Type 3) on axes (0, 1) corresponding to (nz, nx)
    # Note: dctn axes refer to the dimensions of A_xz
    scale_xz = cp.sqrt(nx * nz / (geom.Lx * geom.Lz))
    T_xz = scale_xz * cupy_fft.dctn(A_xz, type=3, norm='ortho', axes=(0, 1))
    
    return SsState.x, SsState.z, T_xz.astype(cp.float32)


def calculate_subgrid_indices(pos, dx, n_total_fine, n_box):
    """
    Calculate start/end indices. 
    Returns host integers (needed for slicing). 
    Calculation is scalar, so efficient on CPU.
    """
    idx_global = int(round(max(0.0, min(pos / dx, n_total_fine - 1))))
    
    target_center_offset = n_box // 2
    idx_start = idx_global - target_center_offset
    
    max_start = max(0, n_total_fine - n_box)
    idx_start = max(0, min(idx_start, max_start))
    idx_end = idx_start + n_box
    
    idx_relative = idx_global - idx_start
    idx_relative = max(0, min(idx_relative, n_box - 1))
    
    return idx_start, idx_end, idx_relative


def update_fine_mesh(SsState, x_laser, y_laser):
    """
    Update fine mesh box coordinates and basis subsets on GPU.
    Only x and y are updated since z is static (considering flat top).
    """
    # 1. Update X-Axis
    ix_start, ix_end, ix_laser_rel = calculate_subgrid_indices(
        x_laser, SsState.dx_fine, SsState.nx_fine_total, SsState.nx_box
    )
    
    # Slicing CuPy arrays with host integers is standard
    SsState.box_x[:] = SsState.x_fine[ix_start:ix_end]
    SsState.ix_laser_box = ix_laser_rel

    # Copy basis subset
    SsState.Bx_fine[:, :] = SsState.Bx_fine_full[:, ix_start:ix_end]
    
    # 2. Update Y-Axis
    iy_start, iy_end, _ = calculate_subgrid_indices(
        y_laser, SsState.dy_fine, SsState.ny_fine_total, SsState.ny_box
    )
    
    SsState.box_y[:] = SsState.y_fine[iy_start:iy_end]
    SsState.By_fine[:, :] = SsState.By_fine_full[:, iy_start:iy_end]

    # 3. Status
    SsState.fine_mesh_initialized = True


def shift_flux(field: cp.ndarray, shift: tuple, geom) -> cp.ndarray:
    """Translate a surface flux field by ``shift=(dx, dy)`` meters on GPU."""
    dx, dy = shift
    shift_pixels = (dy / geom.dy, dx / geom.dx)
    # cupyx shift
    cupy_ndimage.shift(field, shift_pixels, order=1, mode='constant', cval=0.0, output=field)
    return field