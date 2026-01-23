import numpy as np
from pyfftw.interfaces.scipy_fft import dctn
import os 
# TO DO - Refactor cpu - gpu dispatching logic here
import implementations.physics.spectral_cpu_kernels as kernels

try:
    import cupy as cp
    import cupyx.scipy.fft as cupy_fft
except ImportError:
    cp = None

def get_array_module(arr):
    if cp is not None:
        return cp.get_array_module(arr)
    return np

def C_coef(N, L, xp=np):
    """Compute normalization coefficients for DCT-II."""
    C = xp.sqrt(2.0 / L) * xp.ones(N)
    C[0] = xp.sqrt(1.0 / L)
    return C

def DCT_II(q):
    """Apply Discrete Cosine Transform Type II (Ortho)."""
    xp = get_array_module(q)
    if cp is not None and xp == cp:
        return cupy_fft.dctn(q, type=2, norm='ortho').astype(np.float32, copy=False)
    return dctn(q.astype(np.float32, copy=False), type=2, norm='ortho', workers=-1).astype(np.float32, copy=False)

def _cosine_basis_along_axis(n_modes, length, coords):
    """Compute cosine basis values cos(k*pi*x/L) for given coordinates."""
    indices = np.arange(n_modes, dtype=np.float64)
    return np.cos(np.pi * indices[:, None] * coords[None, :] / length)

def reconstruct_temperature_volume(a, SsState):
    """Reconstruct the temperature field on the full simulation grid."""
    xp = get_array_module(a)
    # To be implemented later, proper DCT-based reconstruction for full volume
    Bx = (SsState.Cm[:, None] * SsState.cos_mx).astype(np.float32)
    By = (SsState.Cn[:, None] * SsState.cos_ny).astype(np.float32)
    Bz = (SsState.Cp[:, None] * SsState.cos_pz).astype(np.float32)

    T_step1 = xp.tensordot(a, Bx, axes=(2, 0))  # (nz, ny, nx)
    T_step2 = xp.tensordot(T_step1, By, axes=(1, 0))  # (nz, nx, ny)
    T_full = xp.tensordot(T_step2, Bz, axes=(0, 0))  # (nx, ny, nz)

    return T_full.astype(np.float32)

def reconstruct_temperature_volume_at_points(a, num, geom, SsState, coords):
    """Evaluate the temperature field at arbitrary points using modal expansion."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be of shape (N, 3)")

    x_vals = np.clip(coords[:, 0], 0.0, geom.Lx)
    y_vals = np.clip(coords[:, 1], 0.0, geom.Ly)
    z_vals = np.clip(coords[:, 2], 0.0, geom.Lz)

    Bx = (SsState.Cm[:, None] * _cosine_basis_along_axis(num.nx, geom.Lx, x_vals)).astype(np.float64)
    By = (SsState.Cn[:, None] * _cosine_basis_along_axis(num.ny, geom.Ly, y_vals)).astype(np.float64)
    Bz = (SsState.Cp[:, None] * _cosine_basis_along_axis(num.nz, geom.Lz, z_vals)).astype(np.float64)

    temps = np.einsum('pnm,pi,ni,mi->i', a.astype(np.float64), Bz, By, Bx, optimize=True)

    return temps.astype(np.float32)


def precompute_K_KK(phys, num, geom):
    """
    Compute spectral Propagators (K, KK) based on grid and time step.
    K = 1 / (1 + alpha*dt/2 * k^2) ... (Semi-implicit propagator)
    KK = dt / (rho * Cp) ...          (Source propagator)
    """
    xp = np # Default to numpy for precalc
    if hasattr(num, 'use_gpu') and num.use_gpu and cp is not None:
         xp = cp

    kx = (np.pi * xp.arange(num.nx) / geom.Lx)
    ky = (np.pi * xp.arange(num.ny) / geom.Ly)
    kz = (np.pi * xp.arange(num.nz) / geom.Lz)

    KX, KY, KZ = xp.meshgrid(kx, ky, kz, indexing='ij') # Note: meshgrid indexing might need check vs implementation
    k2 = (kx[None, None, :]**2 + ky[None, :, None]**2 + kz[:, None, None]**2)
    
    # Thermal Diffusivity alpha = k / (rho * Cp)
    alpha = phys.k / (phys.rho * phys.Cp)
    
    # Backward Euler / Crank-Nicolson Factor
    # If ETD1 or CN, standard form is usually:
    # K = exp(-alpha * k^2 * dt) -- for ETD1 (Exact integration of linear part)
    K = xp.exp(-alpha * k2 * num.dt).astype(np.float32)
    
    # Source Logic for ETD1: define integral factor (phi_1 function)
    # phi_1(z) = (exp(z) - 1) / z
    # Here z = -alpha * k2 * num.dt
    # So KK = (K - 1) / (-alpha*k2) / (rho*Cp)
    # Careful at k=0 (z=0) -> phi_1(0) = 1
    
    # Denominator for source term integration
    denom = alpha * k2
    
    # Avoid div by zero at k=0
    mask_zero = (denom == 0)
    denom[mask_zero] = 1.0 # arbitrary non-zero
    
    # Phi_1 factor * dt / (rho*Cp)
    # Standard ETD1: a_{n+1} = a_n * exp(L*dt) + N(a_n) * (exp(L*dt)-1)/L
    # Here source term Q is in W/m^3. Equation: dT/dt = alpha*Laplacian(T) + Q/(rho*Cp)
    # Modes: da/dt = -alpha*k^2*a + q_modes/(rho*Cp)
    # Solution: a(t+dt) = a(t)*e^z + (q/(rho*Cp)) * (e^z - 1)/(-alpha*k^2)
    
    phi_1 = (K - 1.0) / (-denom)
    phi_1[mask_zero] = num.dt # Limit as k->0 is just dt
    
    KK = (phi_1 / (phys.rho * phys.Cp)).astype(np.float32)
    
    # Return K (Decay), KK (Source Propagator)
    # Ensure correct shape (nz, ny, nx) match
    # k2 was (nz, ny, nx) if constructed carefully??
    # Current broadcasting: kz(:, None, None) + ky(...) + kx(...) -> (nz, ny, nx)
    
    return K, KK

# goes to spectral_helpers.py
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
def update_fine_mesh(SsState, laser):
    """
    Update fine mesh box coordinates and basis subsets so the laser remains centered.
    """
    # 1. Update X-Axis (Scanning Direction)
    ix_start, ix_end, ix_laser_rel = calculate_subgrid_indices(
        laser.x, SsState.dx_fine, SsState.nx_fine_total, SsState.nx_box
    )
    
    SsState.box_x[:] = SsState.x_fine[ix_start:ix_end]
    SsState.ix_laser_box = ix_laser_rel

    # Helper to slice and copy bases safely (handles non-contiguous memory)
    # Bx_fine shape: (nx_global, nx_box)
    SsState.Bx_fine[:, :] = SsState.Bx_fine_full[:, ix_start:ix_end]
    
    # 2. Update Y-Axis (Transverse Direction)
    iy_start, iy_end, _ = calculate_subgrid_indices(
        laser.y, SsState.dy_fine, SsState.ny_fine_total, SsState.ny_box
    )
    
    SsState.box_y[:] = SsState.y_fine[iy_start:iy_end]
    # By_fine shape: (ny_global, ny_box)
    SsState.By_fine[:, :] = SsState.By_fine_full[:, iy_start:iy_end]

    # 3. Update Volume Metric & Status
    SsState.dV_fine = SsState.dx_fine * SsState.dy_fine * SsState.dz_fine
    SsState.fine_mesh_initialized = True

OUT_DIR = "out"

def save_temp_profiles(
    a,
    num,
    geom,
    SsState,
    laser,
    center="laser",
    num_points=1000,
    output_dir=OUT_DIR):
    """
    Sample temperature along dense lines using exact modal expansion.
    
    Args:
        a: Spectral coefficients (nz, ny, nx).
        num: Numerical params (nx, ny, nz).
        geom: Geometric params (Lx, Ly, Lz).
        SsState: Spectral Solver State.
        laser: Laser object (for centering).
        center: "laser", "hotspot", or tuple (x, y).
        num_points: Number of sampling points along each axis.
        output_dir: Directory to save .txt files.
    """
    if any(v is None for v in (a, num, geom, SsState)):
        raise ValueError("Missing required objects (a, num, geom, SsState).")

    # 1. Determine Sample Center (Intersection Point)
    z_top = 0.0 # Surface

    if center == "hotspot":
        # Scan low-res surface to find approximate max
        # This requires reconstructing a 2D slice first
        # For efficiency, we reconstruct T_surf from kernels
        T_surf = kernels.reconstruct_surface_temperature(a, SsState)
        iy_idx, ix_idx = np.unravel_index(np.argmax(T_surf), T_surf.shape)
        x_center = SsState.x[ix_idx]
        y_center = SsState.y[iy_idx]
        
    elif center == "laser":
        if laser is None:
            raise ValueError("Laser object required for center='laser'")
        x_center, y_center = float(laser.x), float(laser.y)
    elif isinstance(center, (tuple, list, np.ndarray)):
        x_center, y_center = float(center[0]), float(center[1])
    else:
        raise ValueError(f"Unknown center method: {center}")

    # Clamp to domain
    x_center = float(np.clip(x_center, 0.0, geom.Lx))
    y_center = float(np.clip(y_center, 0.0, geom.Ly))

    # 2. Generate Dense Sampling Coordinates
    coords_x = np.linspace(0.0, geom.Lx, num_points, dtype=np.float64)
    coords_y = np.linspace(0.0, geom.Ly, num_points, dtype=np.float64)
    coords_z = np.linspace(0.0, geom.Lz, num_points, dtype=np.float64)

    # 3. Create Point Clouds for Batched Evaluation
    # Line along X through (y_c, 0)
    points_x = np.column_stack((coords_x, np.full_like(coords_x, y_center), np.full_like(coords_x, z_top)))
    # Line along Y through (x_c, 0)
    points_y = np.column_stack((np.full_like(coords_y, x_center), coords_y, np.full_like(coords_y, z_top)))
    # Line along Z through (x_c, y_c)
    points_z = np.column_stack((np.full_like(coords_z, x_center), np.full_like(coords_z, y_center), coords_z))

    # 4. Evaluate using Spectral Kernel
    # (reconstruct_temperature_volume_at_points should be available in kernels import or helper)
    # Using the one currently in helpers until refactor is 100% complete
    T_x = reconstruct_temperature_volume_at_points(a, num, geom, SsState, points_x)
    T_y = reconstruct_temperature_volume_at_points(a, num, geom, SsState, points_y)
    T_z = reconstruct_temperature_volume_at_points(a, num, geom, SsState, points_z)

    # 5. Save Results
    os.makedirs(output_dir, exist_ok=True)
    # TO DO WE SHOULDNT DO IO HERE
    for direction, coords, profile in [
        ('x_fine', coords_x, T_x),
        ('y_fine', coords_y, T_y),
        ('z_fine', coords_z, T_z)
    ]:
        fname = os.path.join(output_dir, f"{direction}_spectral_latent_heat.txt")
        # Format: Coord [m] | Temp [K]
        np.savetxt(fname, np.vstack([coords, profile]).T, header=f'{direction}(m) T(K)', fmt='% .6e')
        
    print(f"Saved fine temp profiles centered at ({x_center:.2e}, {y_center:.2e}) to {output_dir}")

    


