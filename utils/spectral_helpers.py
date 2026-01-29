import numpy as np
import os 
# TO DO - Refactor cpu - gpu dispatching logic here
import implementations.physics.spectral_cpu_kernels as kernels


def get_array_module(arr):
    return np

def C_coef(N, L, xp=np):
    """Compute normalization coefficients for DCT-II."""
    C = xp.sqrt(2.0 / L) * xp.ones(N)
    C[0] = xp.sqrt(1.0 / L)
    return C

def _cosine_basis_along_axis(n_modes, length, coords):
    """Compute cosine basis values cos(k*pi*x/L) for given coordinates."""
    indices = np.arange(n_modes, dtype=np.float64)
    return np.cos(np.pi * indices[:, None] * coords[None, :] / length)

def reconstruct_temperature_volume(a, SsState):
    """Reconstruct the temperature field on the full simulation grid."""
    xp = get_array_module(a)
    
    Bx = SsState.Bx_recon  # (modes_x, nx_points)
    By = SsState.By_recon  # (modes_y, ny_points)
    Bz = SsState.Bz_recon  # (modes_z, nz_points)

    T_step1 = xp.tensordot(a, Bx, axes=(2, 0))  # (nz, ny, nx)
    T_step2 = xp.tensordot(T_step1, By, axes=(1, 0))  # (nz, nx, ny)
    T_full = xp.tensordot(T_step2, Bz, axes=(0, 0))  # (nx, ny, nz)

    #T_step1 = xp.tensordot(a, Bz, axes=(0, 0))       # (ny_modes, nx_modes, nz_pts)
    #T_step2 = xp.tensordot(T_step1, By, axes=(0, 0)) # (nx_modes, nz_pts, ny_pts)
    #T_full = xp.tensordot(T_step2, Bx, axes=(0, 0))  # (nz_pts, ny_pts, nx_pts)

    return T_full.astype(np.float32)

def reconstruct_temperature_DCT(a, num, geom, SsState):
    return 

def reconstruct_temperature_volume_at_points(a, num, geom, SsState, coords):
    """Evaluate the temperature field at arbitrary points using modal expansion."""
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be of shape (N, 3)")

    x_vals = np.clip(coords[:, 0], 0.0, geom.Lx)
    y_vals = np.clip(coords[:, 1], 0.0, geom.Ly)
    z_vals = np.clip(coords[:, 2], 0.0, geom.Lz)

    # Ensure modal coefficient arrays and spectral coefficients are NumPy arrays
    # This avoids mixed NumPy/CuPy arithmetic when CPU-based helpers are used.
    def _to_numpy(x):
        # If x is a CuPy array with .get(), move to host; otherwise use np.asarray
        if hasattr(x, 'get') and callable(x.get):
            return np.asarray(x.get())
        return np.asarray(x)

    Cm = _to_numpy(getattr(SsState, 'Cm', None))
    Cn = _to_numpy(getattr(SsState, 'Cn', None))
    Cp = _to_numpy(getattr(SsState, 'Cp', None))
    a_np = _to_numpy(a).astype(np.float32)

    Bx = (Cm[:, None] * _cosine_basis_along_axis(num.nx, geom.Lx, x_vals)).astype(np.float32)
    By = (Cn[:, None] * _cosine_basis_along_axis(num.ny, geom.Ly, y_vals)).astype(np.float32)
    Bz = (Cp[:, None] * _cosine_basis_along_axis(num.nz, geom.Lz, z_vals)).astype(np.float32)
    temps = np.einsum('pnm,pi,ni,mi->i', a_np, Bz, By, Bx, optimize=True)

    return temps.astype(np.float32)


OUT_DIR = "out"

def save_temp_profiles(
    a,
    num,
    geom,
    SsState,
    laser_position,
    center="laser",
    num_points=1000):
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

    # 1. Determine Sample Center (Intersection Point)
    # Use the TOP surface (z = Lz) as reference
    z_top = float(geom.Lz)

    if center == "hotspot":
        # Scan low-res surface to find approximate max
        # This requires reconstructing a 2D slice first
        # For efficiency, we reconstruct T_surf from kernels
        T_surf = kernels.reconstruct_surface_temperature(a, SsState)
        iy_idx, ix_idx = np.unravel_index(np.argmax(T_surf), T_surf.shape)
        x_center = SsState.x[ix_idx]
        y_center = SsState.y[iy_idx]
        
    elif center == "laser":
        if laser_position is None:
            raise ValueError("Laser position required for center='laser'")
        x_center, y_center = float(laser_position[0]), float(laser_position[1])
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
    # Line along X through (y_c, z_top)
    points_x = np.column_stack((coords_x, np.full_like(coords_x, y_center), np.full_like(coords_x, z_top)))
    # Line along Y through (x_c, z_top)
    points_y = np.column_stack((np.full_like(coords_y, x_center), coords_y, np.full_like(coords_y, z_top)))
    # Line along Z through (x_c, y_c)
    points_z = np.column_stack((np.full_like(coords_z, x_center), np.full_like(coords_z, y_center), coords_z))

    # 4. Evaluate using Spectral Kernel
    # (reconstruct_temperature_volume_at_points should be available in kernels import or helper)
    # Using the one currently in helpers until refactor is 100% complete
    T_x = reconstruct_temperature_volume_at_points(a, num, geom, SsState, points_x)
    T_y = reconstruct_temperature_volume_at_points(a, num, geom, SsState, points_y)
    T_z = reconstruct_temperature_volume_at_points(a, num, geom, SsState, points_z)

    # 5. Return computed profiles as a dictionary
    return {
        'x': (coords_x, T_x),
        'y': (coords_y, T_y),
        'z': (coords_z, T_z)
    }
    
    


