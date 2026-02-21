import numpy as np
import os 

try:
    import cupy as cp
except ImportError:
    cp = None

# Default to CPU kernels for module-level access, but dispatch properly in functions
import implementations.physics.spectral_cpu_kernels as kernels


def get_array_module(arr):
    if cp is not None and hasattr(arr, 'device'): # Check if it's a cupy array
        return cp
    return np

def _get_kernels(arr):
    xp = get_array_module(arr)
    if xp == cp:
         import implementations.physics.spectral_gpu_kernels as gpu_kernels
         return gpu_kernels
    return kernels


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
    if hasattr(a, 'get'):
        a = a.get()  # Move to CPU if it's a CuPy array
    
    # Support both old monolithic state and new decoupled state
    grid = SsState.grid if hasattr(SsState, 'grid') else SsState

    Bx = grid.Bx_recon  # (modes_x, nx_points)
    By = grid.By_recon  # (modes_y, ny_points)
    Bz = grid.Bz_recon  # (modes_z, nz_points)
    # Validate reconstruction bases

    if Bx is None or By is None or Bz is None:
        raise RuntimeError("Reconstruction bases not initialized on SsState. Call prepare_full_reconstruction()/full_reconstruction() first.")

    T_step1 = np.tensordot(a, Bx, axes=(2, 0))  # (nz, ny, nx)
    T_step2 = np.tensordot(T_step1, By, axes=(1, 0))  # (nz, nx, ny)
    T_full = np.tensordot(T_step2, Bz, axes=(0, 0))  # (nx, ny, nz)

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
    
    # Support both old monolithic state and new decoupled state
    grid = SsState.grid if hasattr(SsState, 'grid') else SsState

    Cm = _to_numpy(getattr(grid, 'Cm', None))
    Cn = _to_numpy(getattr(grid, 'Cn', None))
    Cp = _to_numpy(getattr(grid, 'Cp', None))
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
        loc_kernels = _get_kernels(a)
        T_surf = loc_kernels.reconstruct_surface_temperature(a, SsState)
        
        # Ensure T_surf is on CPU for coordinate extraction
        if hasattr(T_surf, 'get'):
            T_surf = T_surf.get()
            
        iy_idx, ix_idx = np.unravel_index(np.argmax(T_surf), T_surf.shape)
        
        # Handle decoupled state or monolithic state
        grid = SsState.grid if hasattr(SsState, 'grid') else SsState
        x_center = grid.x[ix_idx] 
        y_center = grid.y[iy_idx]
        
        # Helper to safely scalarize
        def _scalar(val):
            if hasattr(val, 'item'): return val.item()
            return val
            
        x_center = _scalar(x_center)
        y_center = _scalar(y_center)
        
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
    
    


