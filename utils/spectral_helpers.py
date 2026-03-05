import numpy as np
import os 
from pyfftw.interfaces.scipy_fft import dctn

try:
    import cupy as cp
except ImportError:
    cp = None

# Default to CPU kernels for module-level access, but dispatch properly in functions
import implementations.physics.spectral_cpu_kernels as kernels


def _get_array_module(arr):
    if cp is not None and hasattr(arr, 'device'): # Check if it's a cupy array
        return cp
    return np

def _get_kernels(arr):
    xp = _get_array_module(arr)
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
    """Reconstruct the temperature field on the full simulation grid.
    
    note : the reconstruction bases (Bx, By, Bz) must be precomputed before, 
    The reconstruction grid is node centered in x, y, z
    """

    if hasattr(a, 'get'):
        a = a.get()  # Move to CPU if it's a CuPy array
    
    grid = SsState.grid

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

def reconstruct_temperature_DCT(a, SsState):
    """Reconstruct node-centered temperature field using DCT type I.

    The modal expansion is::

        T(x,y,z) = sum_{m,n,p} a[p,n,m] * Cm*cos(m*pi*x/Lx)
                                          * Cn*cos(n*pi*y/Ly)
                                          * Cp*cos(p*pi*z/Lz)

    evaluated on the **node-centered** grid  x_j = j*dx  (j = 0..nx),
    and likewise for y and z.  Output shape is **(nx+1, ny+1, nz+1)**,
    identical to :func:`reconstruct_temperature_volume`.

    **Why DCT-I?**
    DCT type 3 (``IDCT_II``) evaluates at half-integer (cell-centred)
    nodes.  The reconstruction grid is integer-spaced (node-centred),
    which maps to the DCT type I kernel::

        DCT-I of length M (unnorm):
        y[k] = x[0] + (-1)^k x[M-1] + 2 * sum_{n=1}^{M-2} x[n] cos(pi*n*k/(M-1))

    By padding N modes to length N+1 and halving interior modes, the
    DCT-I output equals the desired cosine series at node points.
    Complexity is O(N^3 log N) vs O(N^4) for explicit tensor products.

    Parameters
    ----------
    a : ndarray, shape (nz, ny, nx)
        Spectral coefficients (CuPy arrays are moved to CPU automatically).
    SsState : SpectralSolverState
        Must have ``grid.Cm``, ``grid.Cn``, ``grid.Cp`` normalization vectors.

    Returns
    -------
    T : ndarray, shape (nx+1, ny+1, nz+1), dtype float32
        Node-centred temperature field.
    """
    import pyfftw
    pyfftw.interfaces.cache.enable()

    if hasattr(a, 'get'):
        a = a.get()

    a = np.asarray(a, dtype=np.float32)
    grid = SsState.grid
    nz, ny, nx = a.shape

    # Normalization coefficients  Cm[0]=sqrt(1/L), Cm[m>=1]=sqrt(2/L)
    Cm = np.asarray(grid.Cm, dtype=np.float32)  # (nx,)
    Cn = np.asarray(grid.Cn, dtype=np.float32)  # (ny,)
    Cp = np.asarray(grid.Cp, dtype=np.float32)  # (nz,)

    # Step 1: weight by normalization
    # b[p,n,m] = a[p,n,m] * Cp[p] * Cn[n] * Cm[m]
    b = a * (Cp[:, None, None] * Cn[None, :, None] * Cm[None, None, :])

    # Step 2: per-axis halving for DCT-I input
    # w[0]=1, w[1:]=0.5  — factors multiply across axes
    wx = np.ones(nx, dtype=np.float32); wx[1:] = 0.5
    wy = np.ones(ny, dtype=np.float32); wy[1:] = 0.5
    wz = np.ones(nz, dtype=np.float32); wz[1:] = 0.5
    b *= wz[:, None, None] * wy[None, :, None] * wx[None, None, :]

    # Step 3: pad to (nz+1, ny+1, nx+1) with zeros
    padded = np.zeros((nz + 1, ny + 1, nx + 1), dtype=np.float32)
    padded[:nz, :ny, :nx] = b

    # Step 4: 3-D DCT-I (type 1, unnorm) → node-centred values
    T = pyfftw.interfaces.scipy_fft.dctn(
        padded, type=1, norm=None, axes=(0, 1, 2), workers=-1
    )

    # Step 5: transpose (nz+1, ny+1, nx+1) → (nx+1, ny+1, nz+1)
    T = T.transpose(2, 1, 0)

    return T.astype(np.float32)

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
    
    


