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

    


