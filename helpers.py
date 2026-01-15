import h5py
import os
import time
import numpy as np
from scipy.ndimage import shift as scipy_shift
from scipy.interpolate import RegularGridInterpolator
from pyfftw.interfaces.scipy_fft import dctn
import pyfftw
from numba import njit, prange

OUT_DIR = "out"

def save_field_to_hdf5(filename_base, field, grid_coords, value_name="Field", geom=None, verbose=False):
        """Serialize a 3D scalar field to HDF5 with an accompanying XDMF wrapper."""
        h5_name = f"{filename_base}.h5"
        xmf_name = f"{filename_base}.xmf"
        h5_ref = os.path.basename(h5_name)

        x_coords, y_coords, z_coords = grid_coords
        nz, ny, nx = field.shape

        if verbose and geom is not None:
                print(f"Exporting HDF5/XDMF. Domain Size: {geom.Lx:.2e} x {geom.Ly:.2e} x {geom.Lz:.2e}")

        with h5py.File(h5_name, "w") as f:
                f.create_dataset("X", data=x_coords)
                f.create_dataset("Y", data=y_coords)
                f.create_dataset("Z", data=z_coords)
                f.create_dataset(value_name, data=field)

        xmf_content = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
     <Grid Name="Mesh" GridType="Uniform">
         <Topology TopologyType="3DRectMesh" Dimensions="{nz} {ny} {nx}"/>
         <Geometry GeometryType="VXVYVZ">
             <DataItem Dimensions="{nx}" NumberType="Float" Precision="4" Format="HDF">
                {h5_ref}:/X
             </DataItem>
             <DataItem Dimensions="{ny}" NumberType="Float" Precision="4" Format="HDF">
                {h5_ref}:/Y
             </DataItem>
             <DataItem Dimensions="{nz}" NumberType="Float" Precision="4" Format="HDF">
                {h5_ref}:/Z
             </DataItem>
         </Geometry>
         <Attribute Name="{value_name}" AttributeType="Scalar" Center="Node">
             <DataItem Dimensions="{nz} {ny} {nx}" NumberType="Float" Precision="4" Format="HDF">
                {h5_ref}:/{value_name}
             </DataItem>
         </Attribute>
     </Grid>
 </Domain>
</Xdmf>
"""
        with open(xmf_name, "w") as f:
                f.write(xmf_content)

        if verbose:
                print(f"Saved debug files: {xmf_name} (Open this in Paraview)")


# ============================================================
#   COSINE NORMALIZATION COEFFICIENTS
# ============================================================

def C_coef(N, L):
    C = np.sqrt(2.0 / L) * np.ones(N)
    C[0] = np.sqrt(1.0 / L)
    return C


# ============================================================
#   ETD PHI FUNCTIONS
# ============================================================

def phi_functions(z):
    """ Functions used for the time stepping, taking into account exponential decay"""
    small_threshold = 1e-6
    phi_0 = np.exp(z)
    
    mask_small = np.abs(z) < small_threshold
    phi_1 = np.zeros_like(z)
    phi_1[~mask_small] = (np.exp(z[~mask_small]) - 1.0) / z[~mask_small]
    phi_1[mask_small] = 1.0 + z[mask_small] / 2.0 + z[mask_small]**2 / 6.0
    
    phi_2 = np.zeros_like(z)
    phi_2[~mask_small] = (np.exp(z[~mask_small]) - 1.0 - z[~mask_small]) / (z[~mask_small]**2)
    phi_2[mask_small] = 0.5 + z[mask_small] / 6.0 + z[mask_small]**2 / 24.0
    
    return phi_0, phi_1, phi_2


# ============================================================
#   HEAT FLUX
# ============================================================

def q_laser(geom, laser):
    """Gaussian laser heat flux using current Laser position (laser.x, laser.y)."""
    r_sq = (geom.X - laser.x) ** 2 + (geom.Y - laser.y) ** 2
    return (geom.laser_coef * np.exp(-2.0 * r_sq / laser.r_b ** 2)).astype(np.float32)


def q_evap_point(T: np.ndarray, phys) -> np.ndarray:
    """Evaporative heat flux."""
    q = 0.82 * phys.DeltaH_LV * phys.Pa/ np.sqrt(2 * np.pi * phys.R_v * T) * \
        np.exp((phys.DeltaH_LV / (phys.R_v * phys.T_boil)) * (1.0 - phys.T_boil / T))
    q[T < phys.T_liquidus] = 0.0
    return q.astype(np.float32)


def shift_flux(field: np.ndarray, shift: tuple, geom) -> np.ndarray:
    """Translate a surface flux field by ``shift=(dx, dy)`` meters."""
    if field is None or field.size == 0:
        return np.zeros((geom.ny, geom.nx), dtype=np.float32)
    dx, dy = shift
    shift_pixels = (dy / geom.dy, dx / geom.dx)
    return scipy_shift(field, shift_pixels, order=1, mode='constant', cval=0.0).astype(np.float32) 


# ============================================================
#   DCT-II 2D 
# ============================================================

def DCT_II(q):
    return dctn(q.astype(np.float32, copy=False), type=2, norm='ortho', workers=-1).astype(np.float32, copy=False)

# ============================================================
#   LATENT HEAT SOURCE HELPERS
# ============================================================


@njit(parallel=True, fastmath=True)
def _compute_latent_heat_gradient(Q_box, T_box, dx, dy, dz, rho, L, v, T_S, T_L, dt):
    """
    Compute latent heat source from temperature gradient.
    Q = rho * L * v * d(f_solid)/dT * dT/dx
    Assumes arrays are (nz, ny, nx) [ZYX layout].
    """
    nz, ny, nx = T_box.shape
    
    # dfs/dT = -1 / (TL - TS) in the mushy zone
    dfs_dT = -1.0 / (T_L - T_S)
    
    # Central difference factor: 1/(2*dx)
    # Total factor
    factor = rho * L * v * dfs_dT / (2.0 * dx)
    
    # Physical limit (Maximum tolerated power corresponding to full phase change)
    Q_max = rho *  L / dt

    for z in prange(nz):
        Q_box[z, :, :] = 0.0
        for y in range(ny):
            # Interior X
            # Fix: range must prevent out-of-bounds access at T_box[..., x+1]. 
            # Max valid index is nx-1, so max x is nx-2.
            for x in range(nx - 2, 0, -1):
                T = T_box[z, y, x]
                if T >= T_S and T <= T_L:
                    dTdx = T_box[z, y, x+1] - T_box[z, y, x-1]
                    val = -factor * dTdx
                    # Clamp to physical limit
                    if abs(val) + abs(sum(Q_box[z, y, x:x+10])) > Q_max:
                        if val > 0:
                            #print("Latent heat limit reached (+):", val, ">", Q_max)
                            val = Q_max - abs(sum(Q_box[z, y, x:x+10]))
                        if val < 0:
                            #print("Latent heat limit reached (-):", val, "<", -Q_max)
                            val = -Q_max + abs(sum(Q_box[z, y, x:x+10]))
                    Q_box[z, y, x] = val

def compute_latent_heat_source(Q_buffer, box_coords, phys, laser, geom, num, timers=None):
    """
    Compute volumetric latent heat source Q (W/m^3) based on the temperature gradient.
    Formula: Q = rho * L * v * d(f_solid)/dT * dT/dx
    """
    # 1. Reconstruct Temperature on Fine Mesh
    # T_box should be returned in (nz, ny, nx) layout to match Q_buffer
    T_box, _ = reconstruct_temperature_box(num.a_temp, num, geom, timers=timers)
    
    # 2. Physics Parameters
    v = np.linalg.norm(laser.v)
    dx = geom.dx_fine
    dy = geom.dy_fine
    dz = geom.dz_fine
    
    # 3. Compute Gradient and Source
    # Q_buffer is (nz, ny, nx)
    t0 = time.perf_counter()
    _compute_latent_heat_gradient(
        Q_buffer, T_box, 
        dx, dy, dz, phys.rho, phys.L_f, v, 
        phys.T_solidus, phys.T_liquidus,
        num.dt
    )
    if timers is not None: timers['lh_gradient'] += time.perf_counter() - t0


# ============================================================
#   RECONSTRUCT TEMPERATURE FIELDS HELPER FUNCTIONS
# ============================================================

def reconstruct_temperature_box(a, num, geom, symmetrize=False, timers=None):
    """
    Reconstructs temperature in a small ROI around the laser using 
    precomputed fine mesh cosine bases from geom.
    """
    if not geom.fine_mesh_initialized:
        raise RuntimeError("Fine mesh not initialized. Call geom.update_fine_mesh(laser) first.")
    
    # Tensor Contraction using precomputed bases
    # T(x,y,z) = sum_p sum_n sum_m  a[p,n,m] * Bz[p,z] * By[n,y] * Bx[m,x]
    
    # 1. Contract Z: (nz, ny, nx) . (nz, nz_box) -> (ny, nx, nz_box)
    t0 = time.perf_counter()
    T_step1 = np.tensordot(a, geom.Bz_fine, axes=(0, 0))
    if timers is not None: timers['lh_recon_step1'] += time.perf_counter() - t0
    
    # 2. Contract Y: (ny, nx, nz_box) . (ny, ny_box) -> (nx, nz_box, ny_box)
    t0 = time.perf_counter()
    T_step2 = np.tensordot(T_step1, geom.By_fine, axes=(0, 0))
    if timers is not None: timers['lh_recon_step2'] += time.perf_counter() - t0
    
    # 3. Contract X: (nx, nz_box, ny_box) . (nx, nx_box) -> (nz_box, ny_box, nx_box)
    t0 = time.perf_counter()
    T_box = np.tensordot(T_step2, geom.Bx_fine, axes=(0, 0))
    if timers is not None: timers['lh_recon_step3'] += time.perf_counter() - t0
    
    return T_box.astype(np.float32), (geom.box_x, geom.box_y, geom.box_z)


def reconstruct_temperature_top(a, num, geom):
    """Reconstruct top surface temperature from modal coefficients."""
    A = (geom.Cp32[:, None, None] * a).sum(axis=0)
    return (geom.recon_scale * dctn(A, type=3, norm='ortho', axes=(0, 1), workers=-1)).astype(np.float32, copy=False)

def reconstruct_temperature_xz(a, num, geom, phys, laser, y0=None):
    """
    Optimized reconstruction of X-Z temperature slice using FFTW/DCT.
    Returns (x_vals, z_vals, T_xz).
    """
    y0 = laser.y0
    nx, ny, nz = num.nx, num.ny, num.nz
    # Evaluate cosine basis at specific y0 (ny,)
    cos_y = geom.Cn * np.cos(np.pi * np.arange(ny) * y0 / geom.Ly)
    # Contract Y axis: (nz, ny, nx) dot (ny,)
    A_xz = np.tensordot(a, cos_y, axes=(1, 0)) 
    # Reconstruct X-Z field using 2D IDCT (Type 3)
    scale_xz = np.sqrt(nx * nz / (geom.Lx * geom.Lz))
    T_xz = scale_xz * dctn(A_xz, type=3, norm='ortho', axes=(0, 1))
    
    return geom.x, geom.z, T_xz.astype(np.float32)



def reconstruct_temperature_volume(a, num, geom):
    """Reconstruct the temperature field on the full simulation grid."""
    # To be implemented later, proper DCT-based reconstruction for full volume
    Bx = (geom.Cm[:, None] * geom.cos_mx).astype(np.float32)
    By = (geom.Cn[:, None] * geom.cos_ny).astype(np.float32)
    Bz = (geom.Cp[:, None] * geom.cos_pz).astype(np.float32)

    T_step1 = np.tensordot(a, Bx, axes=(2, 0))  # (nz, ny, nx)
    T_step2 = np.tensordot(T_step1, By, axes=(1, 0))  # (nz, nx, ny)
    T_full = np.tensordot(T_step2, Bz, axes=(0, 0))  # (nx, ny, nz)

    return T_full.astype(np.float32)


def _cosine_basis_along_axis(n_modes, length, coords):
    """Compute cosine basis values for given coordinates along one axis."""
    indices = np.arange(n_modes, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    return np.cos(np.pi * indices[:, None] * coords[None, :] / length)


def reconstruct_temperature_volume_at_points(a, num, geom, coords):
    """Evaluate the temperature field at arbitrary points using modal expansion."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be of shape (N, 3)")

    x_vals = np.clip(coords[:, 0], 0.0, geom.Lx)
    y_vals = np.clip(coords[:, 1], 0.0, geom.Ly)
    z_vals = np.clip(coords[:, 2], 0.0, geom.Lz)

    Bx = (geom.Cm[:, None] * _cosine_basis_along_axis(num.nx, geom.Lx, x_vals)).astype(np.float64)
    By = (geom.Cn[:, None] * _cosine_basis_along_axis(num.ny, geom.Ly, y_vals)).astype(np.float64)
    Bz = (geom.Cp[:, None] * _cosine_basis_along_axis(num.nz, geom.Lz, z_vals)).astype(np.float64)

    temps = np.einsum('pnm,pi,ni,mi->i', a.astype(np.float64), Bz, By, Bx, optimize=True)

    return temps.astype(np.float32)

def save_temp_profiles(a, num, geom, phys, laser, t=None, center="laser"):
    """
    Save 1D temperature profiles intersecting at the specified center.
    Coordinates generated start at 0.0 (aligned with save_temp_profiles_fine).
    Using explicit tensor contraction for speed.
    """
    t = num.t_final if t is None else t
    
    # 1. Coordinate Arrays (Explicit float64 for safety)
    x_line = np.linspace(0.0, geom.Lx, num.nx, dtype=np.float64)
    y_line = np.linspace(0.0, geom.Ly, num.ny, dtype=np.float64)
    z_line = np.linspace(0.0, geom.Lz, num.nz, dtype=np.float64)

    # 2. Determine Center Coordinates
    if center == "hotspot":
        # Simplified: Fallback to center of domain if hotspot logic not robust without full reconstruct
        # Ideally would scan low-res modes, but let's stick to laser or center-of-domain
         x_center, y_center = 0.5 * geom.Lx, 0.5 * geom.Ly
    elif center == "laser":
        x_center, y_center = float(laser.x), float(laser.y)
    else:
        x_center, y_center = float(center[0]), float(center[1])

    z_top = 0.0 # default 

    # 3. Helpers for Basis Evaluation
    def eval_cos(N, L, vals):
        """Evaluate cos(k*pi*x/L) for all k=0..N-1 and given vals."""
        # ids shape: (N, 1)
        ids = np.arange(N, dtype=np.float64)[:, None]
        # vals shape: (1, M)
        # Returns (N, M)
        return np.cos(np.pi * ids * vals[None, :] / L)

    # Precompute basis vectors at intersection point (x_c, y_c, z_c)
    # Shapes: (Nx,), (Ny,), (Nz,)
    phi_x_c = eval_cos(num.nx, geom.Lx, np.array([x_center], dtype=np.float64)).flatten()
    phi_y_c = eval_cos(num.ny, geom.Ly, np.array([y_center], dtype=np.float64)).flatten()
    phi_z_c = eval_cos(num.nz, geom.Lz, np.array([z_top], dtype=np.float64)).flatten()

    # Pre-multiply by Normalization Coefficients C_k
    # Effective projection vector K = C_k * phi_c
    Kx_c = phi_x_c * geom.Cm
    Ky_c = phi_y_c * geom.Cn
    Kz_c = phi_z_c * geom.Cp

    # --- X Profile ( at y=y_c, z=z_top ) ---
    # T(x) = sum_m [ sum_n sum_p a_pnm * Kz_p * Ky_n ] * (Cm_m * cos_m(x))
    # 1. Contract Z (axis 0 of a): Result (ny, nx)
    a_yx = np.tensordot(a, Kz_c, axes=(0, 0)) 
    # 2. Contract Y (axis 0 of a_yx): Result (nx,)
    a_x = np.dot(Ky_c, a_yx)
    # 3. Evaluate along line
    Bx_line = eval_cos(num.nx, geom.Lx, x_line) # (nx, points)
    T_x = (a_x * geom.Cm) @ Bx_line

    # --- Y Profile ( at x=x_c, z=z_top ) ---
    # Use a_yx from above
    # 1. Contract X (axis 1 of a_yx): Result (ny,)
    a_y = np.dot(a_yx, Kx_c)
    # 2. Evaluate along line
    By_line = eval_cos(num.ny, geom.Ly, y_line)
    T_y = (a_y * geom.Cn) @ By_line

    # --- Z Profile ( at x=x_c, y=y_c ) ---
    # 1. Contract X (axis 2 of a): Result (nz, ny)
    a_zy = np.tensordot(a, Kx_c, axes=(2, 0))
    # 2. Contract Y (axis 1 of a_zy): Result (nz,)
    a_z = np.dot(a_zy, Ky_c)
    # 3. Evaluate along line
    Bz_line = eval_cos(num.nz, geom.Lz, z_line)
    T_z = (a_z * geom.Cp) @ Bz_line

    # 4. Save
    for direction, coords, profile in [
        ('x', x_line, T_x),
        ('y', y_line, T_y),
        ('z', z_line, T_z)
    ]:
        fname = f"{OUT_DIR}/{direction}_spectral_latent_heat.txt"
        np.savetxt(fname, np.vstack([coords, profile]).T, header=f'{direction}(m) T(K)', fmt='% .6e')
        print(f"Saved: {fname}")

def save_temp_profiles_from_hdf5(h5_path, center="laser", laser_position=None, output_dir=OUT_DIR):
    """Re-sample temperature profiles from an HDF5 volume using tri-linear interpolation."""
    with h5py.File(h5_path, "r") as f:
        x = f["X"][:]
        y = f["Y"][:]
        z = f["Z"][:]
        T = f["Temperature"][:]

    interpolator = RegularGridInterpolator((z, y, x), T, bounds_error=False, fill_value=np.nan)

    top_idx = int(np.argmin(np.abs(z)))
    z_top = z[top_idx]

    if center == "hotspot":
        top_plane = T[top_idx, :, :]
        iy_idx, ix_idx = np.unravel_index(np.nanargmax(top_plane), top_plane.shape)
        y_center = y[iy_idx]
        x_center = x[ix_idx]
        print(
            "Hotspot center detected at "
            f"x={x_center:.6e} m, y={y_center:.6e} m, z={z_top:.6e} m"
        )
    elif center == "laser":
        if laser_position is None:
            raise ValueError("laser_position must be provided when center='laser'.")
        x_center = float(np.clip(laser_position[0], x[0], x[-1]))
        y_center = float(np.clip(laser_position[1], y[0], y[-1]))
    else:
        raise ValueError(f"Unknown center option: {center}")

    line_x_points = np.column_stack((np.full(len(x), z_top, dtype=np.float64),
                                     np.full(len(x), y_center, dtype=np.float64),
                                     x.astype(np.float64)))
    line_y_points = np.column_stack((np.full(len(y), z_top, dtype=np.float64),
                                     y.astype(np.float64),
                                     np.full(len(y), x_center, dtype=np.float64)))
    line_z_points = np.column_stack((z.astype(np.float64),
                                     np.full(len(z), y_center, dtype=np.float64),
                                     np.full(len(z), x_center, dtype=np.float64)))

    T_x = interpolator(line_x_points).astype(np.float32)
    T_y = interpolator(line_y_points).astype(np.float32)
    T_z = interpolator(line_z_points).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)

    for direction, coords, profile in [
        ('x', x, T_x),
        ('y', y, T_y),
        ('z', z, T_z)
    ]:
        fname = os.path.join(output_dir, f"{direction}_spectral_latent_heat.txt")
        np.savetxt(fname, np.vstack([coords, profile]).T, header=f'{direction}(m) T(K)', fmt='% .6e')
        print(f"Saved: {fname}")
        

def save_temp_profiles_fine(
    a=None,
    num=None,
    geom=None,
    laser=None,
    coords_x=None,
    coords_y=None,
    coords_z=None,
    center="laser",
    output_dir=OUT_DIR,
    volume=None,
    grid_coords=None,
):
    """
    Sample temperature along dense lines either from modal coefficients or a stored volume.

    Provide either (a, num, geom) for exact modal evaluation or (volume, grid_coords).
    """
    if volume is None:
        if any(v is None for v in (a, num, geom)):
            raise ValueError("Provide modal data (a, num, geom) when volume is not supplied.")
        x_axis, y_axis, z_axis = geom.x, geom.y, geom.z
    else:
        if grid_coords is None:
            raise ValueError("grid_coords must be provided with volume data.")
        x_axis, y_axis, z_axis = grid_coords

    coords_x = np.asarray(coords_x if coords_x is not None else np.linspace(0.0, x_axis[-1], 1000, endpoint=False), dtype=np.float64)
    coords_y = np.asarray(coords_y if coords_y is not None else np.linspace(0.0, y_axis[-1], 1000, endpoint=False), dtype=np.float64)
    coords_z = np.asarray(coords_z if coords_z is not None else np.linspace(0.0, z_axis[-1], 1000), dtype=np.float64)

    top_idx = int(np.argmin(np.abs(z_axis)))
    z_top = z_axis[top_idx]

    if center == "hotspot":
        if volume is None:
            T_top = reconstruct_temperature_top(a, num, geom)
        else:
            T_top = volume[top_idx, :, :]
        iy_idx, ix_idx = np.unravel_index(np.nanargmax(T_top), T_top.shape)
        x_center = x_axis[ix_idx]
        y_center = y_axis[iy_idx]
    elif center == "laser":
        if laser is None:
            raise ValueError("laser must be provided when center='laser'")
        if hasattr(laser, "x"):
            x_center, y_center = float(laser.x), float(laser.y)
        else:
            x_center, y_center = laser
    else:
        raise ValueError(f"Unknown center option: {center}")

    x_center = float(np.clip(x_center, x_axis[0], x_axis[-1]))
    y_center = float(np.clip(y_center, y_axis[0], y_axis[-1]))

    points_x = np.column_stack((coords_x, np.full_like(coords_x, y_center), np.full_like(coords_x, z_top)))
    points_y = np.column_stack((np.full_like(coords_y, x_center), coords_y, np.full_like(coords_y, z_top)))
    points_z = np.column_stack((np.full_like(coords_z, x_center), np.full_like(coords_z, y_center), coords_z))

    if volume is None:
        T_x = reconstruct_temperature_volume_at_points(a, num, geom, points_x)
        T_y = reconstruct_temperature_volume_at_points(a, num, geom, points_y)
        T_z = reconstruct_temperature_volume_at_points(a, num, geom, points_z)
    else:
        interpolator = RegularGridInterpolator((z_axis, y_axis, x_axis), volume.astype(np.float64), bounds_error=False, fill_value=np.nan)
        T_x = interpolator(points_x[:, [2, 1, 0]]).astype(np.float32)
        T_y = interpolator(points_y[:, [2, 1, 0]]).astype(np.float32)
        T_z = interpolator(points_z[:, [2, 1, 0]]).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)

    for direction, coords, profile in [
        ('x_fine', coords_x, T_x),
        ('y_fine', coords_y, T_y),
        ('z_fine', coords_z, T_z)
    ]:
        fname = os.path.join(output_dir, f"{direction}_spectral_latent_heat.txt")
        np.savetxt(fname, np.vstack([coords, profile]).T, header=f'{direction}(m) T(K)', fmt='% .6e')
        print(f"Saved fine profile: {fname}")

def box_field_to_modes(field_box, geom, timers=None):
    """
    Convert a field defined in the fine box (e.g. Q_latent) to global spectral modes.
    Input field_box is (nz_box, ny_box, nx_box) [ZYX layout].
    Returns modes in (nz, ny, nx).
    """
    if not geom.fine_mesh_initialized:
        raise RuntimeError("Fine mesh not initialized. Call geom.update_fine_mesh(laser) first.")
    
    # Direct tensor contraction avoiding intermediate reshapes/transposes
    
    # 1. Contract Z_box: (nz_box, ny_box, nx_box) . (nz, nz_box) -> (ny_box, nx_box, nz)
    t0 = time.perf_counter()
    temp1 = np.tensordot(field_box, geom.Bz_fine, axes=(0, 1))
    if timers is not None: timers['lh_modes_step1'] += time.perf_counter() - t0
    
    # 2. Contract Y_box: (ny_box, nx_box, nz) . (ny, ny_box) -> (nx, nz_box, ny_box)
    t0 = time.perf_counter()
    temp2 = np.tensordot(temp1, geom.By_fine, axes=(0, 1))
    if timers is not None: timers['lh_modes_step2'] += time.perf_counter() - t0
    
    # 3. Contract X_box: (nx_box, nz, ny) . (nx, nx_box) -> (nz, ny, nx)
    t0 = time.perf_counter()
    modes = np.tensordot(temp2, geom.Bx_fine, axes=(0, 1))
    if timers is not None: timers['lh_modes_step3'] += time.perf_counter() - t0
    
    # Multiply by dV for the numerical integration
    modes *= geom.dV_fine
    
    return modes.astype(np.float32)
