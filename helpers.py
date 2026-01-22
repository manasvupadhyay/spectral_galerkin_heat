import h5py
import os
import numpy as np
from scipy.ndimage import shift as scipy_shift
from scipy.interpolate import RegularGridInterpolator
from pyfftw.interfaces.scipy_fft import dctn
import pyfftw
from numba import njit, prange

try:
    import cupy as cp
    import cupyx.scipy.fft as cupy_fft
    import cupyx.scipy.ndimage as cupy_ndimage
except ImportError:
    cp = None
    cupy_fft = None
    cupy_ndimage = None

def get_array_module(arr):
    if cp is not None:
        return cp.get_array_module(arr)
    return np

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

def C_coef(N, L, xp=np):
    C = xp.sqrt(2.0 / L) * xp.ones(N)
    C[0] = xp.sqrt(1.0 / L)
    return C


# ============================================================
#   ETD PHI FUNCTIONS
# ============================================================

def phi_functions(z):
    """ Functions used for the time stepping, taking into account exponential decay"""
    xp = get_array_module(z)
    small_threshold = 1e-6
    phi_0 = xp.exp(z)
    
    mask_small = xp.abs(z) < small_threshold
    phi_1 = xp.zeros_like(z)
    phi_1[~mask_small] = (xp.exp(z[~mask_small]) - 1.0) / z[~mask_small]
    phi_1[mask_small] = 1.0 + z[mask_small] / 2.0 + z[mask_small]**2 / 6.0
    
    phi_2 = xp.zeros_like(z)
    phi_2[~mask_small] = (xp.exp(z[~mask_small]) - 1.0 - z[~mask_small]) / (z[~mask_small]**2)
    phi_2[mask_small] = 0.5 + z[mask_small] / 6.0 + z[mask_small]**2 / 24.0
    
    return phi_0, phi_1, phi_2


# ============================================================
#   HEAT FLUX
# ============================================================

def q_laser(geom, laser):
    """Gaussian laser heat flux using current Laser position (laser.x, laser.y)."""
    xp = get_array_module(geom.X)
    r_sq = (geom.X - laser.x) ** 2 + (geom.Y - laser.y) ** 2
    return (geom.laser_coef * xp.exp(-2.0 * r_sq / laser.r_b ** 2)).astype(np.float32)


def q_evap_point(T: np.ndarray, phys) -> np.ndarray:
    """Evaporative heat flux."""
    xp = get_array_module(T)
    q = 0.82 * phys.DeltaH_LV * phys.Pa/ xp.sqrt(2 * np.pi * phys.R_v * T) * \
        xp.exp((phys.DeltaH_LV / (phys.R_v * phys.T_boil)) * (1.0 - phys.T_boil / T))
    q[T < phys.T_liquidus] = 0.0
    return q.astype(np.float32)


def shift_flux(field: np.ndarray, shift: tuple, geom) -> np.ndarray:
    """Translate a surface flux field by ``shift=(dx, dy)`` meters."""
    xp = get_array_module(field)
    if field is None or field.size == 0:
        return xp.zeros((geom.ny, geom.nx), dtype=np.float32)
    dx, dy = shift
    shift_pixels = (dy / geom.dy, dx / geom.dx)
    
    if cp is not None and xp == cp:
        return cupy_ndimage.shift(field, shift_pixels, order=1, mode='constant', cval=0.0).astype(np.float32)
    
    return scipy_shift(field, shift_pixels, order=1, mode='constant', cval=0.0).astype(np.float32) 


# ============================================================
#   DCT-II 2D 
# ============================================================

def DCT_II(q):
    xp = get_array_module(q)
    if cp is not None and xp == cp:
        # Cupy DCT does not support 'workers' argument
        return cupy_fft.dctn(q, type=2, norm='ortho').astype(np.float32, copy=False)
    return dctn(q.astype(np.float32, copy=False), type=2, norm='ortho', workers=-1).astype(np.float32, copy=False)

# ============================================================
#   LATENT HEAT SOURCE HELPERS
# ============================================================

@njit(parallel=True, fastmath=True)
def _compute_source_term_from_temperature(T_curr, T_prev, T_S, T_L, rho, L, dt, out):
    """
    Compute Q = - rho * L * (1 / (TL - TS)) * (dT/dt) * Indicator(TS <= T <= TL)
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

def compute_latent_heat_source(Q_buffer, box_coords, phys, laser, geom, num, alpha=0.4):
    """
    Compute volumetric latent heat source Q (W/m^3) using temperature derivative.
    Equation: Q = - rho * L * (1/DeltaT) * (dT/dt) * Indicator
    Applies under-relaxation: Q_applied = alpha * Q_new + (1-alpha) * Q_old
    """
    # 1. Reconstruct Temperature on Fine Mesh
    T_box, _ = reconstruct_temperature_box(num.a_temp, num, geom)
    
    # 2. Initialize/Retrieve State buffers
    if not hasattr(num, 'T_prev'):
        num.T_prev = np.zeros_like(T_box)
        num.T_prev[:] = T_box[:] # Initialize with current T
        
        # Initialize Previous Q for relaxation
        num.Q_prev = np.zeros_like(Q_buffer)
        
        num.laser_x_prev = laser.x
        num.laser_y_prev = laser.y
        Q_buffer.fill(0.0)
        return

    # 3. Shift Previous Fields to Current Frame
    # The grid has moved by (dx_shift, dy_shift)
    shift_x = laser.x - num.laser_x_prev
    shift_y = laser.y - num.laser_y_prev
    
    # Calculate shift in pixels (shift > 0 means grid moved right, so we look left into old array)
    shift_pixels = (0, -shift_y / geom.dy_fine, -shift_x / geom.dx_fine)
    
    # Shift Temperature (continuous field, use order=1)
    T_prev_aligned = scipy_shift(num.T_prev, shift_pixels, order=1, mode='nearest')
    
    # Shift Previous Q (source term, use constant fill for outside)
    if not hasattr(num, 'Q_prev'): num.Q_prev = np.zeros_like(Q_buffer)
    Q_prev_aligned = scipy_shift(num.Q_prev, shift_pixels, order=1, mode='constant', cval=0.0)
    
    # 4. Compute Source Term (New)
    _compute_source_term_from_temperature(T_box, T_prev_aligned, 
                                          phys.T_solidus, phys.T_liquidus, 
                                          phys.rho, phys.L_f, num.dt, 
                                          Q_buffer)
    
    # 5. Apply Relaxation
    # Q_applied = alpha * Q_new + (1 - alpha) * Q_old
    if alpha < 1.0:
        Q_buffer[:] = alpha * Q_buffer + (1.0 - alpha) * Q_prev_aligned
    
    # 6. Update History
    num.T_prev[:] = T_box[:]
    num.Q_prev[:] = Q_buffer[:] # Store the applied Q
    num.laser_x_prev = laser.x
    num.laser_y_prev = laser.y


# ============================================================
#   RECONSTRUCT TEMPERATURE FIELDS HELPER FUNCTIONS
# ============================================================

def reconstruct_temperature_box(a, num, geom, symmetrize=False):
    """
    Reconstructs temperature in a small ROI around the laser using 
    precomputed fine mesh cosine bases from geom.
    """
    if not geom.fine_mesh_initialized:
        raise RuntimeError("Fine mesh not initialized. Call geom.update_fine_mesh(laser) first.")
    
    xp = get_array_module(a)
    
    # Tensor Contraction using precomputed bases
    # T(x,y,z) = sum_p sum_n sum_m  a[p,n,m] * Bz[p,z] * By[n,y] * Bx[m,x]
    
    # 1. Contract Z: (nz, ny, nx) . (nz, nz_box) -> (ny, nx, nz_box)
    T_step1 = xp.tensordot(a, geom.Bz_fine, axes=(0, 0))
    
    # 2. Contract Y: (ny, nx, nz_box) . (ny, ny_box) -> (nx, nz_box, ny_box)
    T_step2 = xp.tensordot(T_step1, geom.By_fine, axes=(0, 0))
    
    # 3. Contract X: (nx, nz_box, ny_box) . (nx, nx_box) -> (nz_box, ny_box, nx_box)
    T_box = xp.tensordot(T_step2, geom.Bx_fine, axes=(0, 0))
    
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
    xp = get_array_module(a)
    # To be implemented later, proper DCT-based reconstruction for full volume
    Bx = (geom.Cm[:, None] * geom.cos_mx).astype(np.float32)
    By = (geom.Cn[:, None] * geom.cos_ny).astype(np.float32)
    Bz = (geom.Cp[:, None] * geom.cos_pz).astype(np.float32)

    T_step1 = xp.tensordot(a, Bx, axes=(2, 0))  # (nz, ny, nx)
    T_step2 = xp.tensordot(T_step1, By, axes=(1, 0))  # (nz, nx, ny)
    T_full = xp.tensordot(T_step2, Bz, axes=(0, 0))  # (nx, ny, nz)

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

def box_field_to_modes(field_box, geom):
    """
    Convert a field defined in the fine box (e.g. Q_latent) to global spectral modes.
    Input field_box is (nz_box, ny_box, nx_box) [ZYX layout].
    Returns modes in (nz, ny, nx).
    """
    if not geom.fine_mesh_initialized:
        raise RuntimeError("Fine mesh not initialized. Call geom.update_fine_mesh(laser) first.")
    
    xp = get_array_module(field_box)
    
    # Direct tensor contraction avoiding intermediate reshapes/transposes
    
    # 1. Contract Z_box: (nz_box, ny_box, nx_box) . (nz, nz_box) -> (ny_box, nx_box, nz)
    temp1 = xp.tensordot(field_box, geom.Bz_fine, axes=(0, 1))
    
    # 2. Contract Y_box: (ny_box, nx_box, nz) . (ny, ny_box) -> (nx, nz_box, ny_box)
    temp2 = xp.tensordot(temp1, geom.By_fine, axes=(0, 1))
    
    # 3. Contract X_box: (nx_box, nz, ny) . (nx, nx_box) -> (nz, ny, nx)
    modes = xp.tensordot(temp2, geom.Bx_fine, axes=(0, 1))
    
    # Multiply by dV for the numerical integration
    modes *= geom.dV_fine
    
    return modes.astype(np.float32)

# ============================================================
#   KERNEL ABSTRACTIONS
#   (These handle CPU/GPU dispatch for loop-heavy operations)
# ============================================================

@njit(parallel=True, fastmath=True)
def _compute_a_temp_numba(aK, KK_by_Cp, B_scaled, a_temp_out):
    nz = aK.shape[0]
    for p in prange(nz):
        a_temp_out[p, :, :] = aK[p, :, :] + KK_by_Cp[p, :, :] * B_scaled

@njit(parallel=True, fastmath=True)
def _add_source_term_numba(a_temp, KK, Q_modes):
    nz = a_temp.shape[0]
    for p in prange(nz):
        for i in range(a_temp.shape[1]):
            for j in range(a_temp.shape[2]):
                a_temp[p, i, j] += KK[p, i, j] * Q_modes[p, i, j]

def compute_a_temp(aK, KK_by_Cp, B_scaled, a_temp_out):
    """
    Update spectral coefficients for ETD1 scheme.
    a_temp_out = aK + KK_by_Cp * B_scaled (with appropriate broadcasting)
    """
    xp = get_array_module(aK)
    if xp is np:
        _compute_a_temp_numba(aK, KK_by_Cp, B_scaled, a_temp_out)
    else:
        # GPU / CuPy: Generic broadcasting logic
        # B_scaled is 2D (ny, nx), needs to broadcast over Z axis (dim 0)
        # aK and KK_by_Cp are 3D (nz, ny, nx)
        xp.add(aK, KK_by_Cp * B_scaled[None, :, :], out=a_temp_out)

def add_source_term(a_temp, KK, Q_modes):
    """
    Add volumetric source term to temperature modes.
    a_temp += KK * Q_modes
    """
    xp = get_array_module(a_temp)
    if xp is np:
        _add_source_term_numba(a_temp, KK, Q_modes)
    else:
        # Element-wise addition for GPU
        xp.add(a_temp, KK * Q_modes, out=a_temp)
