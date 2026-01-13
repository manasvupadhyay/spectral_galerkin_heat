import h5py
import os
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
#   LATENT HEAT SOURCE RASTERIZATION
# ============================================================

@njit(parallel=True, fastmath=True)
def rasterize_latent_heat(Q_box, isotherm_entries, 
                          nx_box, ny_box, nz_box, 
                          dx, dy, dz, 
                          rho, L_f, dS, v, xi_box_offset):
    """
    Rasterize latent heat source onto the fine grid.
    
    Args:
        Q_box: output array (nx, ny, nz) - volumetric heat source W/m^3
        isotherm_entries: array of shape (n_entries, 8)
        nx_box, ny_box, nz_box: dimensions
        dx, dy, dz: cell sizes
        rho, L_f: material properties
        dS: cross-section area of the tube (dy*dz)
        v: velocity magnitude
        xi_box_offset: offset to convert relative coordinate xi to box index
    """
    # Box start position in relative coordinates (approx)
    # box_xi goes from -Lx_box/2 to +Lx_box/2
    # ix=0 corresponds to xi = -nx_box*dx/2
    # xi_box_offset passed as argument now
    
    # Volumetric source magnitude base: Power / Volume
    # Power = rho * v * dS * L_f * f_scale
    # Volume = dx * dy * dz
    # Q_vol = (rho * v * dS * L_f * f_scale) / (dx * dy * dz)
    # Since dS = dy * dz (usually), Q_vol = rho * v * L_f * f_scale / dx
    
    # Precompute constant part
    # We use dS explicitly in case the tube area is different from voxel face
    Q_pre = (rho * v * dS * L_f) / (dx * dy * dz)
    
    n_entries = isotherm_entries.shape[0]
    
    # We iterate over entries and fill the grid
    # Since multiple entries might map to the same voxel (unlikely if 1-to-1 map), 
    # or we just want to parallelize over entries.
    # Parallelizing over entries is safer if we write to different y,z lines.
    # Isotherm entries are distinct in (y,z).
    
    for ie in prange(n_entries):
        iy_src = int(isotherm_entries[ie, 0])
        iz_src = int(isotherm_entries[ie, 1])
        
        # Check bounds
        if iy_src < 0 or iy_src >= ny_box or iz_src < 0 or iz_src >= nz_box:
            continue
            
        xi_S_back = isotherm_entries[ie, 2]
        xi_L_back = isotherm_entries[ie, 3]
        xi_L_front = isotherm_entries[ie, 4]
        xi_S_front = isotherm_entries[ie, 5]
        f_scale = isotherm_entries[ie, 6]
        
        Q_mag = Q_pre * f_scale
        
        # --- 1. Back Zone (Solidification -> Heat Source +) ---
        # From xi_S_back to xi_L_back
        # Convert to indices
        ix_start_raw = int(round((xi_S_back + xi_box_offset) / dx))
        ix_end_raw = int(round((xi_L_back + xi_box_offset) / dx))
        
        # Ensure start < end
        if ix_start_raw > ix_end_raw: ix_start_raw, ix_end_raw = ix_end_raw, ix_start_raw
        
        n_pixels = ix_end_raw - ix_start_raw
        if n_pixels == 0:
            n_pixels = 1
            ix_end_raw += 1
            
        Q_applied = Q_mag / n_pixels
        
        # Clamp to box
        ix_start = max(0, ix_start_raw)
        ix_end = min(nx_box, ix_end_raw)
        
        for ix in range(ix_start, ix_end):
            Q_box[ix, iy_src, iz_src] += Q_applied

        # --- 2. Front Zone (Melting -> Heat Sink -) ---
        # From xi_L_front to xi_S_front
        ix_start_raw = int(round((xi_L_front + xi_box_offset) / dx))
        ix_end_raw = int(round((xi_S_front + xi_box_offset) / dx))
        
        if ix_start_raw > ix_end_raw: ix_start_raw, ix_end_raw = ix_end_raw, ix_start_raw
        
        n_pixels = ix_end_raw - ix_start_raw
        if n_pixels == 0:
            n_pixels = 1
            ix_end_raw += 1
            
        Q_applied = Q_mag / n_pixels
        
        ix_start = max(0, ix_start_raw)
        ix_end = min(nx_box, ix_end_raw)
        
        for ix in range(ix_start, ix_end):
            Q_box[ix, iy_src, iz_src] -= Q_applied

def compute_latent_heat_source(Q_box, box_coords, isotherm_data, phys, laser, geom, num, verbose=False):
    """
    Compute volumetric latent heat source Q (W/m^3).
    
    Args:
        Q_box: output array (nx, ny, nz) to fill
        box_coords: (box_x, box_y, box_z) tuple
        isotherm_data: list of tuples
        phys: PhysParams object
        laser: Laser object
        geom: GeomParams object
        dt: Time step (s). If non-zero, projects laser position to t + dt.
            Use dt=num.dt to apply heat at the end of the step (implicit-like).
    """
    box_x, box_y, box_z = box_coords
    
    # Project laser position: x_effective = x(t) + v_x * dt
    # If dt=0, uses laser.x (start of step). If dt=step_size, uses end of step.
    x_laser = laser.x + laser.v[0] * num.dt
    
    if len(isotherm_data) == 0:
        Q_box.fill(0.0)
        return
    
    # Convert isotherm data to numpy array with PHYSICAL coordinates in moving frame
    n_entries = len(isotherm_data)
    isotherm_entries = np.zeros((n_entries, 8), dtype=np.float64)
    
    for i, entry in enumerate(isotherm_data):
        iy, iz, i_S_back, i_L_back, i_L_front, i_S_front, f_scale, is_partial = entry
        
        # Convert indices to physical x coordinates, then to moving frame
        xi_S_back = box_x[i_S_back] - x_laser
        xi_L_back = box_x[i_L_back] - x_laser
        xi_L_front = box_x[i_L_front] - x_laser
        xi_S_front = box_x[i_S_front] - x_laser
        
        isotherm_entries[i, :] = [iy, iz, xi_S_back, xi_L_back, xi_L_front, xi_S_front, f_scale, float(is_partial)]
    
    # Physical parameters
    v = np.sqrt(laser.v[0]**2 + laser.v[1]**2)
    rho = phys.rho
    L_f = phys.L_f
    
    # Grid spacing
    dx = box_x[1] - box_x[0] if len(box_x) > 1 else geom.dx
    dy = box_y[1] - box_y[0] if len(box_y) > 1 else geom.dy
    dz = box_z[1] - box_z[0] if len(box_z) > 1 else geom.dz
    dS = dy * dz
    
    nx_box, ny_box, nz_box = Q_box.shape
    
    # Offset for rasterization: maps xi (relative to laser) to box index
    # ix = (xi + xi_box_offset) / dx
    xi_box_offset = x_laser - box_x[0]

    # Rasterize
    Q_box.fill(0.0)
    rasterize_latent_heat(
        Q_box, isotherm_entries,
        nx_box, ny_box, nz_box,
        dx, dy, dz,
        rho, L_f, dS, v, xi_box_offset
    )

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

def find_isotherms_along_x(T_box, M_yz, mask_full, mask_partial, phys):
    """
    Find isotherm indices for all (y,z) points requiring latent heat correction.
    Vectorized implementation - no Python loops over mask indices.
    
    Returns list of tuples:
        (iy, iz, i_S_back, i_L_back, i_L_front, i_S_front, f_scale, is_partial)
    For partial melt: i_L_back = i_L_front = index of peak temperature along x
    """
    T_S, T_L = phys.T_solidus, phys.T_liquidus
    nx = T_box.shape[0]
    
    results = []
    
    # ===== FULL MELT POINTS (vectorized) =====
    iy_full, iz_full = np.where(mask_full)
    n_full = len(iy_full)
    
    if n_full > 0:
        # Extract all temperature lines at once: shape (n_full, nx)
        T_lines_full = T_box[:, iy_full, iz_full].T  # (n_full, nx)
        
        # Boolean masks for solidus and liquidus
        above_S = T_lines_full >= T_S  # (n_full, nx)
        above_L = T_lines_full >= T_L  # (n_full, nx)
        
        # Count how many points are above each threshold per line
        count_S = above_S.sum(axis=1)  # (n_full,)
        count_L = above_L.sum(axis=1)  # (n_full,)
        
        # Valid lines have at least 2 points above both thresholds
        valid_full = (count_S >= 2) & (count_L >= 2)
        
        if np.any(valid_full):
            # Get indices of valid lines
            valid_idx = np.where(valid_full)[0]
            iy_valid = iy_full[valid_idx]
            iz_valid = iz_full[valid_idx]
            above_S_valid = above_S[valid_idx]  # (n_valid, nx)
            above_L_valid = above_L[valid_idx]  # (n_valid, nx)
            
            # First index where True: argmax on boolean array
            i_S_back = np.argmax(above_S_valid, axis=1)  # (n_valid,)
            i_L_back = np.argmax(above_L_valid, axis=1)  # (n_valid,)
            
            # Last index where True: nx - 1 - argmax(flipped)
            i_S_front = nx - 1 - np.argmax(above_S_valid[:, ::-1], axis=1)  # (n_valid,)
            i_L_front = nx - 1 - np.argmax(above_L_valid[:, ::-1], axis=1)  # (n_valid,)
            
            # Build results for full melt
            for i in range(len(valid_idx)):
                results.append((
                    iy_valid[i], iz_valid[i],
                    i_S_back[i], i_L_back[i], i_L_front[i], i_S_front[i],
                    1.0, False
                ))
    
    # ===== PARTIAL MELT POINTS (vectorized) =====
    iy_partial, iz_partial = np.where(mask_partial)
    n_partial = len(iy_partial)
    
    if n_partial > 0:
        # Extract all temperature lines at once: shape (n_partial, nx)
        T_lines_partial = T_box[:, iy_partial, iz_partial].T  # (n_partial, nx)
        
        # Boolean mask for solidus
        above_S = T_lines_partial >= T_S  # (n_partial, nx)
        count_S = above_S.sum(axis=1)  # (n_partial,)
        
        # Valid lines have at least 2 points above solidus
        valid_partial = count_S >= 2
        
        if np.any(valid_partial):
            valid_idx = np.where(valid_partial)[0]
            iy_valid = iy_partial[valid_idx]
            iz_valid = iz_partial[valid_idx]
            T_lines_valid = T_lines_partial[valid_idx]  # (n_valid, nx)
            above_S_valid = above_S[valid_idx]  # (n_valid, nx)
            
            # First and last solidus indices
            i_S_back = np.argmax(above_S_valid, axis=1)
            i_S_front = nx - 1 - np.argmax(above_S_valid[:, ::-1], axis=1)
            
            # Peak location and temperature
            i_peak = np.argmax(T_lines_valid, axis=1)  # (n_valid,)
            T_peak = np.max(T_lines_valid, axis=1)  # (n_valid,)
            f_scale = (T_peak - T_S) / (T_L - T_S)
            
            # Build results for partial melt
            for i in range(len(valid_idx)):
                results.append((
                    iy_valid[i], iz_valid[i],
                    i_S_back[i], i_peak[i], i_peak[i], i_S_front[i],
                    f_scale[i], True
                ))
    
    return results


# ============================================================
#   DCT-II 2D 
# ============================================================

def DCT_II(q):
    return dctn(q.astype(np.float32, copy=False), type=2, norm='ortho', workers=-1).astype(np.float32, copy=False)

# ============================================================
#   RECONSTRUCT TEMPERATURE FIELD
# ============================================================

def reconstruct_temperature_top(a, num, geom):
    """Reconstruct top surface temperature from modal coefficients."""
    A = (geom.Cp32[:, None, None] * a).sum(axis=0)
    return (geom.recon_scale * dctn(A, type=3, norm='ortho', axes=(0, 1), workers=-1)).astype(np.float32, copy=False)

def reconstruct_temperature_xz(a, num, geom, phys, laser, y0=None, mode='full_field'):
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
    
    # 4. Early exit if full field requested
    if mode != 'meltpool':
        return geom.x, geom.z, T_xz.astype(np.float32)

    mask = T_xz >= phys.T_liquidus
    if not np.any(mask):
        print("Warning: No melt pool detected.")
        return geom.x, geom.z, T_xz.astype(np.float32)

    z_idx, x_idx = np.where(mask)       # Get bounding box indices directly from mask
    z0, z1 = z_idx.min(), z_idx.max()
    x0, x1 = x_idx.min(), x_idx.max()
    w_pad = int((x1 - x0) * 0.25) # Crop window adds 0.25 on each side -> 1.5x total
    ix0, ix1 = max(0, x0 - w_pad), min(nx, x1 + w_pad)
    
    d_depth = z1 - z0
    zc = (z0 + z1) // 2
    iz0, iz1 = max(0, zc - d_depth), min(nz, zc + d_depth)

    return geom.x[ix0:ix1], geom.z[iz0:iz1], T_xz[iz0:iz1, ix0:ix1].astype(np.float32)

def reconstruct_temperature_box(a, num, geom, symmetrize=False):
    """
    Reconstructs temperature in a small ROI around the laser using 
    precomputed fine mesh cosine bases from geom.
    """
    if not geom.fine_mesh_initialized:
        raise RuntimeError("Fine mesh not initialized. Call geom.update_fine_mesh(laser) first.")
    
    # Tensor Contraction using precomputed bases
    # T(x,y,z) = sum_p sum_n sum_m  a[p,n,m] * Bz[p,z] * By[n,y] * Bx[m,x]
    
    T_step1 = np.tensordot(a, geom.Bx_fine, axes=(2, 0))  # (nz, ny, nx_box)
    T_step2 = np.tensordot(T_step1, geom.By_fine, axes=(1, 0))  # (nz, nx_box, ny_box)
    T_box = np.tensordot(T_step2, geom.Bz_fine, axes=(0, 0))  # (nx_box, ny_box, nz_box)
    
    return T_box.astype(np.float32), (geom.box_x, geom.box_y, geom.box_z)


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
        # Fallback for manual coords if passed as tuple?
        if isinstance(center, (tuple, list, np.ndarray)):
            x_center, y_center = float(center[0]), float(center[1])
        else:
            raise ValueError(f"Unknown center option: {center}")

    # Determine Z level for X/Y cuts
    # Assuming laser source is at z=0 (based on box_z used in latent heat)
    z_top = 0.0 

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

def box_field_to_modes(field_box, geom):
    """
    Convert a field defined in the fine box (e.g. Q_latent) to global spectral modes.
    Uses precomputed fine mesh cosine bases from geom.
    Optimized using explicit matrix multiplications.
    """
    if not geom.fine_mesh_initialized:
        raise RuntimeError("Fine mesh not initialized. Call geom.update_fine_mesh(laser) first.")
    
    # field_box shape: (nx_box, ny_box, nz_box)
    nx_box, ny_box, nz_box = field_box.shape
    nx, _ = geom.Bx_fine.shape
    ny, _ = geom.By_fine.shape
    nz, _ = geom.Bz_fine.shape
    
    # 1. Contract X: (nx, nx_box) @ (nx_box, ny_box*nz_box) -> (nx, ny_box*nz_box)
    # Reshape field to combine Y and Z
    field_reshaped = field_box.reshape(nx_box, ny_box * nz_box)
    temp1 = geom.Bx_fine @ field_reshaped
    
    # 2. Contract Y: (ny, ny_box) @ (ny_box, nx*nz_box) -> (ny, nx*nz_box)
    # Reshape temp1 to (nx, ny_box, nz_box) and transpose to (ny_box, nx, nz_box)
    # Then flatten last two dims
    temp1 = temp1.reshape(nx, ny_box, nz_box).transpose(1, 0, 2).reshape(ny_box, nx * nz_box)
    temp2 = geom.By_fine @ temp1
    
    # 3. Contract Z: (nz, nz_box) @ (nz_box, ny*nx) -> (nz, ny*nx)
    # Reshape temp2 to (ny, nx, nz_box) and transpose to (nz_box, ny, nx)
    # Then flatten last two dims
    temp2 = temp2.reshape(ny, nx, nz_box).transpose(2, 0, 1).reshape(nz_box, ny * nx)
    modes = geom.Bz_fine @ temp2
    
    # Final reshape to (nz, ny, nx)
    modes = modes.reshape(nz, ny, nx)
    
    # Multiply by dV for the numerical integration
    modes *= geom.dV_fine
    
    return modes.astype(np.float32)

def cell_to_node_reconstruction(T_cells):
    """
    Converts cell-centered data (nz, ny, nx) to node-centered data (nz+1, ny+1, nx+1)
    by performing a 3D moving average.
    """
    # 1. Pad boundary (Neumann-like)
    T_pad = np.pad(T_cells, pad_width=1, mode='edge')

    # 2. Average successively along Z, Y, X to get to corners
    T_z = 0.5 * (T_pad[:-1, :, :] + T_pad[1:, :, :])      # Avg Z
    T_zy = 0.5 * (T_z[:, :-1, :] + T_z[:, 1:, :])         # Avg Y
    T_nodes = 0.5 * (T_zy[:, :, :-1] + T_zy[:, :, 1:])    # Avg X
    
    return T_nodes
