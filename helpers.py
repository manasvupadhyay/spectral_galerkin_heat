import h5py
import os
import numpy as np
from scipy.ndimage import shift as scipy_shift
from pyfftw.interfaces.scipy_fft import dctn
import pyfftw

OUT_DIR = "out"

def save_field_to_hdf5(filename_base, field, grid_coords, value_name="Field", geom=None, verbose=False):
        """Serialize a 3D scalar field to HDF5 with an accompanying XDMF wrapper."""
        h5_name = f"{filename_base}.h5"
        xmf_name = f"{filename_base}.xmf"

        # Use basename for the reference inside XDMF to avoid double directory paths
        # when ParaView resolves relative paths.
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
#   LINE SOURCE GREEN'S FUNCTION FOR LATENT HEAT CORRECTION
# ============================================================

import numpy as np
from numba import njit, prange
from scipy.integrate import quad

@njit(fastmath=True)
def rosenthal_point_source(xi, y, z, z_s, v, D, lmbda):
    """
    Rosenthal point source Green's function with image source for Neumann BC at z=0.
    
    dT = (q / 4πλ) * [exp(-v(ξ+R1)/2D)/R1 + exp(-v(ξ+R2)/2D)/R2]
    
    Args:
        xi: x - vt (position in moving frame)
        y: y coordinate relative to source
        z: z coordinate (depth, positive downward)
        z_s: source depth (positive)
        v: laser velocity magnitude
        D: thermal diffusivity (k / (rho * Cp))
        lmbda: thermal conductivity
    
    Returns:
        Green's function value (without q/4πλ prefactor)
    """
    # Distance to real source at (0, 0, z_s)
    R1_sq = xi**2 + y**2 + (z - z_s)**2
    R1 = np.sqrt(R1_sq) if R1_sq > 1e-20 else 1e-10
    
    # Distance to image source at (0, 0, -z_s)
    R2_sq = xi**2 + y**2 + (z + z_s)**2
    R2 = np.sqrt(R2_sq) if R2_sq > 1e-20 else 1e-10
    
    # Exponential decay terms
    v_over_2D = v / (2.0 * D)
    
    term1 = np.exp(-v_over_2D * (xi + R1)) / R1
    term2 = np.exp(-v_over_2D * (xi + R2)) / R2
    # Warning temporary removal of image source contribution
    # Is is probably implicitely accounted in the modal basis 
    return term1 #+ term2


@njit(fastmath=True)
def line_source_integrand(u, xi, y, z, z_s, v, D, R_min):
    """
    Integrand for the line source integral.
    
    The line source extends along x (the ξ direction in moving frame).
    Integration variable u is the position along the source line.
    
    Args:
        u: integration variable (source position along x)
        xi, y, z: field point coordinates
        z_s: source depth
        v, D: velocity and diffusivity
        R_min: regularization radius
    
    Returns:
        Integrand value
    """
    xi_rel = xi - u  # Relative position from source point at u
    
    # Distance to real source
    R1_sq = xi_rel**2 + y**2 + (z - z_s)**2
    R1 = max(np.sqrt(R1_sq), R_min)
    
    # Distance to image source  
    R2_sq = xi_rel**2 + y**2 + (z + z_s)**2
    R2 = max(np.sqrt(R2_sq), R_min)
    
    v_over_2D = v / (2.0 * D)
    
    term1 = np.exp(-v_over_2D * (xi_rel + R1)) / R1
    term2 = np.exp(-v_over_2D * (xi_rel + R2)) / R2
    
    # removal of image source (term2) as it is implicitly 
    # handled by the spectral cosine basis (even extension).
    return term1 # + term2


def integrate_line_source_scipy(xi, y, z, z_s, xi_start, xi_end, v, D, lmbda, q_line, R_min=1e-6):
    """
    Integrate the line source Green's function from xi_start to xi_end.
    Uses scipy.integrate.quad for accuracy.
    
    T_corr = (q_line / 4πλ) * ∫[xi_start to xi_end] G(xi-u, y, z, z_s) du
    
    Args:
        xi, y, z: field point coordinates
        z_s: source depth
        xi_start, xi_end: integration bounds (isotherm positions)
        v: laser velocity
        D: thermal diffusivity
        lmbda: thermal conductivity
        q_line: linear power density (W/m)
        R_min: regularization radius
    
    Returns:
        Temperature correction at (xi, y, z)
    """
    if abs(xi_end - xi_start) < 1e-12:
        return 0.0
    
    prefactor = q_line / (4.0 * np.pi * lmbda)
    
    def integrand(u):
        return line_source_integrand(u, xi, y, z, z_s, v, D, R_min)
    
    result, _ = quad(integrand, xi_start, xi_end, limit=100)
    
    return prefactor * result


@njit(fastmath=True)
def integrate_line_source_trapz(xi, y, z, z_s, xi_start, xi_end, v, D, lmbda, q_line, n_points=10, R_min=1e-6):
    """
    Integrate the line source using trapezoidal rule (Numba compatible).
    
    Args:
        xi, y, z: field point
        z_s: source depth
        xi_start, xi_end: integration bounds
        v, D, lmbda: physical parameters
        q_line: linear power density
        n_points: number of integration points
        R_min: regularization radius
    
    Returns:
        Temperature correction
    """
    if abs(xi_end - xi_start) < 1e-12:
        return 0.0
    
    prefactor = q_line / (4.0 * np.pi * lmbda)
    
    du = (xi_end - xi_start) / (n_points - 1)
    integral = 0.0
    
    for i in range(n_points):
        u = xi_start + i * du
        val = line_source_integrand(u, xi, y, z, z_s, v, D, R_min)
        
        if i == 0 or i == n_points - 1:
            integral += 0.5 * val
        else:
            integral += val
    
    integral *= du
    
    return prefactor * integral


@njit(parallel=True, fastmath=True)
def compute_T_corr_for_box(T_corr, T_corr_back, T_corr_front, box_xi, box_y_rel, box_z, 
                            isotherm_entries, 
                            v, D, lmbda, rho, L_f, dS):
    """
    Compute temperature correction for entire box from all line sources.
    
    All coordinates are in the MOVING FRAME (relative to laser position).
    
    Args:
        T_corr: output array (nx, ny, nz) - total correction (modified in place)
        T_corr_back: output array (nx, ny, nz) - back mushy zone contribution (solidification)
        T_corr_front: output array (nx, ny, nz) - front mushy zone contribution (melting)
        box_xi: 1D array of xi coordinates (x - x_laser) in moving frame
        box_y_rel: 1D array of y coordinates relative to laser
        box_z: 1D array of z coordinates (depth)
        isotherm_entries: array of shape (n_entries, 8) containing:
            [iy, iz, xi_S_back, xi_L_back, xi_L_front, xi_S_front, f_scale, is_partial]
            where xi values are in MOVING FRAME coordinates
        v: laser velocity magnitude
        D: thermal diffusivity
        lmbda: thermal conductivity
        rho: density
        L_f: latent heat of fusion
        dS: cross-section area for each tube (dy * dz)
    """
    nx = len(box_xi)
    ny = len(box_y_rel)
    nz = len(box_z)
    n_entries = isotherm_entries.shape[0]
    
    # Calculate R_min based on tube cross-section
    R_min = np.sqrt(dS / np.pi)
    
    # Loop over all field points in parallel
    for ix in prange(nx):
        xi = box_xi[ix]  # Field point xi in moving frame
        for iy_field in range(ny):
            y_field = box_y_rel[iy_field]  # Field point y relative to laser
            for iz_field in range(nz):
                z_field = box_z[iz_field]
                
                T_sum = 0.0
                
                # Accumulate contributions from all line sources
                for ie in range(n_entries):
                    iy_src = int(isotherm_entries[ie, 0])
                    iz_src = int(isotherm_entries[ie, 1])
                    xi_S_back = isotherm_entries[ie, 2]
                    xi_L_back = isotherm_entries[ie, 3]
                    xi_L_front = isotherm_entries[ie, 4]
                    xi_S_front = isotherm_entries[ie, 5]
                    f_scale = isotherm_entries[ie, 6]
                    
                    # Source location in y-z plane (relative coordinates)
                    y_s = box_y_rel[iy_src]  # y position of source relative to laser
                    z_s = box_z[iz_src]      # z position (depth) of source
                    
                    # Relative y position between field point and source
                    y_rel = y_field - y_s
                    
                    # Mushy zone lengths (in moving frame)
                    l_m_back = xi_L_back - xi_S_back    # Back mushy zone (negative xi, behind laser)
                    l_m_front = xi_S_front - xi_L_front  # Front mushy zone (positive xi, ahead of laser)
                    
                    # Back mushy zone (solidification - heat SOURCE, positive)
                    # This is BEHIND the laser where material is solidifying
                    if abs(l_m_back) > 1e-10:
                        q_line_back = rho * v * dS * L_f * f_scale / abs(l_m_back)
                        T_back = integrate_line_source_trapz(
                            xi, y_rel, z_field, z_s,
                            xi_S_back, xi_L_back,
                            v, D, lmbda, q_line_back, 20, R_min
                        )
                        T_sum += T_back
                        T_corr_back[ix, iy_field, iz_field] += T_back
                    
                    # Front mushy zone (melting - heat SINK, negative)
                    # This is AHEAD of the laser where material is melting
                    if abs(l_m_front) > 1e-10:
                        q_line_front = -rho * v * dS * L_f * f_scale / abs(l_m_front)
                        T_front = integrate_line_source_trapz(
                            xi, y_rel, z_field, z_s,
                            xi_L_front, xi_S_front,
                            v, D, lmbda, q_line_front, 20, R_min
                        )
                        T_sum += T_front
                        T_corr_front[ix, iy_field, iz_field] += T_front
                
                T_corr[ix, iy_field, iz_field] += T_sum


def compute_latent_heat_correction(T_corr, box_coords, isotherm_data, phys, laser, geom, verbose=False):
    """
    High-level function to compute latent heat correction.
    
    Args:
        T_corr: output array (nx, ny, nz) to fill
        box_coords: (box_x, box_y, box_z) tuple
        isotherm_data: list of tuples (iy, iz, i_S_back, i_L_back, i_L_front, i_S_front, f_scale, is_partial)
        phys: PhysParams object
        laser: Laser object
        geom: GeomParams object
        verbose: if True, print detailed statistics (default: False)
    
    Returns:
        dict with T_corr_back, T_corr_front arrays and energy statistics
    """
    box_x, box_y, box_z = box_coords
    
    # Convert to moving frame: ξ = x - x_laser
    x_laser = laser.x
    y_laser = laser.y
    
    # Relative coordinates (moving frame centered on laser)
    box_xi = box_x - x_laser
    box_y_rel = box_y - y_laser
    
    if len(isotherm_data) == 0:
        if verbose:
            print("  [WARN] No isotherm data, returning zero correction!")
        return {'T_corr_back': np.zeros_like(T_corr), 'T_corr_front': np.zeros_like(T_corr)}
    
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
    v = np.sqrt(laser.v[0]**2 + laser.v[1]**2)  # velocity magnitude
    D = phys.k / (phys.rho * phys.Cp)  # thermal diffusivity
    lmbda = phys.k
    rho = phys.rho
    L_f = phys.L_f
    
    # Cross-section area (tube area = dy * dz)
    dy_box = box_y[1] - box_y[0] if len(box_y) > 1 else geom.dy
    dz_box = box_z[1] - box_z[0] if len(box_z) > 1 else geom.dz
    dS = dy_box * dz_box
    
    # Create arrays to track back and front contributions separately
    T_corr_back = np.zeros_like(T_corr)
    T_corr_front = np.zeros_like(T_corr)
    
    # Compute correction using MOVING FRAME coordinates
    # compute_T_corr_for_box(
    #     T_corr, 
    #     T_corr_back,
    #     T_corr_front,
    #     box_xi.astype(np.float64),
    #     box_y_rel.astype(np.float64),
    #     box_z.astype(np.float64),
    #     isotherm_entries,
    #     v, D, lmbda, rho, L_f, dS
    # )
    
    # Use Fast Kernel Method
    if not hasattr(geom, 'kernel'):
        geom.kernel = precompute_rosenthal_kernel(phys, geom, laser)
        
    apply_latent_heat_correction_fast(
        T_corr, T_corr_back, T_corr_front,
        geom.kernel, isotherm_entries,
        geom.nx_box, geom.ny_box, geom.nz_box,
        geom.dx_fine, geom.dy_fine, geom.dz_fine,
        rho, L_f, dS, v
    )
    
    # Compute energy statistics (integrate T * rho * Cp over volume)
    # Energy = ∫ T * rho * Cp dV ≈ sum(T) * dx * dy * dz * rho * Cp
    dx_box = box_x[1] - box_x[0] if len(box_x) > 1 else geom.dx
    dV = dx_box * dy_box * dz_box
    
    # Energy in Joules (per unit time? Actually this is temperature field)
    # More meaningful: total power = rho * Cp * dT/dt * V, but we have steady-state
    # Let's report "heat content" change = rho * Cp * sum(T_corr) * dV
    Cp = phys.Cp
    E_back = rho * Cp * np.sum(T_corr_back) * dV   # J (heating from solidification)
    E_front = rho * Cp * np.sum(T_corr_front) * dV  # J (cooling from melting)
    E_total = rho * Cp * np.sum(T_corr) * dV
    
    # Also compute the theoretical power release rate
    # P = rho * v * A_melt * L_f where A_melt is total cross-sectional area of melt pool
    # Each tube has area dS, and we have n_entries tubes
    P_latent = rho * v * n_entries * dS * L_f  # W (theoretical latent heat release rate)
    
    if verbose:
        print(f"  [Latent Heat] {n_entries} source tubes")
        print(f"    Back (solidification):  T_corr in [{T_corr_back.min():+.1f}, {T_corr_back.max():+.1f}] K, E_back = {E_back:+.3e} J")
        print(f"    Front (melting):        T_corr in [{T_corr_front.min():+.1f}, {T_corr_front.max():+.1f}] K, E_front = {E_front:+.3e} J")
        print(f"    Net correction:         T_corr in [{T_corr.min():+.1f}, {T_corr.max():+.1f}] K, E_net = {E_total:+.3e} J")
        print(f"    Theoretical P_latent = {P_latent:.1f} W")
    
    return {'T_corr_back': T_corr_back, 'T_corr_front': T_corr_front, 'E_back': E_back, 'E_front': E_front}

import numpy as np
from scipy.ndimage import shift as scipy_shift
from pyfftw.interfaces.scipy_fft import dctn
import pyfftw
import os

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
#   FAST LATENT HEAT CORRECTION (PRECOMPUTED KERNEL)
# ============================================================

def precompute_rosenthal_kernel(phys, geom, laser):
    """
    Precompute the temperature field of a finite line source of length dx_fine
    centered at (0,0,z_s) for all possible relative positions in the fine mesh box.
    
    Returns:
        kernel: 4D array (nz_box, 2*nx_box+1, 2*ny_box+1, nz_box)
                Indices: [iz_source, ix_rel, iy_rel, iz_field]
                Values: Temperature contribution per unit linear power density (K / (W/m))
    """
    print("Precomputing Rosenthal Green's Function Kernel...")
    
    nx_box, ny_box, nz_box = geom.nx_box, geom.ny_box, geom.nz_box
    dx, dy, dz = geom.dx_fine, geom.dy_fine, geom.dz_fine
    
    # Relative coordinate ranges
    # ix_rel goes from -nx_box to +nx_box (size 2*nx_box + 1)
    # iy_rel goes from -ny_box to +ny_box (size 2*ny_box + 1)
    
    xi_range = np.arange(-nx_box, nx_box + 1) * dx
    y_range = np.arange(-ny_box, ny_box + 1) * dy
    z_range = geom.box_z # Absolute z coordinates
    
    # Physical parameters
    v = np.linalg.norm(laser.v)
    D = phys.k / (phys.rho * phys.Cp)
    lmbda = phys.k
    
    # Source segment: length dx, centered at 0
    xi_start = -dx / 2.0
    xi_end = dx / 2.0
    q_unit = 1.0 # Unit power density W/m
    
    # Calculate R_min based on tube cross-section
    R_min = np.sqrt(dy * dz / np.pi)
    
    # Allocate kernel
    # Shape: (nz_source, nx_rel, ny_rel, nz_field)
    # We use a list of 3D arrays to avoid one giant 4D block if memory is tight, 
    # but here it's small enough.
    kernel = np.zeros((nz_box, len(xi_range), len(y_range), nz_box), dtype=np.float32)
    
    # Compute kernel
    # We can parallelize this loop or use the vectorized integration function if adapted.
    # For simplicity and since it's precomputation, we loop over source depths.
    
    for iz_s in range(nz_box):
        z_s = z_range[iz_s]
        
        # Create meshgrid for field points
        # We compute for all relative x, y and all absolute z
        XI, Y, Z = np.meshgrid(xi_range, y_range, z_range, indexing='ij')
        
        # Flatten for parallel computation
        XI_flat = XI.ravel()
        Y_flat = Y.ravel()
        Z_flat = Z.ravel()
        T_flat = np.zeros_like(XI_flat)
        
        # Use the Numba integrator
        # We use a higher number of points for the kernel to be accurate
        _compute_kernel_layer(T_flat, XI_flat, Y_flat, Z_flat, z_s, xi_start, xi_end, v, D, lmbda, q_unit, R_min)
        
        kernel[iz_s] = T_flat.reshape(XI.shape).astype(np.float32)
        
    return kernel

@njit(parallel=True, fastmath=True)
def _compute_kernel_layer(T_out, xi, y, z, z_s, xi_start, xi_end, v, D, lmbda, q_line, R_min):
    n = len(xi)
    for i in prange(n):
        T_out[i] = integrate_line_source_trapz(xi[i], y[i], z[i], z_s, xi_start, xi_end, v, D, lmbda, q_line, 100, R_min)

@njit(parallel=True, fastmath=True)
def apply_latent_heat_correction_fast(T_corr, T_corr_back, T_corr_front, 
                                      kernel, isotherm_entries, 
                                      nx_box, ny_box, nz_box, 
                                      dx, dy, dz, 
                                      rho, L_f, dS, v):
    """
    Apply latent heat correction using precomputed kernel (Gather approach).
    Parallelized over field points.
    """
    # Kernel center offsets
    kc_x = nx_box
    kc_y = ny_box
    
    # Box start position in relative coordinates (approx)
    # box_xi goes from -Lx_box/2 to +Lx_box/2
    # We assume the grid is centered on the laser
    # ix=0 corresponds to xi = -nx_box*dx/2
    xi_box_offset = (nx_box * dx) / 2.0
    
    # Parallel loop over field points (x, y, z)
    for ix in prange(nx_box):
        for iy in range(ny_box):
            for iz in range(nz_box):
                
                val_back = 0.0
                val_front = 0.0
                
                # Loop over all source tubes
                for ie in range(isotherm_entries.shape[0]):
                    iy_src = int(isotherm_entries[ie, 0])
                    iz_src = int(isotherm_entries[ie, 1])
                    
                    # Relative y index for kernel lookup
                    # ky = (iy_field - iy_source) + center_offset
                    ky = (iy - iy_src) + kc_y
                    
                    # Quick bounds check for y
                    if ky < 0 or ky >= 2*ny_box + 1: continue
                    
                    # Extract source parameters
                    xi_S_back = isotherm_entries[ie, 2]
                    xi_L_back = isotherm_entries[ie, 3]
                    xi_L_front = isotherm_entries[ie, 4]
                    xi_S_front = isotherm_entries[ie, 5]
                    f_scale = isotherm_entries[ie, 6]
                    
                    # --- 1. Back Zone (Solidification) ---
                    L_back = abs(xi_L_back - xi_S_back)
                    if L_back > 1e-10:
                        n_segs = int(round(L_back / dx))
                        if n_segs < 1: n_segs = 1
                        
                        # Effective source strength per segment
                        q_eff = (rho * v * dS * L_f * f_scale) / (n_segs * dx)
                        
                        # Center of the mushy zone
                        xi_center = (xi_S_back + xi_L_back) / 2.0
                        
                        # Convert center to grid index
                        # ix_s = (xi_center - (-offset)) / dx = (xi_center + offset) / dx
                        ix_s_center = (xi_center + xi_box_offset) / dx
                        
                        # Loop over segments
                        for s in range(n_segs):
                            # Segment offset from center
                            s_off = s - (n_segs - 1) / 2.0
                            
                            # Segment grid index
                            ix_s = int(round(ix_s_center + s_off))
                            
                            # Kernel x index
                            # kx = (ix_field - ix_source) + center_offset
                            kx = (ix - ix_s) + kc_x
                            
                            if kx >= 0 and kx < 2*nx_box + 1:
                                val_back += q_eff * kernel[iz_src, kx, ky, iz]

                    # --- 2. Front Zone (Melting) ---
                    L_front = abs(xi_S_front - xi_L_front)
                    if L_front > 1e-10:
                        n_segs = int(round(L_front / dx))
                        if n_segs < 1: n_segs = 1
                        
                        q_eff = -(rho * v * dS * L_f * f_scale) / (n_segs * dx)
                        
                        xi_center = (xi_L_front + xi_S_front) / 2.0
                        ix_s_center = (xi_center + xi_box_offset) / dx
                        
                        for s in range(n_segs):
                            s_off = s - (n_segs - 1) / 2.0
                            ix_s = int(round(ix_s_center + s_off))
                            kx = (ix - ix_s) + kc_x
                            
                            if kx >= 0 and kx < 2*nx_box + 1:
                                val_front += q_eff * kernel[iz_src, kx, ky, iz]
                
                # Write results
                T_corr_back[ix, iy, iz] = val_back
                T_corr_front[ix, iy, iz] = val_front
                T_corr[ix, iy, iz] = val_back + val_front

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
    
    Args:
        a: spectral coefficients (nz, ny, nx)
        num: NumericalParams
        geom: GeomParams (must have fine_mesh_initialized=True)
        symmetrize: if True, use symmetric extension at boundaries (Neumann BC behavior)
                    if False (default), set temperature to zero outside domain
    
    Returns:
        T_box: temperature field on fine mesh, shape (nx_box, ny_box, nz_box)
        box_coords: (box_x, box_y, box_z) coordinate arrays
    """
    if not geom.fine_mesh_initialized:
        raise RuntimeError("Fine mesh not initialized. Call geom.update_fine_mesh(laser) first.")
    
    # Tensor Contraction using precomputed bases
    # T(x,y,z) = sum_p sum_n sum_m  a[p,n,m] * Bz[p,z] * By[n,y] * Bx[m,x]
    # a shape: (nz, ny, nx) = (p, n, m)
    # Result shape: (nx_box, ny_box, nz_box)
    
    T_step1 = np.tensordot(a, geom.Bx_fine, axes=(2, 0))  # (nz, ny, nx_box)
    T_step2 = np.tensordot(T_step1, geom.By_fine, axes=(1, 0))  # (nz, nx_box, ny_box)
    T_box = np.tensordot(T_step2, geom.Bz_fine, axes=(0, 0))  # (nx_box, ny_box, nz_box)
    
    return T_box.astype(np.float32), (geom.box_x, geom.box_y, geom.box_z)

def save_temp_profiles(a, num, geom, phys, laser, t=None):
    """Save 1D temperature profiles."""
    t = num.t_final if t is None else t
    
    T_top = reconstruct_temperature_top(a, num, geom)
    x_vals, y_vals = geom.X[0, :], geom.Y[:, 0]
    
      # Find location of maximum temperature
    iy_max, ix_max = np.unravel_index(np.argmax(T_top), T_top.shape)
    x_center = x_vals[ix_max]
    y_center = y_vals[iy_max]
    # Check work here
    # Profiles through the hotspot
    ix = ix_max
    iy = iy_max

    # For XZ slice, we take the slice at y = y_center (max temp)
    x_xz, z_vals, T_xz = reconstruct_temperature_xz(a, num, geom, phys, laser, y0=y_center, mode='full_field')
    ix_xz = int(np.argmin(np.abs(x_xz - x_center)))
    
    for direction, coords, profile in [
        ('x', x_vals, T_top[iy, :]),
        ('y', y_vals, T_top[:, ix]),
        ('z', z_vals, T_xz[:, ix_xz])
    ]:
        fname = f"{OUT_DIR}/{direction}_spectral_latent_heat.txt"
        np.savetxt(fname, np.vstack([coords, profile]).T, header=f'{direction}(m) T(K)', fmt='% .6e')
        print(f"Saved: {fname}")

def T_corr_to_modes(T_corr, geom):
    """
    Convert a local temperature correction field to spectral mode corrections.
    Uses precomputed fine mesh cosine bases from geom.
    Optimized using explicit matrix multiplications.
    """
    if not geom.fine_mesh_initialized:
        raise RuntimeError("Fine mesh not initialized. Call geom.update_fine_mesh(laser) first.")
    
    # T_corr shape: (nx_box, ny_box, nz_box)
    nx_box, ny_box, nz_box = T_corr.shape
    nx, _ = geom.Bx_fine.shape
    ny, _ = geom.By_fine.shape
    nz, _ = geom.Bz_fine.shape
    
    # 1. Contract X: (nx, nx_box) @ (nx_box, ny_box*nz_box) -> (nx, ny_box*nz_box)
    # Reshape T_corr to combine Y and Z
    T_reshaped = T_corr.reshape(nx_box, ny_box * nz_box)
    temp1 = geom.Bx_fine @ T_reshaped
    
    # 2. Contract Y: (ny, ny_box) @ (ny_box, nx*nz_box) -> (ny, nx*nz_box)
    # Reshape temp1 to (nx, ny_box, nz_box) and transpose to (ny_box, nx, nz_box)
    # Then flatten last two dims
    temp1 = temp1.reshape(nx, ny_box, nz_box).transpose(1, 0, 2).reshape(ny_box, nx * nz_box)
    temp2 = geom.By_fine @ temp1
    
    # 3. Contract Z: (nz, nz_box) @ (nz_box, ny*nx) -> (nz, ny*nx)
    # Reshape temp2 to (ny, nx, nz_box) and transpose to (nz_box, ny, nx)
    # Then flatten last two dims
    temp2 = temp2.reshape(ny, nx, nz_box).transpose(2, 0, 1).reshape(nz_box, ny * nx)
    delta_a = geom.Bz_fine @ temp2
    
    # Final reshape to (nz, ny, nx)
    delta_a = delta_a.reshape(nz, ny, nx)
    
    # Multiply by dV for the numerical integration
    delta_a *= geom.dV_fine
    
    return delta_a.astype(np.float32)
