import numpy as np
import pyfftw
from pyfftw.interfaces.scipy_fft import dctn
import matplotlib.pyplot as plt
import time
from numba import njit, prange
import os

pyfftw.config.NUM_THREADS = os.cpu_count()

# Numba-parallel slice updater: compute a_temp[p,:,:] = aK[p,:,:] + KK_by_Cp[p,:,:] * B_scaled
@njit(parallel=True, fastmath=True)
def compute_a_temp_numba(aK, KK_by_Cp, B_scaled, a_temp_out):
    nz = aK.shape[0]
    for p in prange(nz):
        # elementwise multiply KK_by_Cp[p] (ny,nx) with B_scaled (ny,nx)
        a_temp_out[p, :, :] = aK[p, :, :] + KK_by_Cp[p, :, :] * B_scaled


@njit(parallel=True, fastmath=True)
def compute_a_temp_ETD2(a_phi0, KK_phi1, KKK_phi2, S_prev, S_next, a_temp_out):
    """ETD2 update: a_temp = a_phi0 + KK_phi1*S_prev + KKK_phi2*(S_next - S_prev)"""
    nz = a_phi0.shape[0]
    for p in prange(nz):
        for i in range(a_phi0.shape[1]):
            for j in range(a_phi0.shape[2]):
                dS = S_next[i, j] - S_prev[i, j]
                a_temp_out[p, i, j] = a_phi0[p, i, j] + KK_phi1[p, i, j] * S_prev[i, j] + KKK_phi2[p, i, j] * dS


#make .out directory if not exists
OUT_DIR = ".out"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
#  GEOMETRY PARAMETERS (
# ============================================================

class GeomParams:
    def __init__(self, Lx, Ly, Lz, num, phys):
        """Geometry and precomputed cosine bases (NumPy).

        Everything is stored as NumPy arrays: `self.X`, `self.Y`, and the
        precomputed cosine bases `Bx_np`, `By_np`, `Bz_np`.
        """
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.Lz = float(Lz)
        self.nx = num.nx
        self.ny = num.ny
        self.nz = num.nz
        self.dx = Lx / num.nx
        self.dy = Ly / num.ny
        self.dz = Lz / num.nz

        # coordinates (NumPy)
        x_np = np.linspace(0.0, self.Lx, self.nx, endpoint=False)
        y_np = np.linspace(0.0, self.Ly, self.ny, endpoint=False)
        z_np = np.linspace(0.0, self.Lz, self.nz)
        self.x = x_np.astype(np.float32)
        self.y = y_np.astype(np.float32)
        self.z = z_np.astype(np.float32)

        # meshgrid (NumPy)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        self.X = self.X.astype(np.float32)
        self.Y = self.Y.astype(np.float32)

        # Precompute normalization coefficients
        self.Cm = C_coef(self.nx, self.Lx)
        self.Cn = C_coef(self.ny, self.Ly)
        self.Cp = C_coef(self.nz, self.Lz)
        self.Cp32 = self.Cp.astype(np.float32)
        
        # Precompute DCT scaling constant
        self.dct_scale = np.float32((self.dx * self.dy) * np.sqrt((self.nx * self.ny) / (self.Lx * self.Ly)))
        self.recon_scale = np.float32(np.sqrt(self.nx * self.ny) / np.sqrt(self.Lx * self.Ly))
        
        # Laser coefficient base
        self.laser_coef = phys.Absorptivity * 2.0 * phys.P / (np.pi * phys.r_b ** 2)

        # Numerical checks
        lambda_max_z = (phys.k / (phys.rho * phys.Ceff)) * (np.pi * num.nz / self.Lz) ** 2
        dx_rb = self.dx / phys.r_b
        dy_rb = self.dy / phys.r_b
        vx_criterion = phys.vx / (10 * self.dx / num.dt)
        lambda_dt = lambda_max_z * num.dt
        
        print("Geometry parameters:")
        print(f"dx={self.dx:.3e}, dy={self.dy:.3e}, r_b={phys.r_b:.3e}, vx={phys.vx}, lambda_max_z*dt={lambda_dt:.3e}")
        print(f"dx/r_b={dx_rb:.3f}, dy/r_b={dy_rb:.3f}, vx/(10*dx/dt)={vx_criterion:.3f}")
        print("---------------------------------------------------------------------------")
        
        # Check resolution criteria and issue warnings
        warnings_issued = False
        if dx_rb > 0.4:
            print(f"WARNING: dx/r_b = {dx_rb:.3f} > 0.4 - Spatial resolution in x may be insufficient!")
            warnings_issued = True
        if dy_rb > 0.4:
            print(f"WARNING: dy/r_b = {dy_rb:.3f} > 0.4 - Spatial resolution in y may be insufficient!")
            warnings_issued = True
        if vx_criterion > 1.0:
            print(f"WARNING: vx = {phys.vx:.3f} > 10*dx/dt = {10*self.dx/num.dt:.3f} - Laser moves too fast for time resolution!")
            warnings_issued = True
        if lambda_dt < 100:
            print(f"WARNING: lambda_max_z*dt = {lambda_dt:.3e} < 100 - Time step may be too small or not enough modes for convergence!")
            warnings_issued = True
        
        if not warnings_issued:
            print("All resolution criteria satisfied.")
        print("---------------------------------------------------------------------------")

# ============================================================
#   MATERIAL & LASER PARAMETERS
# ============================================================

class PhysParams:
    def __init__(self, rho, Cp, k,
                 P, Absorptivity, r_b, x0, y0, vx,
                 L_f=267700.0, DeltaH_LV=7.41e6, R_v=150.774, T0=293.0, Pa = 101325.0, 
                 T_boil=3090.0, T_liquidus=1800.0, T_solidus=1700.0):

        self.rho = rho
        self.Cp = Cp
        self.k = k
        self.L_f = L_f
        print("Warning: Latent heat currently disabled in Ceff calculation.")
        # Compute effective heat capacity including latent heat of fusion
        # C_eff = C_p + L_f / (T_L - T_S)
        self.Ceff = Cp #+ L_f / (T_liquidus - T_solidus)  temporarily disabled

        self.P = P
        self.Absorptivity = Absorptivity
        self.r_b = r_b
        self.x0 = x0
        self.y0 = y0
        self.vx = vx

        self.DeltaH_LV = DeltaH_LV
        self.R_v = R_v
        self.T0 = T0
        self.Pa = Pa
        self.T_boil = T_boil
        self.T_liquidus = T_liquidus
        self.T_solidus = T_solidus
        
        # Display effective heat capacity
        print("\n" + "="*60)
        print("Material Properties:")
        print(f"Cp (base specific heat): {self.Cp:.2f} J/(kg·K)")
        print(f"L_f (latent heat of fusion): {self.L_f:.2f} J/kg")
        print(f"Melting range: T_S = {self.T_solidus:.0f} K, T_L = {self.T_liquidus:.0f} K")
        print("="*60 + "\n")


# ============================================================
#   NUMERICAL PARAMETERS
# ============================================================

class NumericalParams:
    def __init__(self, dt, t_final, nx, ny, nz, ETD='ETD1', debug=False):
        self.dt = dt
        self.t_final = t_final
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ETD = ETD  # 'ETD1' or 'ETD2'
        self.debug = debug

        self.K = None   # CPU arrays (nz, ny, nx)
        self.KK = None
        self.KK_by_Cp = None
        # ETD2 coefficients
        self.K_phi0 = None  # phi_0 * a (for ETD2)
        self.KK_phi1 = None  # dt/(rho*Cp) * phi_1 (for ETD2)
        self.KKK_phi2 = None  # dt/(rho*Cp) * phi_2 (for ETD2)
        self.q_diff = None
        self.B_buffer = None
        self.a_temp = None
        self.aK = None
        self.S_n = None  # Forcing term from previous time step (for ETD2)
        self.q_evap_old = np.zeros((ny, nx), dtype=np.float32)


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
    """Compute ETD phi functions: phi_0(z), phi_1(z), phi_2(z).
    
    phi_0(z) = exp(z)
    phi_1(z) = (exp(z) - 1) / z
    phi_2(z) = (exp(z) - 1 - z) / z^2
    ===> to be checked, taylor is necessary ? 
    Returns: (phi_0, phi_1, phi_2) as arrays matching z.shape
    """
    # Handle small z values to avoid division by zero
    small_threshold = 1e-6
    
    phi_0 = np.exp(z)
    
    # phi_1
    mask_small = np.abs(z) < small_threshold
    phi_1 = np.zeros_like(z)
    phi_1[~mask_small] = (np.exp(z[~mask_small]) - 1.0) / z[~mask_small]
    phi_1[mask_small] = 1.0 + z[mask_small] / 2.0 + z[mask_small]**2 / 6.0  # Taylor series
    
    # phi_2
    phi_2 = np.zeros_like(z)
    phi_2[~mask_small] = (np.exp(z[~mask_small]) - 1.0 - z[~mask_small]) / (z[~mask_small]**2)
    phi_2[mask_small] = 0.5 + z[mask_small] / 6.0 + z[mask_small]**2 / 24.0  # Taylor series
    
    return phi_0, phi_1, phi_2


# ============================================================
#   HEAT FLUX (GPU)
# ============================================================

def q_laser(geom, t, phys):
    """Gaussian laser heat flux."""
    x0t = phys.x0 + phys.vx * t
    r_sq = (geom.X - x0t) ** 2 + (geom.Y - phys.y0) ** 2
    return (geom.laser_coef * np.exp(-2.0 * r_sq / phys.r_b ** 2)).astype(np.float32)


def q_evap_point(T: np.ndarray, phys: PhysParams) -> np.ndarray:
    """Evaporative heat flux."""
    q = 0.82 * phys.DeltaH_LV * phys.Pa/ np.sqrt(2 * np.pi * phys.R_v * T) * \
        np.exp((phys.DeltaH_LV / (phys.R_v * phys.T_boil)) * (1.0 - phys.T_boil / T))
    q[T < phys.T_liquidus] = 0.0
    return q.astype(np.float32)


def shift_flux_along_x(field: np.ndarray, shift: float, geom: GeomParams) -> np.ndarray:
    """Translate a surface flux field along +x by ``shift`` meters using linear interpolation."""
    if field is None or field.size == 0:
        return np.zeros((geom.ny, geom.nx), dtype=np.float32)

    if abs(shift) < 1e-12:
        return field.astype(np.float32, copy=True)

    x_coords = geom.x.astype(np.float64)
    x_shifted = (x_coords - shift).astype(np.float64)
    shifted = np.empty_like(field, dtype=np.float32)

    for j in range(field.shape[0]):
        row = np.asarray(field[j], dtype=np.float64)
        shifted_row = np.interp(x_shifted, x_coords, row, left=0.0, right=0.0)
        shifted[j] = shifted_row.astype(np.float32)

    return shifted

# ============================================================
#   DCT-II 2D 
# ============================================================

def DCT_II(q):
    """2D DCT-II on surface q."""
    return dctn(q.astype(np.float32, copy=False), type=2, norm='ortho', workers=-1).astype(np.float32, copy=False)


# ============================================================
#   PRECOMPUTE 3D K / KK 
# ============================================================

def precompute_K_KK(phys, num, geom):
    nx, ny, nz = num.nx, num.ny, num.nz
    Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
    dt = num.dt

    m = np.arange(nx)[None, None, :]
    n = np.arange(ny)[None, :, None]
    p = np.arange(nz)[:, None, None]

    mu = (m * np.pi / Lx)**2 + (n * np.pi / Ly)**2 + (p * np.pi / Lz)**2
    lambda_j = (phys.k / (phys.rho * phys.Ceff)) * mu

    K = np.exp(-lambda_j * dt)
    K[0, 0, 0] = 1.0

    KK = np.zeros_like(K)
    mask = lambda_j > 0
    KK[mask] = (1 - K[mask]) / (phys.rho * phys.Ceff * lambda_j[mask])
    KK[~mask] = dt / (phys.rho * phys.Ceff)

    # For ETD2: compute phi-function-based coefficients
    if num.ETD == 'ETD2':
        z = -lambda_j * dt  # z = -lambda_j * dt
        phi_0, phi_1, phi_2 = phi_functions(z)
        
        dt_over_rhoCp = dt / (phys.rho * phys.Ceff)
        
        # Compute Cp normalization (needed for surface flux projection)
        Cp = C_coef(nz, Lz)[:, None, None]
        
        K_phi0 = phi_0.astype(np.float32)
        KK_phi1 = (dt_over_rhoCp * phi_1 * Cp).astype(np.float32)
        KKK_phi2 = (dt_over_rhoCp * phi_2 * Cp).astype(np.float32)
        
        return K.astype(np.float32), KK.astype(np.float32), K_phi0, KK_phi1, KKK_phi2
    
    return K.astype(np.float32), KK.astype(np.float32), None, None, None


# ============================================================
#   TIME STEP 
# ============================================================

def time_step(a, t, phys, num, geom, epsilon=2e+1):
    """Time stepping with ETD1 or ETD2 scheme."""
    
    if num.ETD == 'ETD1':
        # ============ ETD1: First-order (constant forcing) ============
        q_las = q_laser(geom, t, phys)

        # Warm-start evaporation flux using previous step translated by the laser travel distance
        x_shift = phys.vx * num.dt
        q_evap = shift_flux_along_x(num.q_evap_old, x_shift, geom)

        q_dct = DCT_II(q_las - q_evap)
        
        # Compute a*K
        np.multiply(a, num.K, out=num.aK, casting='same_kind')
        
        # Initial forcing S_n^{n+1} (before iteration)
        S_n = geom.dct_scale * q_dct
        omega = 0.1
        # Initial prediction: a_temp = a*K + KK_by_Cp * S_n
        np.multiply(geom.dct_scale, q_dct, out=num.B_buffer, casting='same_kind')
        compute_a_temp_numba(num.aK, num.KK_by_Cp, num.B_buffer, num.a_temp)
        S_current = S_n.copy()
        # Fixed-point iteration for nonlinear evaporation
        T_temp = reconstruct_temperature_top(num.a_temp, num, geom)
        for k in range(30):
            q_evap = q_evap_point(T_temp, phys)
            
            # Calculate the full difference (Laser - Evap)
            np.subtract(q_las, q_evap, out=num.q_diff, casting='same_kind')
            
            # Calculate the NEW target forcing term
            S_target = geom.dct_scale * DCT_II(num.q_diff)
            
            # --- UNDER-RELAXATION STEP ---
            # Instead of S_current = S_target, we blend them:
            # New = omega * Target + (1 - omega) * Old
            S_current = omega * S_target + (1.0 - omega) * S_current
                
            # Update simulation with this smoothed forcing
            np.multiply(1.0, S_current, out=num.B_buffer, casting='same_kind')
            compute_a_temp_numba(num.aK, num.KK_by_Cp, num.B_buffer, num.a_temp)
            
            T_old = T_temp
            T_temp = reconstruct_temperature_top(num.a_temp, num, geom)
            
            # Check convergence
            if np.max(np.abs(T_temp - T_old)) < epsilon:
                break
        
        # Store forcing and evaporation for next time step
        num.S_n = S_n.copy()
        num.q_evap_old = q_evap.astype(np.float32, copy=True)
        
        # Compute exact laser power (integral of q_las over domain)
        P_laser = np.sum(q_las) * geom.dx * geom.dy

        return num.a_temp, T_temp, P_laser, k+1
    
    elif num.ETD == 'ETD2':
        print("Warning: ETD2 scheme has no under relaxation.")
        # ============ ETD2: Second-order (linear forcing) ============
        # Using precomputed K_phi0, KK_phi1, KKK_phi2
        
        # Get forcing from previous time step (S_n^n)
        S_n_prev = num.S_n
        
        # Compute current laser flux and initial forcing estimate (without evaporation)
        q_las = q_laser(geom, t, phys)
        q_dct = DCT_II(q_las)
        S_n_next = geom.dct_scale * q_dct
        
        # Compute a*phi_0 once (stored in aK buffer)
        np.multiply(a, num.K_phi0, out=num.aK, casting='same_kind')
        
        # Initial prediction: a_temp = a*phi_0 + KK_phi1*S_prev + KKK_phi2*(S_next - S_prev)
        compute_a_temp_ETD2(num.aK, num.KK_phi1, num.KKK_phi2, S_n_prev, S_n_next, num.a_temp)
        
        # Fixed-point iteration for nonlinear evaporation
        T_temp = reconstruct_temperature_top(num.a_temp, num, geom)
        for k in range(20):
            q_evap = q_evap_point(T_temp, phys)
            np.subtract(q_las, q_evap, out=num.q_diff, casting='same_kind')
            
            # Update S_n^{n+1}
            S_n_next = geom.dct_scale * DCT_II(num.q_diff)
            
            # Update a_temp with new forcing (reuse aK which contains a*phi_0)
            compute_a_temp_ETD2(num.aK, num.KK_phi1, num.KKK_phi2, S_n_prev, S_n_next, num.a_temp)
            
            T_old = T_temp
            T_temp = reconstruct_temperature_top(num.a_temp, num, geom)
            if np.max(np.abs(T_temp - T_old)) < epsilon:
                break
        
        # Store forcing for next time step
        num.S_n = S_n_next.copy()
        
        # Compute exact laser power (integral of q_las over domain)
        P_laser = np.sum(q_las) * geom.dx * geom.dy
        
        return num.a_temp, T_temp, P_laser, k+1
    
    else:
        raise ValueError(f"Unknown ETD scheme: {num.ETD}. Use 'ETD1' or 'ETD2'.")


# ============================================================
#   RUN SIMULATION
# ============================================================

def run_simulation(phys, num, geom):
    print(f"Precomputing K, KK, buffers... (using {num.ETD})")
    K, KK, K_phi0, KK_phi1, KKK_phi2 = precompute_K_KK(phys, num, geom)
    num.K = K
    num.KK = KK
    num.KK_by_Cp = (num.KK * geom.Cp[:, None, None]).astype(np.float32)
    
    # Store ETD2 coefficients if applicable
    if num.ETD == 'ETD2':
        num.K_phi0 = K_phi0
        num.KK_phi1 = KK_phi1
        num.KKK_phi2 = KKK_phi2
    
    num.q_diff = np.empty((num.ny, num.nx), dtype=np.float32)
    num.B_buffer = np.empty((num.ny, num.nx), dtype=np.float32)
    num.a_temp = np.empty((num.nz, num.ny, num.nx), dtype=np.float32)
    num.aK = np.empty((num.nz, num.ny, num.nx), dtype=np.float32)
    num.S_n = np.zeros((num.ny, num.nx), dtype=np.float32)  # Initialize forcing term
    num.q_evap_old.fill(0.0)
    
    a = np.zeros((num.nz, num.ny, num.nx), dtype=np.float32)
    a[0,0,0] = phys.T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz)
    
    nsteps = int(np.ceil(num.t_final / num.dt))
    T_top_history = []
    P_laser_history = []
    start = time.perf_counter()
    
    for step in range(nsteps+1):
        t = step * num.dt
        a, T_top, P_laser, n_iter = time_step(a, t, phys, num, geom)
        print(f"Step {step}/{nsteps} | t={t:.6e}s | Peak T: {np.max(T_top):.2f} K | P_laser: {P_laser:.3f} W | Iterations: {n_iter}")
        T_top_history.append(T_top)
        P_laser_history.append(P_laser)

    elapsed = time.perf_counter() - start
    print(f"\nTotal: {elapsed:.3f}s ({nsteps} steps, {elapsed/nsteps:.3f}s/step)")
    print(f"Laser power: min={min(P_laser_history):.3f} W, max={max(P_laser_history):.3f} W, mean={np.mean(P_laser_history):.3f} W")
    print(f"Nominal laser power: {phys.P} W")
    return a, T_top_history, P_laser_history


# ============================================================
#   RECONSTRUCT TEMPERATURE FIELD (GPU)
# ============================================================

def reconstruct_temperature_top(a, num, geom):
    """Reconstruct top surface temperature from modal coefficients."""
    A = (geom.Cp32[:, None, None] * a).sum(axis=0)
    return (geom.recon_scale * dctn(A, type=3, norm='ortho', axes=(0, 1), workers=-1)).astype(np.float32, copy=False)

def reconstruct_temperature_xz(a, num, geom, phys, y0=None, mode='full_field'):
    """Compact vectorized x-z slice at y=y0. Returns (x_vals, z_vals, T_xz).
    """
    if y0 is None:
        y0 = phys.y0

    nx, ny, nz = num.nx, num.ny, num.nz
    if ny == 0 or nx == 0 or nz == 0:
        return geom.x, geom.z, np.zeros((nz, nx), dtype=np.float32)

    n_idx = np.arange(ny)
    m_idx = np.arange(nx)
    p_idx = np.arange(nz)

    # Evaluate modal contributions along y at the requested slice
    cos_n_y0 = geom.Cn * np.cos(np.pi * n_idx * y0 / geom.Ly)  # (ny,)
    modal_pm = (a * cos_n_y0[None, :, None]).sum(axis=1)       # (nz, nx)

    # Project along x using the same cosine basis used for the top reconstruction
    modal_pm *= geom.Cm[None, :]
    cos_m_x = np.cos(np.pi * m_idx[:, None] * geom.x[None, :] / geom.Lx)  # (nx, nx)
    temp_p_x = modal_pm @ cos_m_x                                        # (nz, nx)

    # Project along z to obtain the spatial slice
    temp_p_x *= geom.Cp[:, None]
    cos_p_z = np.cos(np.pi * p_idx[:, None] * geom.z[None, :] / geom.Lz)  # (nz, nz)
    T_xz = (cos_p_z.T @ temp_p_x).astype(np.float32)                      # (nz, nx)

    z_vals = geom.z
    x_vals = geom.x

    if mode == 'meltpool':
        # Determine melt extents from the x-z slice itself
        melt_mask_xz = T_xz >= phys.T_liquidus
        
        # Check if there's any melt pool
        if not np.any(melt_mask_xz):
            # No melt pool - return full domain
            print("Warning: No melt pool detected (T < T_liquidus everywhere), returning full domain")
            return x_vals, z_vals, T_xz
        
        x_mask = np.any(melt_mask_xz, axis=0)
        z_mask = np.any(melt_mask_xz, axis=1)
        x_min0, x_max0 = x_vals[x_mask].min(), x_vals[x_mask].max()
        z_min0, z_max0 = z_vals[z_mask].min(), z_vals[z_mask].max()
        mp_len = x_max0 - x_min0
        mp_depth = z_max0 - z_min0
        

        # Horizontal cropping: center on laser spot and extend 1.5x melt length
        half_x = 1.5 * mp_len / 2.0
        xc = phys.x0
        xmin, xmax = max(0.0, xc - half_x), min(geom.Lx, xc + half_x)
        ix0, ix1 = int(np.searchsorted(x_vals, xmin)), int(np.searchsorted(x_vals, xmax))
        if ix0 == ix1:
            ix0 = max(0, ix0 - 1); ix1 = min(nx, ix1 + 1)

        # Vertical cropping: extend vertical range to 2x melt depth (show front/back)
        zc = 0.5 * (z_min0 + z_max0) if np.any(melt_mask_xz) else mp_depth / 2.0
        half_z = max(mp_depth, mp_depth)  # base depth; we'll double it
        # target half span = mp_depth (so total = 2*mp_depth)
        zmin, zmax = max(0.0, zc - mp_depth), min(geom.Lz, zc + mp_depth)
        iz0, iz1 = int(np.searchsorted(z_vals, zmin)), int(np.searchsorted(z_vals, zmax))
        if iz0 == iz1:
            iz0 = max(0, iz0 - 1); iz1 = min(nz, iz1 + 1)

        x_sel = x_vals[ix0:ix1]
        z_sel = z_vals[iz0:iz1]
        return x_sel, z_sel, T_xz[iz0:iz1, ix0:ix1]

    return x_vals, z_vals, T_xz


def save_temp_profiles(a, num, geom, phys, t=None):
    """Save 1D temperature profiles through the highest temperature location."""
    t = num.t_final if t is None else t
    
    T_top = reconstruct_temperature_top(a, num, geom)
    x_vals, y_vals = geom.X[0, :], geom.Y[:, 0]
    
    # Find location of maximum temperature
    iy_max, ix_max = np.unravel_index(np.argmax(T_top), T_top.shape)
    x_center = x_vals[ix_max]
    
    ix = ix_max
    iy = int(np.argmin(np.abs(y_vals - phys.y0)))
    
    x_xz, z_vals, T_xz = reconstruct_temperature_xz(a, num, geom, phys, y0=phys.y0, mode='full_field')
    ix_xz = int(np.argmin(np.abs(x_xz - x_center)))
    
    for direction, coords, profile in [
        ('x', x_vals, T_top[iy, :]),
        ('y', y_vals, T_top[:, ix]),
        ('z', z_vals, T_xz[:, ix_xz])
    ]:
        fname = f"{OUT_DIR}/{direction}_spectral_latent_heat.txt"
        np.savetxt(fname, np.vstack([coords, profile]).T, header=f'{direction}(m) T(K)', fmt='% .6e')
        print(f"Saved: {fname}")

# ============================================================
#   MAIN SCRIPT
# ============================================================

phys = PhysParams(rho=7850, Cp=500, k=15, T0=293.0,
                  P=200.0, Absorptivity=0.30, r_b=6e-5,
                  x0=0.0, y0=0.0025, vx=0.8)

num = NumericalParams(dt=6e-6, t_final=0.012, nx=512, ny=256, nz=1000, ETD='ETD1')

geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, num=num, phys=phys)

a_final, T_top_history, P_laser_history = run_simulation(phys, num, geom)
save_temp_profiles(a_final, num, geom, phys)
T_final = T_top_history[-1]

# Save laser power history
np.savetxt(f"{OUT_DIR}/laser_power_history.csv", 
           np.column_stack([np.arange(len(P_laser_history)) * num.dt, P_laser_history]),
           header='time(s) P_laser(W)', delimiter=',', fmt='%.6e')
print(f"Saved laser power history to {OUT_DIR}/laser_power_history.csv")

# Probing a point on top surface
# chose x probe so that temperature max is reached on a chosen time step
x_probe = 30 * phys.vx * num.dt + phys.x0 
y_probe = 0.0025
ix = int(x_probe / geom.Lx * num.nx)
iy = int(y_probe / geom.Ly * num.ny)
T_probe = [T_top[iy, ix] for T_top in T_top_history]
# save probe history to csv
np.savetxt("T_probe.csv", np.array(T_probe), delimiter=",")
# for plotting (NumPy arrays)
X = geom.X
Y = geom.Y
T = T_final

# Compute q_laser and q_evap for final time
q_las = q_laser(geom, num.t_final, phys) 
q_eva = q_evap_point(T, phys)             

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])  # 2 rows, 2 cols, top bigger

# --- Temperature (top left) ---
ax0 = fig.add_subplot(gs[0, 0])
im0 = ax0.contourf(X*1e3, Y*1e3, T, levels=50, cmap='hot')
fig.colorbar(im0, ax=ax0, label='Temperature (K)')
ax0.set_title("Final Temperature Field")
ax0.set_xlabel("x (mm)")
ax0.set_ylabel("y (mm)")
ax0.set_aspect('equal')

# --- Temperature xz slice (top right) ---
ax1 = fig.add_subplot(gs[0, 1])
x_vals, z_vals, T_xz = reconstruct_temperature_xz(a_final, num, geom, phys, y0=phys.y0, mode='meltpool')
im1 = ax1.imshow(T_xz, aspect='auto',
                   extent=[x_vals[0]*1e3, x_vals[-1]*1e3, z_vals[0]*1e3, z_vals[-1]*1e3],
                   origin='upper', cmap='hot')
fig.colorbar(im1, ax=ax1, label='Temperature (K)')
ax1.set_title("Temperature x-z slice (y = {:.3f} m)".format(phys.y0))
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('z (mm)')

# --- Laser flux (bottom left) ---
ax1 = fig.add_subplot(gs[1, 0])
im1 = ax1.contourf(X*1e3, Y*1e3, q_las, levels=50, cmap='inferno')
fig.colorbar(im1, ax=ax1, label='q_laser (W/m²)')
ax1.set_title("Laser Heat Flux")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
ax1.set_aspect('equal')

# --- Evaporative flux (bottom right) ---
ax2 = fig.add_subplot(gs[1, 1])
im2 = ax2.contourf(X*1e3, Y*1e3, q_eva, levels=50, cmap='inferno')
fig.colorbar(im2, ax=ax2, label='q_evap (W/m²)')
ax2.set_title("Evaporative Flux")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()

# plot temperature at probe point over time
plt.figure(figsize=(8, 5))
time_array = np.arange(len(T_probe)) * num.dt
plt.plot(time_array * 1e3, T_probe, "-o")
plt.xlabel("Time (ms)")
plt.ylabel("Temperature at probe point (K)")
plt.title("Temperature at Probe Point Over Time")
plt.grid()
plt.show()



# ============================================================
#   SENSITIVITY ANALYSIS OVER nz
# ============================================================
"""
phys = PhysParams(rho=7850, Cp=500, k=15, T0=293.0,
                  P=200.0, Absorptivity=0.30, r_b=6e-5,
                  x0=0.0, y0=0.0025, vx=0.8)

num = NumericalParams(dt=6e-6, t_final=0.0006,
                      nx=512, ny=256, nz=1000)

geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, num=num, phys=phys)

nz_values = [50, 100, 200, 400, 800, 1200, 2000]  # choose feasible values
results_Tmax = []
results_width = []
results_length = []

def meltpool_metrics(T, phys, geom):
    mask = T > phys.T_liquidus
    if not np.any(mask):
        return np.max(T), 0.0, 0.0

    # width along y
    y_mask = np.any(mask, axis=1)
    y_coords = geom.Y[:, 0]
    width = y_coords[y_mask].max() - y_coords[y_mask].min()

    # length along x
    x_mask = np.any(mask, axis=0)
    x_coords = geom.X[0, :]
    length = x_coords[x_mask].max() - x_coords[x_mask].min()

    return np.max(T), width, length


for nz in nz_values:
    print(f"\n=== Running simulation for nz = {nz} ===")
    num = NumericalParams(dt=6e-6, t_final=0.0006, nx=512, ny=256, nz=nz)
    geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, num=num, phys=phys)
    
    a_final, _, _ = run_simulation(phys, num, geom)
    T_final = reconstruct_temperature_top(a_final, num, geom)
    
    Tmax, W, L = meltpool_metrics(T_final, phys, geom)
    results_Tmax.append(Tmax)
    results_width.append(W * 1e3)
    results_length.append(L * 1e3)
    print(f"Tmax = {Tmax:.1f} K, Width = {W*1e3:.3f} mm, Length = {L*1e3:.3f} mm")


# ----------- PLOTS ---------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(nz_values, results_Tmax, "o-")
plt.xlabel("nz")
plt.ylabel("Max temperature (K)")
plt.title("Tmax vs nz")

plt.subplot(1, 3, 2)
plt.plot(nz_values, results_width, "o-")
plt.xlabel("nz")
plt.ylabel("Melt pool width (mm)")
plt.title("Width vs nz")

plt.subplot(1, 3, 3)
plt.plot(nz_values, results_length, "o-")
plt.xlabel("nz")
plt.ylabel("Melt pool length (mm)")
plt.title("Length vs nz")

plt.tight_layout()
plt.show()

"""

# ============================================================
#   ETD1 vs ETD2 COMPARISON TEST
# ============================================================

print("\n" + "="*60)
print("ETD1 vs ETD2 COMPARISON TEST")
print("="*60)

# Shared parameters for comparison
phys_test = PhysParams(rho=7850, Cp=500, k=15, T0=293.0,
                       P=200.0, Absorptivity=0.30, r_b=6e-5,
                       x0=0.0, y0=0.0025, vx=0.8)

# Short simulation for comparison
num_etd1 = NumericalParams(dt=6e-5, t_final=0.0006, nx=512, ny=256, nz=1000, ETD='ETD1')
num_etd2 = NumericalParams(dt=6e-5, t_final=0.0006, nx=512, ny=256, nz=1000, ETD='ETD2')

# Run ETD1
print("\nRunning ETD1 simulation...")
geom_etd1 = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, num=num_etd1, phys=phys_test)
a_final_etd1, T_history_etd1, P_laser_etd1 = run_simulation(phys_test, num_etd1, geom_etd1)
T_final_etd1 = T_history_etd1[-1]

# Run ETD2
print("\nRunning ETD2 simulation...")
geom_etd2 = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, num=num_etd2, phys=phys_test)
a_final_etd2, T_history_etd2, P_laser_etd2 = run_simulation(phys_test, num_etd2, geom_etd2)
T_final_etd2 = T_history_etd2[-1]

# Extract probe temperature along x
x_probe = 30 * phys_test.vx * num_etd1.dt + phys_test.x0
y_probe = 0.0025
ix_probe = int(x_probe / geom_etd1.Lx * num_etd1.nx)
iy_probe = int(y_probe / geom_etd1.Ly * num_etd1.ny)

T_probe_etd1 = [T_top[iy_probe, ix_probe] for T_top in T_history_etd1]
T_probe_etd2 = [T_top[iy_probe, ix_probe] for T_top in T_history_etd2]

# Create comparison plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Probe temperature over time
time_array = np.arange(len(T_probe_etd1)) * num_etd1.dt
axes[0].plot(time_array * 1e3, T_probe_etd1, 'b-', label='ETD1', linewidth=2)
axes[0].plot(time_array * 1e3, T_probe_etd2, 'r--', label='ETD2', linewidth=2)
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Temperature (K)")
axes[0].set_title(f"Temperature at Probe Point (x={x_probe*1e3:.2f}mm, y={y_probe*1e3:.2f}mm)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Final temperature field along x (at y=y0)
x_vals = geom_etd1.X[0, :] * 1e3  # Convert to mm
T_x_etd1 = T_final_etd1[iy_probe, :]
T_x_etd2 = T_final_etd2[iy_probe, :]

axes[1].plot(x_vals, T_x_etd1, 'b-', label='ETD1', linewidth=2)
axes[1].plot(x_vals, T_x_etd2, 'r--', label='ETD2', linewidth=2)
axes[1].set_xlabel("x (mm)")
axes[1].set_ylabel("Temperature (K)")
axes[1].set_title(f"Final Temperature Profile along x (t={num_etd1.t_final*1e3:.2f}ms)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ETD_comparison.png", dpi=150)
print(f"\nComparison plot saved to {OUT_DIR}/ETD_comparison.png")
plt.show()

# Print statistics
print("\nComparison Statistics:")
print(f"Max temperature ETD1: {np.max(T_final_etd1):.2f} K")
print(f"Max temperature ETD2: {np.max(T_final_etd2):.2f} K")
print(f"Max difference: {np.max(np.abs(T_final_etd1 - T_final_etd2)):.2f} K")
print(f"RMS difference: {np.sqrt(np.mean((T_final_etd1 - T_final_etd2)**2)):.2f} K")
print(f"Relative RMS: {np.sqrt(np.mean((T_final_etd1 - T_final_etd2)**2)) / np.max(T_final_etd1) * 100:.2f}%")


# ============================================================
#   CONVERGENCE ANALYSIS
# ============================================================

print("\n" + "="*60)
print("CONVERGENCE ANALYSIS")
print("="*60)

# Reference parameters
phys_conv = PhysParams(rho=7850, Cp=500, k=15, T0=293.0,
                       P=200.0, Absorptivity=0.30, r_b=6e-5,
                       x0=0.0, y0=0.0025, vx=0.8)

t_final_conv = 6e-4  # Short simulation for convergence study
nx_ref, ny_ref, nz_ref, dt_ref = 512, 256, 1000, 6e-6
Lx, Ly, Lz = 0.01, 0.005, 0.0025

# Storage for results
results_nx = {'values': [], 'dx_rb': [], 'Tmax': []}
results_ny = {'values': [], 'dy_rb': [], 'Tmax': []}
results_nz = {'values': [], 'lambda_dt': [], 'Tmax': []}
results_dt = {'values': [], 'lambda_dt': [], 'Tmax': []}

# 1. Vary nx (keeping ny, nz, dt fixed)
print("\n1. Varying nx...")
nx_values = [128, 256, 384, 512, 640, 768]
for nx in nx_values:
    print(f"  nx = {nx}")
    num_conv = NumericalParams(dt=dt_ref, t_final=t_final_conv, nx=nx, ny=ny_ref, nz=nz_ref, ETD='ETD1')
    geom_conv = GeomParams(Lx=Lx, Ly=Ly, Lz=Lz, num=num_conv, phys=phys_conv)
    a_final_conv, T_history_conv, _ = run_simulation(phys_conv, num_conv, geom_conv)
    T_final_conv = T_history_conv[-1]
    
    dx = Lx / nx
    dx_rb = dx / phys_conv.r_b
    Tmax = np.max(T_final_conv)
    
    results_nx['values'].append(nx)
    results_nx['dx_rb'].append(dx_rb)
    results_nx['Tmax'].append(Tmax)
    print(f"    dx/r_b = {dx_rb:.3f}, Tmax = {Tmax:.2f} K")

# 2. Vary ny (keeping nx, nz, dt fixed)
print("\n2. Varying ny...")
ny_values = [128, 192, 256, 320, 384]
for ny in ny_values:
    print(f"  ny = {ny}")
    num_conv = NumericalParams(dt=dt_ref, t_final=t_final_conv, nx=nx_ref, ny=ny, nz=nz_ref, ETD='ETD1')
    geom_conv = GeomParams(Lx=Lx, Ly=Ly, Lz=Lz, num=num_conv, phys=phys_conv)
    a_final_conv, T_history_conv, _ = run_simulation(phys_conv, num_conv, geom_conv)
    T_final_conv = T_history_conv[-1]
    
    dy = Ly / ny
    dy_rb = dy / phys_conv.r_b
    Tmax = np.max(T_final_conv)
    
    results_ny['values'].append(ny)
    results_ny['dy_rb'].append(dy_rb)
    results_ny['Tmax'].append(Tmax)
    print(f"    dy/r_b = {dy_rb:.3f}, Tmax = {Tmax:.2f} K")

# 3. Vary nz (keeping nx, ny, dt fixed)
print("\n3. Varying nz...")
nz_values = [200, 400, 600, 800, 1000, 1200]
for nz in nz_values:
    print(f"  nz = {nz}")
    num_conv = NumericalParams(dt=dt_ref, t_final=t_final_conv, nx=nx_ref, ny=ny_ref, nz=nz, ETD='ETD1')
    geom_conv = GeomParams(Lx=Lx, Ly=Ly, Lz=Lz, num=num_conv, phys=phys_conv)
    a_final_conv, T_history_conv, _ = run_simulation(phys_conv, num_conv, geom_conv)
    T_final_conv = T_history_conv[-1]
    
    # Compute lambda_max_z * dt
    lambda_max_z = (phys_conv.k / (phys_conv.rho * phys_conv.Ceff)) * (np.pi * nz / Lz) ** 2
    lambda_dt = lambda_max_z * dt_ref
    Tmax = np.max(T_final_conv)
    
    results_nz['values'].append(nz)
    results_nz['lambda_dt'].append(lambda_dt)
    results_nz['Tmax'].append(Tmax)
    print(f"    lambda_max_z*dt = {lambda_dt:.3e}, Tmax = {Tmax:.2f} K")

# 4. Vary dt (keeping nx, ny, nz fixed)
print("\n4. Varying dt...")
dt_values = [3e-6, 4.5e-6, 6e-6, 9e-6, 12e-6]
for dt in dt_values:
    print(f"  dt = {dt:.2e}")
    num_conv = NumericalParams(dt=dt, t_final=t_final_conv, nx=nx_ref, ny=ny_ref, nz=nz_ref, ETD='ETD1')
    geom_conv = GeomParams(Lx=Lx, Ly=Ly, Lz=Lz, num=num_conv, phys=phys_conv)
    a_final_conv, T_history_conv, _ = run_simulation(phys_conv, num_conv, geom_conv)
    T_final_conv = T_history_conv[-1]
    
    # Compute lambda_max_z * dt
    lambda_max_z = (phys_conv.k / (phys_conv.rho * phys_conv.Ceff)) * (np.pi * nz_ref / Lz) ** 2
    lambda_dt = lambda_max_z * dt
    Tmax = np.max(T_final_conv)
    
    results_dt['values'].append(dt)
    results_dt['lambda_dt'].append(lambda_dt)
    results_dt['Tmax'].append(Tmax)
    print(f"    lambda_max_z*dt = {lambda_dt:.3e}, Tmax = {Tmax:.2f} K")

# Create convergence plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: nx convergence (dx/r_b vs Tmax)
axes[0, 0].plot(results_nx['dx_rb'], results_nx['Tmax'], 'o-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel(r"$\Delta x / r_b$")
axes[0, 0].set_ylabel("Max Temperature (K)")
axes[0, 0].set_title(f"Spatial Convergence in x (ny={ny_ref}, nz={nz_ref}, dt={dt_ref:.1e})")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].invert_xaxis()  # Smaller dx/rb is better (higher resolution)

# Plot 2: ny convergence (dy/r_b vs Tmax)
axes[0, 1].plot(results_ny['dy_rb'], results_ny['Tmax'], 's-', linewidth=2, markersize=8, color='tab:orange')
axes[0, 1].set_xlabel(r"$\Delta y / r_b$")
axes[0, 1].set_ylabel("Max Temperature (K)")
axes[0, 1].set_title(f"Spatial Convergence in y (nx={nx_ref}, nz={nz_ref}, dt={dt_ref:.1e})")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].invert_xaxis()

# Plot 3: nz convergence (lambda_max_z*dt vs Tmax)
axes[1, 0].plot(results_nz['lambda_dt'], results_nz['Tmax'], '^-', linewidth=2, markersize=8, color='tab:green')
axes[1, 0].set_xlabel(r"$\lambda_{max,z} \cdot \Delta t$")
axes[1, 0].set_ylabel("Max Temperature (K)")
axes[1, 0].set_title(f"Spatial Convergence in z (nx={nx_ref}, ny={ny_ref}, dt={dt_ref:.1e})")
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: dt convergence (lambda_max_z*dt vs Tmax)
axes[1, 1].plot(results_dt['lambda_dt'], results_dt['Tmax'], 'd-', linewidth=2, markersize=8, color='tab:red')
axes[1, 1].set_xlabel(r"$\lambda_{max,z} \cdot \Delta t$")
axes[1, 1].set_ylabel("Max Temperature (K)")
axes[1, 1].set_title(f"Temporal Convergence (nx={nx_ref}, ny={ny_ref}, nz={nz_ref})")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/convergence_analysis.png", dpi=150)
print(f"\nConvergence analysis plot saved to {OUT_DIR}/convergence_analysis.png")
plt.show()

# Summary statistics
print("\nConvergence Summary:")
print(f"nx variation: Tmax range = [{min(results_nx['Tmax']):.2f}, {max(results_nx['Tmax']):.2f}] K, "
      f"variation = {max(results_nx['Tmax']) - min(results_nx['Tmax']):.2f} K")
print(f"ny variation: Tmax range = [{min(results_ny['Tmax']):.2f}, {max(results_ny['Tmax']):.2f}] K, "
      f"variation = {max(results_ny['Tmax']) - min(results_ny['Tmax']):.2f} K")
print(f"nz variation: Tmax range = [{min(results_nz['Tmax']):.2f}, {max(results_nz['Tmax']):.2f}] K, "
      f"variation = {max(results_nz['Tmax']) - min(results_nz['Tmax']):.2f} K")
print(f"dt variation: Tmax range = [{min(results_dt['Tmax']):.2f}, {max(results_dt['Tmax']):.2f}] K, "
      f"variation = {max(results_dt['Tmax']) - min(results_dt['Tmax']):.2f} K")




