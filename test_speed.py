import numpy as np
import pyfftw
from pyfftw.interfaces.scipy_fft import dctn
from scipy.ndimage import shift as scipy_shift
import matplotlib.pyplot as plt
import time
from numba import njit, prange
import os

pyfftw.config.NUM_THREADS = os.cpu_count()

# Numba-parallel slice updater
@njit(parallel=True, fastmath=True)
def compute_a_temp_numba(aK, KK_by_Cp, B_scaled, a_temp_out):
    nz = aK.shape[0]
    for p in prange(nz):
        a_temp_out[p, :, :] = aK[p, :, :] + KK_by_Cp[p, :, :] * B_scaled

@njit(parallel=True, fastmath=True)
def compute_a_temp_ETD2(a_phi0, KK_phi1, KKK_phi2, S_prev, S_next, a_temp_out):
    nz = a_phi0.shape[0]
    for p in prange(nz):
        for i in range(a_phi0.shape[1]):
            for j in range(a_phi0.shape[2]):
                dS = S_next[i, j] - S_prev[i, j]
                a_temp_out[p, i, j] = a_phi0[p, i, j] + KK_phi1[p, i, j] * S_prev[i, j] + KKK_phi2[p, i, j] * dS

OUT_DIR = ".out"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
#  LASER CLASS
# ============================================================

class Laser:
    def __init__(self, P, r_b, x0, y0, v, Absorptivity):
        """
        Laser parameters and state.
        
        :param P: Power (W)
        :param r_b: Beam radius (m)
        :param x0: Initial x position (m)
        :param y0: Initial y position (m)
        :param v: Velocity tuple (vx, vy) in m/s
        :param Absorptivity: Laser absorptivity (0-1)
        """
        self.P = P
        self.r_b = r_b
        self.x0 = x0
        self.y0 = y0
        self.v = np.array(v, dtype=np.float64) # Ensure array for vector math
        self.Absorptivity = Absorptivity
        
        # Current position (initialized to start)
        self.x = x0
        self.y = y0
        
    def update(self, dt):
        """Update current x and y positions based on velocity and time step."""
        self.x += self.v[0] * dt
        self.y += self.v[1] * dt

# ============================================================
#  GEOMETRY PARAMETERS
# ============================================================

class GeomParams:
    def __init__(self, Lx, Ly, Lz, num, phys, laser):
        """Geometry and precomputed cosine bases (NumPy)."""
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
        
        # Laser coefficient base (Uses Laser object parameters)
        self.laser_coef = laser.Absorptivity * 2.0 * laser.P / (np.pi * laser.r_b ** 2)

        # Numerical checks
        lambda_max_z = (phys.k / (phys.rho * phys.Ceff)) * (np.pi * num.nz / self.Lz) ** 2
        dx_rb = self.dx / laser.r_b
        dy_rb = self.dy / laser.r_b
        
        # Criterion for x and y velocity (Uses Laser object velocity)
        v_mag = np.sqrt(laser.v[0]**2 + laser.v[1]**2)
        v_criterion = v_mag / (10 * self.dx / num.dt) if v_mag > 0 else 0
        lambda_dt = lambda_max_z * num.dt
        
        print("Geometry parameters:")
        print(f"dx={self.dx:.3e}, dy={self.dy:.3e}, r_b={laser.r_b:.3e}")
        print(f"v={laser.v}, lambda_max_z*dt={lambda_dt:.3e}")
        print(f"dx/r_b={dx_rb:.3f}, dy/r_b={dy_rb:.3f}")
        print(f"v_crit={v_criterion:.3f}")
        print("---------------------------------------------------------------------------")
        
        # Check resolution criteria
        warnings_issued = False
        if dx_rb > 0.4:
            print(f"WARNING: dx/r_b = {dx_rb:.3f} > 0.4 - Spatial resolution in x may be insufficient!")
            warnings_issued = True
        if dy_rb > 0.4:
            print(f"WARNING: dy/r_b = {dy_rb:.3f} > 0.4 - Spatial resolution in y may be insufficient!")
            warnings_issued = True
        if v_criterion > 1.0:
            print(f"WARNING: v moves > 10*dx/dt! Laser moves too fast for time resolution!")
            warnings_issued = True
        if lambda_dt < 100:
            print(f"WARNING: lambda_max_z*dt = {lambda_dt:.3e} < 100 - Time step may be too small or not enough modes for convergence!")
            warnings_issued = True
        
        if not warnings_issued:
            print("All resolution criteria satisfied.")
        print("---------------------------------------------------------------------------")

# ============================================================
#   MATERIAL PARAMETERS
# ============================================================

class PhysParams:
    def __init__(self, rho, Cp, k,
                 L_f=267700.0, DeltaH_LV=7.41e6, R_v=150.774, T0=293.0, Pa = 101325.0, 
                 T_boil=3090.0, T_liquidus=1800.0, T_solidus=1700.0):
        """Material properties only. Laser params moved to Laser class."""
        self.rho = rho
        self.Cp = Cp
        self.k = k
        self.L_f = L_f
        self.Ceff = Cp 

        self.DeltaH_LV = DeltaH_LV
        self.R_v = R_v
        self.T0 = T0
        self.Pa = Pa
        self.T_boil = T_boil
        self.T_liquidus = T_liquidus
        self.T_solidus = T_solidus
        
        print("\n" + "="*60)
        print("Material Properties:")
        print(f"Cp (base specific heat): {self.Cp:.2f} J/(kg·K)")
        print(f"L_f (latent heat of fusion): {self.L_f:.2f} J/kg")
        print(f"Melting range: T_S = {self.T_solidus:.0f} K, T_L = {self.T_liquidus:.0f} K")


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
        self.K_phi0 = None
        self.KK_phi1 = None
        self.KKK_phi2 = None
        self.q_diff = None
        self.B_buffer = None
        self.a_temp = None
        self.aK = None
        self.S_n = None
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
#   HEAT FLUX (GPU)
# ============================================================

def q_laser(geom, laser):
    """Gaussian laser heat flux using current Laser position (laser.x, laser.y)."""
    r_sq = (geom.X - laser.x) ** 2 + (geom.Y - laser.y) ** 2
    return (geom.laser_coef * np.exp(-2.0 * r_sq / laser.r_b ** 2)).astype(np.float32)


def q_evap_point(T: np.ndarray, phys: PhysParams) -> np.ndarray:
    """Evaporative heat flux."""
    q = 0.82 * phys.DeltaH_LV * phys.Pa/ np.sqrt(2 * np.pi * phys.R_v * T) * \
        np.exp((phys.DeltaH_LV / (phys.R_v * phys.T_boil)) * (1.0 - phys.T_boil / T))
    q[T < phys.T_liquidus] = 0.0
    return q.astype(np.float32)


def shift_flux(field: np.ndarray, shift: tuple, geom: GeomParams) -> np.ndarray:
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

    if num.ETD == 'ETD2':
        z = -lambda_j * dt
        phi_0, phi_1, phi_2 = phi_functions(z)
        dt_over_rhoCp = dt / (phys.rho * phys.Ceff)
        Cp = C_coef(nz, Lz)[:, None, None]
        
        K_phi0 = phi_0.astype(np.float32)
        KK_phi1 = (dt_over_rhoCp * phi_1 * Cp).astype(np.float32)
        KKK_phi2 = (dt_over_rhoCp * phi_2 * Cp).astype(np.float32)
        return K.astype(np.float32), KK.astype(np.float32), K_phi0, KK_phi1, KKK_phi2
    
    return K.astype(np.float32), KK.astype(np.float32), None, None, None


# ============================================================
#   TIME STEP 
# ============================================================

def time_step(a, phys, num, geom, laser, epsilon=2e+1):
    """Time stepping with ETD1 or ETD2 scheme using Laser object."""
    
    if num.ETD == 'ETD1':
        # Use Laser object for heat flux and velocity shift
        q_las = q_laser(geom, laser)

        # Warm-start evaporation by shifting previous solution
        x_shift = laser.v[0] * num.dt
        y_shift = laser.v[1] * num.dt

        q_evap = shift_flux(num.q_evap_old, (x_shift, y_shift), geom)

        q_dct = DCT_II(q_las - q_evap)
        
        np.multiply(a, num.K, out=num.aK, casting='same_kind')
        
        S_n = geom.dct_scale * q_dct
        omega = 0.1
        np.multiply(geom.dct_scale, q_dct, out=num.B_buffer, casting='same_kind')
        compute_a_temp_numba(num.aK, num.KK_by_Cp, num.B_buffer, num.a_temp)
        S_current = S_n.copy()
        
        T_temp = reconstruct_temperature_top(num.a_temp, num, geom)
        for k in range(30):
            q_evap = q_evap_point(T_temp, phys)
            
            np.subtract(q_las, q_evap, out=num.q_diff, casting='same_kind')
            S_target = geom.dct_scale * DCT_II(num.q_diff)
            S_current = omega * S_target + (1.0 - omega) * S_current
                
            np.multiply(1.0, S_current, out=num.B_buffer, casting='same_kind')
            compute_a_temp_numba(num.aK, num.KK_by_Cp, num.B_buffer, num.a_temp)
            
            T_old = T_temp
            T_temp = reconstruct_temperature_top(num.a_temp, num, geom)
            
            if np.max(np.abs(T_temp - T_old)) < epsilon:
                break
        
        num.S_n = S_n.copy()
        num.q_evap_old = q_evap.astype(np.float32, copy=True)
        P_laser = np.sum(q_las) * geom.dx * geom.dy

        return num.a_temp, T_temp, P_laser, k+1
    
    elif num.ETD == 'ETD2':
        print("Warning: ETD2 scheme has no under relaxation.")
        S_n_prev = num.S_n
        
        q_las = q_laser(geom, laser)
        q_dct = DCT_II(q_las)
        S_n_next = geom.dct_scale * q_dct
        
        np.multiply(a, num.K_phi0, out=num.aK, casting='same_kind')
        compute_a_temp_ETD2(num.aK, num.KK_phi1, num.KKK_phi2, S_n_prev, S_n_next, num.a_temp)
        
        T_temp = reconstruct_temperature_top(num.a_temp, num, geom)
        for k in range(20):
            q_evap = q_evap_point(T_temp, phys)
            np.subtract(q_las, q_evap, out=num.q_diff, casting='same_kind')
            
            S_n_next = geom.dct_scale * DCT_II(num.q_diff)
            compute_a_temp_ETD2(num.aK, num.KK_phi1, num.KKK_phi2, S_n_prev, S_n_next, num.a_temp)
            
            T_old = T_temp
            T_temp = reconstruct_temperature_top(num.a_temp, num, geom)
            if np.max(np.abs(T_temp - T_old)) < epsilon:
                break
        
        num.S_n = S_n_next.copy()
        P_laser = np.sum(q_las) * geom.dx * geom.dy
        
        return num.a_temp, T_temp, P_laser, k+1
    else:
        raise ValueError(f"Unknown ETD scheme: {num.ETD}. Use 'ETD1' or 'ETD2'.")


# ============================================================
#   RUN SIMULATION
# ============================================================

def run_simulation(phys, num, geom, laser):
    print(f"Precomputing K, KK, buffers... (using {num.ETD})")
    # Initialiszation shouldn't be done here
    K, KK, K_phi0, KK_phi1, KKK_phi2 = precompute_K_KK(phys, num, geom)
    num.K = K
    num.KK = KK
    num.KK_by_Cp = (num.KK * geom.Cp[:, None, None]).astype(np.float32)
    
    if num.ETD == 'ETD2':
        num.K_phi0 = K_phi0
        num.KK_phi1 = KK_phi1
        num.KKK_phi2 = KKK_phi2
    
    num.q_diff = np.empty((num.ny, num.nx), dtype=np.float32)
    num.B_buffer = np.empty((num.ny, num.nx), dtype=np.float32)
    num.a_temp = np.empty((num.nz, num.ny, num.nx), dtype=np.float32)
    num.aK = np.empty((num.nz, num.ny, num.nx), dtype=np.float32)
    num.S_n = np.zeros((num.ny, num.nx), dtype=np.float32)
    num.q_evap_old.fill(0.0)
    
    a = np.zeros((num.nz, num.ny, num.nx), dtype=np.float32)
    a[0,0,0] = phys.T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz)
    
    nsteps = int(np.ceil(num.t_final / num.dt))
    T_top_history = []
    P_laser_history = []
    start = time.perf_counter()
    
    for step in range(nsteps+1):
        t = step * num.dt
        
        # 1. Compute time step using current laser position
        a, T_top, P_laser, n_iter = time_step(a, phys, num, geom, laser)
        
        # 2. Update laser position for the next step (Implicitly t -> t + dt)
        laser.update(num.dt)
        
        print(f"Step {step}/{nsteps} | t={t:.6e}s | Peak T: {np.max(T_top):.2f} K | P_laser: {P_laser:.3f} W | Iterations: {n_iter}")
        T_top_history.append(T_top)
        P_laser_history.append(P_laser)

    elapsed = time.perf_counter() - start
    print(f"\nTotal: {elapsed:.3f}s ({nsteps} steps, {elapsed/nsteps:.3f}s/step)")
    print(f"Laser power: min={min(P_laser_history):.3f} W, max={max(P_laser_history):.3f} W, mean={np.mean(P_laser_history):.3f} W")
    print(f"Nominal laser power: {laser.P} W")
    return a, T_top_history, P_laser_history


# ============================================================
#   RECONSTRUCT TEMPERATURE FIELD (GPU)
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

# ============================================================
#   MAIN SCRIPT
# ============================================================

# 1. Initialize Laser
laser = Laser(P=200.0, r_b=6e-5, x0=0.0, y0=0.0025, v=(0.8, 0), Absorptivity=0.30)

# 2. Initialize Physics (Material only)
phys = PhysParams(rho=7850, Cp=500, k=15, T0=293.0)

# 3. Initialize Numerical Params
num = NumericalParams(dt=6e-6, t_final=0.0006, nx=512, ny=256, nz=1000, ETD='ETD1')

# 4. Initialize Geometry (Needs Laser for checks/coeffs)
geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, num=num, phys=phys, laser=laser)

# 5. Run Simulation
a_final, T_top_history, P_laser_history = run_simulation(phys, num, geom, laser)

# 6. Post-processing
save_temp_profiles(a_final, num, geom, phys, laser)
T_final = T_top_history[-1]

# Save laser power history
np.savetxt(f"{OUT_DIR}/laser_power_history.csv", 
           np.column_stack([np.arange(len(P_laser_history)) * num.dt, P_laser_history]),
           header='time(s) P_laser(W)', delimiter=',', fmt='%.6e')
print(f"Saved laser power history to {OUT_DIR}/laser_power_history.csv")

# Probing a point on top surface (Re-calculation of x_probe using current laser state logic)
# Note: Laser has moved during simulation. We want to probe relative to start.
x_probe = 30 * laser.v[0] * num.dt + laser.x0 
y_probe = 30 * laser.v[1] * num.dt + laser.y0
ix = int(x_probe / geom.Lx * num.nx)
iy = int(y_probe / geom.Ly * num.ny)
T_probe = [T_top[iy, ix] for T_top in T_top_history]
np.savetxt("T_probe.csv", np.array(T_probe), delimiter=",")

X = geom.X
Y = geom.Y
T = T_final

# Compute q_laser and q_evap for final time using final laser state
q_las = q_laser(geom, laser) 
q_eva = q_evap_point(T, phys)             

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

# --- Temperature (top left) ---
ax0 = fig.add_subplot(gs[0, 0])
im0 = ax0.contourf(X*1e3, Y*1e3, T, levels=50, cmap='hot')
fig.colorbar(im0, ax=ax0, label='Temperature (K)')
ax0.set_title("Final Temperature Field")
ax0.set_xlabel("x (mm)")
ax0.set_ylabel("y (mm)")
ax0.set_aspect('equal')

# --- Temperature xz slice (top right) ---
# Slice at current laser Y position
y_final = laser.y
ax1 = fig.add_subplot(gs[0, 1])
x_vals, z_vals, T_xz = reconstruct_temperature_xz(a_final, num, geom, phys, laser, y0=y_final, mode='meltpool')
im1 = ax1.imshow(T_xz, aspect='auto',
                   extent=[x_vals[0]*1e3, x_vals[-1]*1e3, z_vals[0]*1e3, z_vals[-1]*1e3],
                   origin='upper', cmap='hot')
fig.colorbar(im1, ax=ax1, label='Temperature (K)')
ax1.set_title("Temperature x-z slice (y = {:.3f} m)".format(y_final))
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

plt.figure(figsize=(8, 5))
time_array = np.arange(len(T_probe)) * num.dt
plt.plot(time_array * 1e3, T_probe, "-o")
plt.xlabel("Time (ms)")
plt.ylabel("Temperature at probe point (K)")
plt.title("Temperature at Probe Point Over Time")
plt.grid()
plt.show()