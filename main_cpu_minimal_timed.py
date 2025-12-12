import numpy as np
import pyfftw
from pyfftw.interfaces.scipy_fft import dctn
import time
from numba import njit, prange
import os

pyfftw.config.NUM_THREADS = os.cpu_count()
OUT_DIR = ".out"
os.makedirs(OUT_DIR, exist_ok=True)

# Global dictionary to store timing results
TIMERS = {
    'total_step': 0.0,
    'q_calc': 0.0,
    'dct_forward': 0.0,
    'spectral_update': 0.0,
    'reconstruct_sum': 0.0,
    'reconstruct_idct': 0.0,
    'q_evap': 0.0,
    'convergence_loop': 0.0
}

# ============================================================
#   NUMBA KERNELS
# ============================================================

@njit(parallel=True, fastmath=True)
def compute_a_temp_numba(aK, KK_by_Cp, B_scaled, a_temp_out):
    nz = aK.shape[0]
    for p in prange(nz):
        a_temp_out[p, :, :] = aK[p, :, :] + KK_by_Cp[p, :, :] * B_scaled

# ============================================================
#   CLASSES
# ============================================================

class GeomParams:
    def __init__(self, Lx, Ly, Lz, num, phys):
        self.Lx, self.Ly, self.Lz = float(Lx), float(Ly), float(Lz)
        self.nx, self.ny, self.nz = num.nx, num.ny, num.nz
        self.dx, self.dy, self.dz = Lx / num.nx, Ly / num.ny, Lz / num.nz

        self.x = np.linspace(0.0, self.Lx, self.nx, endpoint=False).astype(np.float32)
        self.y = np.linspace(0.0, self.Ly, self.ny, endpoint=False).astype(np.float32)
        self.z = np.linspace(0.0, self.Lz, self.nz).astype(np.float32)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        self.X, self.Y = self.X.astype(np.float32), self.Y.astype(np.float32)

        self.Cm = C_coef(self.nx, self.Lx)
        self.Cn = C_coef(self.ny, self.Ly)
        self.Cp = C_coef(self.nz, self.Lz)
        self.Cp32 = self.Cp.astype(np.float32)
        
        self.dct_scale = np.float32((self.dx * self.dy) * np.sqrt((self.nx * self.ny) / (self.Lx * self.Ly)))
        self.recon_scale = np.float32(np.sqrt(self.nx * self.ny) / np.sqrt(self.Lx * self.Ly))
        self.laser_coef = phys.Absorptivity * 2.0 * phys.P / (np.pi * phys.r_b ** 2)

class PhysParams:
    def __init__(self, rho, Cp, k, P, Absorptivity, r_b, x0, y0, vx,
                 L_f=267700.0, DeltaH_LV=7.41e6, R_v=150.774, T0=293.0, Pa=101325.0, 
                 T_boil=3090.0, T_liquidus=1800.0, T_solidus=1700.0):
        self.rho, self.Cp, self.k = rho, Cp, k
        self.L_f, self.Ceff = L_f, Cp 
        self.P, self.Absorptivity, self.r_b = P, Absorptivity, r_b
        self.x0, self.y0, self.vx = x0, y0, vx
        self.DeltaH_LV, self.R_v, self.T0, self.Pa = DeltaH_LV, R_v, T0, Pa
        self.T_boil, self.T_liquidus, self.T_solidus = T_boil, T_liquidus, T_solidus

class NumericalParams:
    def __init__(self, dt, t_final, nx, ny, nz):
        self.dt, self.t_final = dt, t_final
        self.nx, self.ny, self.nz = nx, ny, nz
        self.K = self.KK = self.KK_by_Cp = None
        self.q_diff = self.B_buffer = self.a_temp = self.aK = self.S_n = None
        self.q_evap_old = np.zeros((ny, nx), dtype=np.float32)

# ============================================================
#   HELPER FUNCTIONS
# ============================================================

def C_coef(N, L):
    C = np.sqrt(2.0 / L) * np.ones(N)
    C[0] = np.sqrt(1.0 / L)
    return C

def q_laser(geom, t, phys):
    x0t = phys.x0 + phys.vx * t
    r_sq = (geom.X - x0t) ** 2 + (geom.Y - phys.y0) ** 2
    return (geom.laser_coef * np.exp(-2.0 * r_sq / phys.r_b ** 2)).astype(np.float32)

def q_evap_point(T, phys):
    q = 0.82 * phys.DeltaH_LV * phys.Pa/ np.sqrt(2 * np.pi * phys.R_v * T) * \
        np.exp((phys.DeltaH_LV / (phys.R_v * phys.T_boil)) * (1.0 - phys.T_boil / T))
    q[T < phys.T_liquidus] = 0.0
    return q.astype(np.float32)

def shift_flux_along_x(field, shift, geom):
    if field is None or field.size == 0 or abs(shift) < 1e-12:
        return field.astype(np.float32, copy=True) if field is not None else np.zeros((geom.ny, geom.nx), dtype=np.float32)
    x_coords = geom.x.astype(np.float64)
    x_shifted = (x_coords - shift).astype(np.float64)
    shifted = np.empty_like(field, dtype=np.float32)
    for j in range(field.shape[0]):
        shifted[j] = np.interp(x_shifted, x_coords, field[j].astype(np.float64), left=0.0, right=0.0).astype(np.float32)
    return shifted

def DCT_II(q):
    return dctn(q.astype(np.float32, copy=False), type=2, norm='ortho', workers=-1).astype(np.float32, copy=False)

def precompute_K_KK(phys, num, geom):
    m = np.arange(num.nx)[None, None, :]
    n = np.arange(num.ny)[None, :, None]
    p = np.arange(num.nz)[:, None, None]
    mu = (m * np.pi / geom.Lx)**2 + (n * np.pi / geom.Ly)**2 + (p * np.pi / geom.Lz)**2
    lambda_j = (phys.k / (phys.rho * phys.Ceff)) * mu
    K = np.exp(-lambda_j * num.dt)
    K[0, 0, 0] = 1.0
    KK = np.zeros_like(K)
    mask = lambda_j > 0
    KK[mask] = (1 - K[mask]) / (phys.rho * phys.Ceff * lambda_j[mask])
    KK[~mask] = num.dt / (phys.rho * phys.Ceff)
    return K.astype(np.float32), KK.astype(np.float32)

# ============================================================
#   TIMED FUNCTIONS
# ============================================================

def reconstruct_temperature_top_timed(a, num, geom):
    # 1. Summation along Z (Spectral space reduction)
    t0 = time.perf_counter()
    # Broadcasting: (nz, 1, 1) * (nz, ny, nx) -> (nz, ny, nx) -> sum -> (ny, nx)
    # This is memory bandwidth intensive
    A = (geom.Cp32[:, None, None] * a).sum(axis=0)
    t1 = time.perf_counter()
    TIMERS['reconstruct_sum'] += (t1 - t0)
    
    # 2. Inverse DCT (2D)
    t2 = time.perf_counter()
    res = (geom.recon_scale * dctn(A, type=3, norm='ortho', axes=(0, 1), workers=-1)).astype(np.float32, copy=False)
    t3 = time.perf_counter()
    TIMERS['reconstruct_idct'] += (t3 - t2)
    
    return res

def time_step(a, t, phys, num, geom, epsilon=2e+1):
    t_start_step = time.perf_counter()
    
    # --- Flux Calculation ---
    t0 = time.perf_counter()
    q_las = q_laser(geom, t, phys)
    q_evap = shift_flux_along_x(num.q_evap_old, phys.vx * num.dt, geom)
    t1 = time.perf_counter()
    TIMERS['q_calc'] += (t1 - t0)
    
    # --- Forward DCT ---
    t0 = time.perf_counter()
    q_dct = DCT_II(q_las - q_evap)
    t1 = time.perf_counter()
    TIMERS['dct_forward'] += (t1 - t0)
    
    # --- Spectral Update (Linear Part) ---
    t0 = time.perf_counter()
    np.multiply(a, num.K, out=num.aK, casting='same_kind')
    S_n = geom.dct_scale * q_dct
    omega = 0.1
    
    np.multiply(geom.dct_scale, q_dct, out=num.B_buffer, casting='same_kind')
    compute_a_temp_numba(num.aK, num.KK_by_Cp, num.B_buffer, num.a_temp)
    t1 = time.perf_counter()
    TIMERS['spectral_update'] += (t1 - t0)
    
    S_current = S_n.copy()
    
    # --- Initial Reconstruction ---
    T_temp = reconstruct_temperature_top_timed(num.a_temp, num, geom)
    
    # --- Non-linear Loop ---
    t_loop_start = time.perf_counter()
    k = 0
    for k in range(30):
        # Evaporation Physics
        t_evap0 = time.perf_counter()
        q_evap = q_evap_point(T_temp, phys)
        t_evap1 = time.perf_counter()
        TIMERS['q_evap'] += (t_evap1 - t_evap0)
        
        # Update Source
        np.subtract(q_las, q_evap, out=num.q_diff, casting='same_kind')
        
        # Forward DCT inside loop
        t_dct0 = time.perf_counter()
        S_target = geom.dct_scale * DCT_II(num.q_diff)
        t_dct1 = time.perf_counter()
        TIMERS['dct_forward'] += (t_dct1 - t_dct0)
        
        S_current = omega * S_target + (1.0 - omega) * S_current
        
        # Spectral Update inside loop
        t_up0 = time.perf_counter()
        np.multiply(1.0, S_current, out=num.B_buffer, casting='same_kind')
        compute_a_temp_numba(num.aK, num.KK_by_Cp, num.B_buffer, num.a_temp)
        t_up1 = time.perf_counter()
        TIMERS['spectral_update'] += (t_up1 - t_up0)
        
        T_old = T_temp
        # Reconstruction inside loop
        T_temp = reconstruct_temperature_top_timed(num.a_temp, num, geom)
        
        if np.max(np.abs(T_temp - T_old)) < epsilon:
            break
            
    t_loop_end = time.perf_counter()
    TIMERS['convergence_loop'] += (t_loop_end - t_loop_start)
    
    num.S_n = S_n.copy()
    num.q_evap_old = q_evap.astype(np.float32, copy=True)
    P_laser = np.sum(q_las) * geom.dx * geom.dy
    
    t_end_step = time.perf_counter()
    TIMERS['total_step'] += (t_end_step - t_start_step)
    
    return num.a_temp, T_temp, P_laser, k+1

def run_simulation(phys, num, geom):
    print(f"Initializing simulation (Timed Run - 200 steps)...")
    K, KK = precompute_K_KK(phys, num, geom)
    num.K, num.KK = K, KK
    num.KK_by_Cp = (num.KK * geom.Cp[:, None, None]).astype(np.float32)
    
    num.q_diff = np.empty((num.ny, num.nx), dtype=np.float32)
    num.B_buffer = np.empty((num.ny, num.nx), dtype=np.float32)
    num.a_temp = np.empty((num.nz, num.ny, num.nx), dtype=np.float32)
    num.aK = np.empty((num.nz, num.ny, num.nx), dtype=np.float32)
    num.S_n = np.zeros((num.ny, num.nx), dtype=np.float32)
    num.q_evap_old.fill(0.0)
    
    a = np.zeros((num.nz, num.ny, num.nx), dtype=np.float32)
    a[0,0,0] = phys.T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz)
    
    # LIMIT TO 200 STEPS
    nsteps = 200 
    
    start = time.perf_counter()
    
    for step in range(nsteps+1):
        t = step * num.dt
        a, T_top, P_laser, n_iter = time_step(a, t, phys, num, geom)
        if step % 50 == 0:
            print(f"Step {step}/{nsteps} | t={t:.6e}s | Peak T: {np.max(T_top):.2f} K")

    total_time = time.perf_counter() - start
    print(f"\nTotal Wall Time: {total_time:.3f}s")
    
    print("\n=== TIMING BREAKDOWN (Accumulated) ===")
    print(f"Total Step Time:      {TIMERS['total_step']:.4f} s")
    print(f"  > Flux Calc:        {TIMERS['q_calc']:.4f} s")
    print(f"  > Forward DCT:      {TIMERS['dct_forward']:.4f} s")
    print(f"  > Spectral Update:  {TIMERS['spectral_update']:.4f} s")
    print(f"  > Q Evap Physics:   {TIMERS['q_evap']:.4f} s")
    print(f"  > Reconstruct (SUM):{TIMERS['reconstruct_sum']:.4f} s  <-- Summation along Z")
    print(f"  > Reconstruct (IDCT):{TIMERS['reconstruct_idct']:.4f} s <-- Inverse 2D DCT")
    print(f"  > Convergence Loop: {TIMERS['convergence_loop']:.4f} s (Includes inner ops)")
    
    return a

# ============================================================
#   MAIN
# ============================================================

if __name__ == "__main__":
    phys = PhysParams(rho=7850, Cp=500, k=15, T0=293.0, P=200.0, Absorptivity=0.30, r_b=6e-5, x0=0.0, y0=0.0025, vx=0.8)
    num = NumericalParams(dt=6e-6, t_final=0.012, nx=512, ny=256, nz=1000)
    geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, num=num, phys=phys)

    run_simulation(phys, num, geom)
    print("Done.")