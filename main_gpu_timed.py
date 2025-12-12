import numpy as np
import cupy as cp
import cupyx.scipy.fft as fft
import time
import os

# Ensure output directory exists
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
#   CLASSES
# ============================================================

class GeomParams:
    def __init__(self, Lx, Ly, Lz, num, phys):
        self.Lx, self.Ly, self.Lz = float(Lx), float(Ly), float(Lz)
        self.nx, self.ny, self.nz = num.nx, num.ny, num.nz
        self.dx, self.dy, self.dz = Lx / num.nx, Ly / num.ny, Lz / num.nz

        # CPU coordinates for saving
        self.x_cpu = np.linspace(0.0, self.Lx, self.nx, endpoint=False).astype(np.float32)
        self.y_cpu = np.linspace(0.0, self.Ly, self.ny, endpoint=False).astype(np.float32)
        self.z_cpu = np.linspace(0.0, self.Lz, self.nz).astype(np.float32)

        # GPU Meshgrid for flux calculation
        x_gpu = cp.asarray(self.x_cpu)
        y_gpu = cp.asarray(self.y_cpu)
        self.X, self.Y = cp.meshgrid(x_gpu, y_gpu, indexing='xy')
        
        # Precompute reconstruction coefficients on GPU
        self.Cm = cp.asarray(C_coef(self.nx, self.Lx))
        self.Cn = cp.asarray(C_coef(self.ny, self.Ly))
        self.Cp = cp.asarray(C_coef(self.nz, self.Lz))
        self.Cp32 = self.Cp.astype(cp.float32)
        
        # Scaling factors
        scale_val = (self.dx * self.dy) * np.sqrt((self.nx * self.ny) / (self.Lx * self.Ly))
        self.dct_scale = cp.float32(scale_val)
        
        recon_val = np.sqrt(self.nx * self.ny) / np.sqrt(self.Lx * self.Ly)
        self.recon_scale = cp.float32(recon_val)
        
        laser_val = phys.Absorptivity * 2.0 * phys.P / (np.pi * phys.r_b ** 2)
        self.laser_coef = cp.float32(laser_val)

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
        # Placeholders for GPU arrays
        self.K = None
        self.KK = None
        self.KK_by_Cp = None
        self.q_diff = None
        self.a_temp = None
        self.aK = None
        self.S_n = None
        self.q_evap_old = cp.zeros((ny, nx), dtype=cp.float32)

# ============================================================
#   HELPER FUNCTIONS
# ============================================================

def C_coef(N, L):
    C = np.sqrt(2.0 / L) * np.ones(N)
    C[0] = np.sqrt(1.0 / L)
    return C.astype(np.float32)

def q_laser(geom, t, phys):
    x0t = phys.x0 + phys.vx * t
    r_sq = (geom.X - x0t) ** 2 + (geom.Y - phys.y0) ** 2
    return (geom.laser_coef * cp.exp(-2.0 * r_sq / phys.r_b ** 2)).astype(cp.float32)

def q_evap_point(T, phys):
    T_safe = cp.maximum(T, 100.0) 
    q = 0.82 * phys.DeltaH_LV * phys.Pa / cp.sqrt(2 * cp.pi * phys.R_v * T_safe) * \
        cp.exp((phys.DeltaH_LV / (phys.R_v * phys.T_boil)) * (1.0 - phys.T_boil / T_safe))
    q = cp.where(T < phys.T_liquidus, 0.0, q)
    return q.astype(cp.float32)

def shift_flux_along_x(field, shift, geom):
    if field is None or abs(shift) < 1e-12:
        return field
    
    x_indices = (geom.X + shift) / geom.dx
    y_indices = geom.Y / geom.dy 
    
    from cupyx.scipy.ndimage import map_coordinates
    shifted = map_coordinates(field, cp.stack([y_indices, x_indices]), order=1, mode='constant', cval=0.0)
    return shifted.astype(cp.float32)

def DCT_II(q):
    return fft.dctn(q, type=2, norm='ortho', axes=(0, 1)).astype(cp.float32)

def precompute_K_KK(phys, num, geom):
    m = cp.arange(num.nx, dtype=cp.float32)[None, None, :]
    n = cp.arange(num.ny, dtype=cp.float32)[None, :, None]
    p = cp.arange(num.nz, dtype=cp.float32)[:, None, None]
    
    mu = (m * cp.pi / geom.Lx)**2 + (n * cp.pi / geom.Ly)**2 + (p * cp.pi / geom.Lz)**2
    lambda_j = (phys.k / (phys.rho * phys.Ceff)) * mu
    
    K = cp.exp(-lambda_j * num.dt)
    K[0, 0, 0] = 1.0
    
    KK = cp.zeros_like(K)
    mask = lambda_j > 1e-10
    
    KK[mask] = (1.0 - K[mask]) / (phys.rho * phys.Ceff * lambda_j[mask])
    KK[~mask] = num.dt / (phys.rho * phys.Ceff)
    
    return K.astype(cp.float32), KK.astype(cp.float32)

# ============================================================
#   TIMED FUNCTIONS
# ============================================================

def reconstruct_temperature_top_timed(a, num, geom):
    # 1. Summation along Z (Spectral space reduction)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    
    A = (geom.Cp32[:, None, None] * a).sum(axis=0)
    
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    TIMERS['reconstruct_sum'] += (t1 - t0)
    
    # 2. Inverse DCT (2D)
    t2 = time.perf_counter()
    
    res = (geom.recon_scale * fft.dctn(A, type=3, norm='ortho', axes=(0, 1))).astype(cp.float32)
    
    cp.cuda.Device().synchronize()
    t3 = time.perf_counter()
    TIMERS['reconstruct_idct'] += (t3 - t2)
    
    return res

def time_step(a, t, phys, num, geom, epsilon=2e+1):
    cp.cuda.Device().synchronize()
    t_start_step = time.perf_counter()
    
    # --- Flux Calculation ---
    t0 = time.perf_counter()
    q_las = q_laser(geom, t, phys)
    q_evap = shift_flux_along_x(num.q_evap_old, phys.vx * num.dt, geom)
    q_net = q_las - q_evap
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    TIMERS['q_calc'] += (t1 - t0)
    
    # --- Forward DCT ---
    t0 = time.perf_counter()
    q_dct = DCT_II(q_net)
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    TIMERS['dct_forward'] += (t1 - t0)
    
    # --- Spectral Update (Linear Part) ---
    t0 = time.perf_counter()
    cp.multiply(a, num.K, out=num.aK)
    S_n = geom.dct_scale * q_dct
    
    cp.multiply(num.KK_by_Cp, S_n, out=num.a_temp)
    cp.add(num.a_temp, num.aK, out=num.a_temp)
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    TIMERS['spectral_update'] += (t1 - t0)
    
    S_current = S_n
    
    # --- Initial Reconstruction ---
    T_temp = reconstruct_temperature_top_timed(num.a_temp, num, geom)
    
    # --- Non-linear Loop ---
    cp.cuda.Device().synchronize()
    t_loop_start = time.perf_counter()
    
    omega = 0.1
    k = 0
    for k in range(30):
        # Evaporation Physics
        t_evap0 = time.perf_counter()
        q_evap_new = q_evap_point(T_temp, phys)
        cp.cuda.Device().synchronize()
        t_evap1 = time.perf_counter()
        TIMERS['q_evap'] += (t_evap1 - t_evap0)
        
        # Update Source
        cp.subtract(q_las, q_evap_new, out=num.q_diff)
        
        # Forward DCT inside loop
        t_dct0 = time.perf_counter()
        S_target = geom.dct_scale * DCT_II(num.q_diff)
        cp.cuda.Device().synchronize()
        t_dct1 = time.perf_counter()
        TIMERS['dct_forward'] += (t_dct1 - t_dct0)
        
        S_current = omega * S_target + (1.0 - omega) * S_current
        
        # Spectral Update inside loop
        t_up0 = time.perf_counter()
        cp.multiply(num.KK_by_Cp, S_current, out=num.a_temp)
        cp.add(num.a_temp, num.aK, out=num.a_temp)
        cp.cuda.Device().synchronize()
        t_up1 = time.perf_counter()
        TIMERS['spectral_update'] += (t_up1 - t_up0)
        
        T_old = T_temp
        # Reconstruction inside loop
        T_temp = reconstruct_temperature_top_timed(num.a_temp, num, geom)
        
        if cp.max(cp.abs(T_temp - T_old)) < epsilon:
            break
            
    cp.cuda.Device().synchronize()
    t_loop_end = time.perf_counter()
    TIMERS['convergence_loop'] += (t_loop_end - t_loop_start)
    
    num.S_n = S_current
    num.q_evap_old = q_evap_new if k > 0 else q_evap
    P_laser = float(cp.sum(q_las)) * geom.dx * geom.dy
    
    cp.cuda.Device().synchronize()
    t_end_step = time.perf_counter()
    TIMERS['total_step'] += (t_end_step - t_start_step)
    
    return num.a_temp, T_temp, P_laser

def run_simulation(phys, num, geom):
    print(f"Initializing GPU simulation (Timed Run - 200 steps)...")
    
    K, KK = precompute_K_KK(phys, num, geom)
    num.K = K
    num.KK = KK
    num.KK_by_Cp = (num.KK * geom.Cp[:, None, None]).astype(cp.float32)
    
    num.q_diff = cp.empty((num.ny, num.nx), dtype=cp.float32)
    num.a_temp = cp.empty((num.nz, num.ny, num.nx), dtype=cp.float32)
    num.aK = cp.empty((num.nz, num.ny, num.nx), dtype=cp.float32)
    
    a = cp.zeros((num.nz, num.ny, num.nx), dtype=cp.float32)
    a[0,0,0] = phys.T0 * cp.sqrt(geom.Lx * geom.Ly * geom.Lz)
    
    # LIMIT TO 200 STEPS
    nsteps = 200
    
    # Warmup GPU
    print("Warming up GPU...")
    time_step(a, 0.0, phys, num, geom)
    cp.cuda.Device().synchronize()
    
    # Reset timers after warmup
    for key in TIMERS: TIMERS[key] = 0.0
    
    start = time.perf_counter()
    
    for step in range(nsteps + 1):
        t = step * num.dt
        a, T_top_gpu, P_laser = time_step(a, t, phys, num, geom)
        
        if step % 50 == 0:
            peak_T = float(cp.max(T_top_gpu))
            print(f"Step {step}/{nsteps} | t={t:.6e}s | Peak T: {peak_T:.2f} K")
        
    cp.cuda.Device().synchronize()
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

if __name__ == "__main__":
    phys = PhysParams(rho=7850, Cp=500, k=15, T0=293.0, P=200.0, Absorptivity=0.30, r_b=6e-5, x0=0.0, y0=0.0025, vx=0.8)
    num = NumericalParams(dt=6e-6, t_final=0.012, nx=512, ny=256, nz=1000)
    geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, num=num, phys=phys)

    run_simulation(phys, num, geom)
    print("Done.")