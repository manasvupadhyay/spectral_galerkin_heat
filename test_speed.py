import numpy as np
import pyfftw
import matplotlib.pyplot as plt
import time
from numba import njit, prange
import os
import helpers as hp

pyfftw.config.NUM_THREADS = os.cpu_count()
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
#  NUMBA KERNELS
# ============================================================

@njit(parallel=True, fastmath=True)
def compute_a_temp_numba(aK, KK_by_Cp, B_scaled, a_temp_out):
    """Update spectral coefficients for ETD1 scheme."""
    nz = aK.shape[0]
    for p in prange(nz):
        a_temp_out[p, :, :] = aK[p, :, :] + KK_by_Cp[p, :, :] * B_scaled

@njit(parallel=True, fastmath=True)
def compute_a_temp_ETD2(a_phi0, KK_phi1, KKK_phi2, S_prev, S_next, a_temp_out):
    """Update spectral coefficients for ETD2 scheme."""
    nz = a_phi0.shape[0]
    for p in prange(nz):
        for i in range(a_phi0.shape[1]):
            for j in range(a_phi0.shape[2]):
                dS = S_next[i, j] - S_prev[i, j]
                a_temp_out[p, i, j] = a_phi0[p, i, j] + KK_phi1[p, i, j] * S_prev[i, j] + KKK_phi2[p, i, j] * dS

@njit(parallel=True, fastmath=True)
def add_arrays_numba(a, b):
    """Add two 3D arrays in place: a += b."""
    nz = a.shape[0]
    for p in prange(nz):
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                a[p, i, j] += b[p, i, j]

# ============================================================
#  CLASSES
# ============================================================

class Laser:
    def __init__(self, P, r_b, x0, y0, v, Absorptivity):
        self.P = P
        self.r_b = r_b
        self.x0, self.y0 = x0, y0
        self.v = np.array(v, dtype=np.float64)
        self.Absorptivity = Absorptivity
        self.x, self.y, self.t = x0, y0, 0
        
    def update(self, dt):
        """Update laser position."""
        self.x += self.v[0] * dt
        self.y += self.v[1] * dt
        self.t += dt

class GeomParams:
    def __init__(self, Lx, Ly, Lz, num, phys, laser):
        self.Lx, self.Ly, self.Lz = float(Lx), float(Ly), float(Lz)
        self.nx, self.ny, self.nz = num.nx, num.ny, num.nz
        self.dx, self.dy, self.dz = Lx/num.nx, Ly/num.ny, Lz/num.nz

        # Global mesh coordinates
        x_np = np.linspace(0.0, self.Lx, self.nx, endpoint=False)
        y_np = np.linspace(0.0, self.Ly, self.ny, endpoint=False)
        z_np = np.linspace(0.0, self.Lz, self.nz)
        self.x, self.y, self.z = x_np.astype(np.float32), y_np.astype(np.float32), z_np.astype(np.float32)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        self.X, self.Y = self.X.astype(np.float32), self.Y.astype(np.float32)

        # Normalization coefficients
        self.Cm = hp.C_coef(self.nx, self.Lx)
        self.Cn = hp.C_coef(self.ny, self.Ly)
        self.Cp = hp.C_coef(self.nz, self.Lz)
        self.Cp32 = self.Cp.astype(np.float32)
        
        # Scaling factors for DCT/IDCT
        self.dct_scale = np.float32((self.dx * self.dy) * np.sqrt((self.nx * self.ny) / (self.Lx * self.Ly)))
        self.recon_scale = np.float32(np.sqrt(self.nx * self.ny) / np.sqrt(self.Lx * self.Ly))

        # Precomputed cosine bases for reconstruction
        self.cos_mx = np.cos(np.pi * np.arange(self.nx)[:, None] * x_np[None, :] / self.Lx).astype(np.float32)
        self.cos_ny = np.cos(np.pi * np.arange(self.ny)[:, None] * y_np[None, :] / self.Ly).astype(np.float32)
        self.cos_pz = np.cos(np.pi * np.arange(self.nz)[:, None] * z_np[None, :] / self.Lz).astype(np.float32)

        self.laser_coef = laser.Absorptivity * 2.0 * laser.P / (np.pi * laser.r_b ** 2)
        
        # Fine mesh setup for latent heat correction
        self.refinement = 3
        self.Lx_box, self.Ly_box, self.Lz_box = 0.6e-3, 0.25e-3, 0.1e-3
        self.dx_fine, self.dy_fine, self.dz_fine = self.dx/self.refinement, self.dy/self.refinement, self.dz/self.refinement
        
        self.nx_fine_total = int(np.ceil(self.Lx / self.dx_fine))
        self.ny_fine_total = int(np.ceil(self.Ly / self.dy_fine))
        self.nz_fine_total = int(np.ceil(self.Lz_box / self.dz_fine))
        
        x_fine = np.linspace(0.0, self.Lx, self.nx_fine_total, dtype=np.float32)
        y_fine = np.linspace(0.0, self.Ly, self.ny_fine_total, dtype=np.float32)
        z_fine = np.linspace(0.0, self.Lz_box, self.nz_fine_total, dtype=np.float32)
        
        print("Precomputing fine cosine bases...")
        m, n, p = np.arange(self.nx), np.arange(self.ny), np.arange(self.nz)
        self.Bx_fine_full = (self.Cm[:, None] * np.cos(np.pi * m[:, None] * x_fine[None, :] / self.Lx)).astype(np.float32)
        self.By_fine_full = (self.Cn[:, None] * np.cos(np.pi * n[:, None] * y_fine[None, :] / self.Ly)).astype(np.float32)
        self.Bz_fine_full = (self.Cp[:, None] * np.cos(np.pi * p[:, None] * z_fine[None, :] / self.Lz)).astype(np.float32)
        
        # Box dimensions in fine grid points
        self.nx_box = int(np.ceil(self.Lx_box / self.dx_fine))
        self.ny_box = int(np.ceil(self.Ly_box / self.dy_fine))
        self.nz_box = self.nz_fine_total
        
        # Preallocated arrays for fine mesh box
        self.Bx_fine = np.zeros((self.nx, self.nx_box), dtype=np.float32)
        self.By_fine = np.zeros((self.ny, self.ny_box), dtype=np.float32)
        self.Bz_fine = self.Bz_fine_full[:, :self.nz_box]
        self.box_x = np.zeros(self.nx_box, dtype=np.float32)
        self.box_y = np.zeros(self.ny_box, dtype=np.float32)
        self.box_z = np.linspace(0.0, self.Lz_box, self.nz_box, dtype=np.float32)
        self.fine_mesh_initialized = False
        self.check_resolution(phys, laser, num)

    def check_resolution(self, phys, laser, num):
        """Check if spatial and temporal resolutions are sufficient."""
        lambda_max_z = (phys.k / (phys.rho * phys.Ceff)) * (np.pi * num.nz / self.Lz) ** 2
        dx_rb, dy_rb = self.dx / laser.r_b, self.dy / laser.r_b
        v_mag = np.linalg.norm(laser.v)
        v_crit = v_mag / (10 * self.dx / num.dt) if v_mag > 0 else 0
        
        print(f"Resolution: dx/rb={dx_rb:.2f}, dy/rb={dy_rb:.2f}, v_crit={v_crit:.2f}")
        if dx_rb > 0.4 or dy_rb > 0.4: print("WARNING: Spatial resolution insufficient!")
        if v_crit > 1.0: print("WARNING: Laser moves too fast for time step!")

    def update_fine_mesh(self, laser):
        """Update fine mesh box position to center on laser."""
        x_min, x_max = laser.x - self.Lx_box/2, laser.x + self.Lx_box/2
        y_min, y_max = laser.y - self.Ly_box/2, laser.y + self.Ly_box/2
        
        self.box_x[:] = np.linspace(x_min, x_max, self.nx_box, dtype=np.float32)
        self.box_y[:] = np.linspace(y_min, y_max, self.ny_box, dtype=np.float32)
        
        # Calculate indices for slicing precomputed bases
        ix_start, iy_start = int(x_min / self.dx_fine), int(y_min / self.dy_fine)
        ix_s_c = max(0, min(ix_start, self.nx_fine_total - 1))
        iy_s_c = max(0, min(iy_start, self.ny_fine_total - 1))
        ix_e_c = max(0, min(ix_start + self.nx_box, self.nx_fine_total))
        iy_e_c = max(0, min(iy_start + self.ny_box, self.ny_fine_total))
        
        # Offsets for filling the box arrays
        ix_off_s, iy_off_s = max(0, -ix_start), max(0, -iy_start)
        ix_off_e = self.nx_box - max(0, (ix_start + self.nx_box) - self.nx_fine_total)
        iy_off_e = self.ny_box - max(0, (iy_start + self.ny_box) - self.ny_fine_total)
        
        # Fill valid portions from precomputed bases
        self.Bx_fine.fill(0.0); self.By_fine.fill(0.0)
        if ix_e_c > ix_s_c: self.Bx_fine[:, ix_off_s:ix_off_e] = self.Bx_fine_full[:, ix_s_c:ix_e_c]
        if iy_e_c > iy_s_c: self.By_fine[:, iy_off_s:iy_off_e] = self.By_fine_full[:, iy_s_c:iy_e_c]
        
        self.dV_fine = self.dx_fine * self.dy_fine * self.dz_fine
        self.fine_mesh_initialized = True

class PhysParams:
    def __init__(self, rho, Cp, k, L_f=267700.0, DeltaH_LV=7.41e6, R_v=150.774, T0=293.0, 
                 Pa=101325.0, T_boil=3090.0, T_liquidus=1800.0, T_solidus=1700.0):
        self.rho, self.Cp, self.k = rho, Cp, k
        self.L_f, self.Ceff = L_f, Cp
        self.DeltaH_LV, self.R_v, self.T0, self.Pa = DeltaH_LV, R_v, T0, Pa
        self.T_boil, self.T_liquidus, self.T_solidus = T_boil, T_liquidus, T_solidus
        print(f"Material: Cp={Cp}, L_f={L_f}, T_S={T_solidus}, T_L={T_liquidus}")

class NumericalParams:
    def __init__(self, dt, t_final, nx, ny, nz, ETD='ETD1'):
        self.dt, self.t_final = dt, t_final
        self.nx, self.ny, self.nz = nx, ny, nz
        self.ETD, self.iter = ETD, 0
        self.T_corr_prev, self.T_corr_buffer = None, None

    def prepare_K_buffers(self, phys, geom):
        """Precompute spectral propagators and allocate buffers."""
        print(f"Precomputing K, KK... ({self.ETD})")
        self.K, self.KK, self.K_phi0, self.KK_phi1, self.KKK_phi2 = precompute_K_KK(phys, self, geom)
        self.KK_by_Cp = (self.KK * geom.Cp[:, None, None]).astype(np.float32)
        
        # Allocate working arrays
        self.q_diff = np.empty((self.ny, self.nx), dtype=np.float32)
        self.B_buffer = np.empty((self.ny, self.nx), dtype=np.float32)
        self.a_temp = np.empty((self.nz, self.ny, self.nx), dtype=np.float32)
        self.aK = np.empty((self.nz, self.ny, self.nx), dtype=np.float32)
        self.S_n = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.q_evap_old = np.zeros((self.ny, self.nx), dtype=np.float32)

# ============================================================
#   FUNCTIONS
# ============================================================

def precompute_K_KK(phys, num, geom):
    """Compute spectral propagators (K, KK) for heat equation."""
    m, n, p = np.arange(num.nx)[None, None, :], np.arange(num.ny)[None, :, None], np.arange(num.nz)[:, None, None]
    mu = (m * np.pi / geom.Lx)**2 + (n * np.pi / geom.Ly)**2 + (p * np.pi / geom.Lz)**2
    lambda_j = (phys.k / (phys.rho * phys.Ceff)) * mu

    K = np.exp(-lambda_j * num.dt)
    K[0, 0, 0] = 1.0
    KK = np.zeros_like(K)
    mask = lambda_j > 0
    KK[mask] = (1 - K[mask]) / (phys.rho * phys.Ceff * lambda_j[mask])
    KK[~mask] = num.dt / (phys.rho * phys.Ceff) # Limit for lambda -> 0

    K_phi0, KK_phi1, KKK_phi2 = None, None, None
    if num.ETD == 'ETD2':
        z = -lambda_j * num.dt
        phi_0, phi_1, phi_2 = hp.phi_functions(z)
        dt_rhoCp = num.dt / (phys.rho * phys.Ceff)
        Cp = hp.C_coef(num.nz, geom.Lz)[:, None, None]
        K_phi0 = phi_0.astype(np.float32)
        KK_phi1 = (dt_rhoCp * phi_1 * Cp).astype(np.float32)
        KKK_phi2 = (dt_rhoCp * phi_2 * Cp).astype(np.float32)
    
    return K.astype(np.float32), KK.astype(np.float32), K_phi0, KK_phi1, KKK_phi2

def apply_latent_heat(T_box_prev, T_box_base, num, phys, geom, laser, timers=None, epsilon=2e+1, max_iter=30):
    """Iteratively apply latent heat correction using line source method."""
    nx, ny_box, nz_box = T_box_base.shape
    ix_mid = nx // 2
    beta = 0.2 # Under-relaxation factor
    box_coords = (geom.box_x, geom.box_y, geom.box_z)

    # Initialize with previous correction if available
    T_box_current = (T_box_base + num.T_corr_prev.astype(np.float32)) if num.T_corr_prev is not None else T_box_base.copy()
    if num.T_corr_buffer is None or num.T_corr_buffer.shape != T_box_base.shape:
        num.T_corr_buffer = np.zeros_like(T_box_base, dtype=np.float64)
    
    for n_iter in range(1, max_iter + 1):
        # Identify melt pool boundaries
        M_yz = T_box_current[ix_mid, :, :]
        mask_full = M_yz >= phys.T_liquidus
        mask_partial = (M_yz > phys.T_solidus) & (M_yz < phys.T_liquidus)
        
        # Compute correction field
        t0 = time.perf_counter()
        isotherm_data = hp.find_isotherms_along_x(T_box_current, M_yz, mask_full, mask_partial, phys)
        if timers is not None: timers['lh_find_isotherms'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        num.T_corr_buffer.fill(0.0)
        hp.compute_latent_heat_correction(num.T_corr_buffer, box_coords, isotherm_data, phys, laser, geom)
        if timers is not None: timers['lh_compute_correction'] += time.perf_counter() - t0
        
        # Update with under-relaxation
        T_box_new = T_box_base + num.T_corr_buffer.astype(np.float32)
        max_change = beta * np.max(np.abs(T_box_new - T_box_current))
        T_box_current = beta * T_box_new + (1 - beta) * T_box_current
        
        if max_change < epsilon: break
    
    t0 = time.perf_counter()
    delta_a = hp.T_corr_to_modes(num.T_corr_buffer, geom)
    if timers is not None: timers['lh_modes_conversion'] += time.perf_counter() - t0

    return T_box_current, n_iter, delta_a, num.T_corr_buffer.copy()

def time_step(a, phys, num, geom, laser, timers=None, epsilon=2e+1, iter_step=0):
    """Perform one time step of the simulation."""
    if num.ETD == 'ETD1':
        # 1. Compute source terms (Laser + Evaporation)
        t0 = time.perf_counter()
        q_las = hp.q_laser(geom, laser)
        q_evap = hp.shift_flux(num.q_evap_old, (laser.v[0]*num.dt, laser.v[1]*num.dt), geom)
        q_dct = hp.DCT_II(q_las - q_evap)
        if timers is not None: timers['source'] += time.perf_counter() - t0
        
        # 2. Linear step (ETD1)
        t0 = time.perf_counter()
        np.multiply(a, num.K, out=num.aK, casting='same_kind')
        S_n = geom.dct_scale * q_dct
        np.multiply(geom.dct_scale, q_dct, out=num.B_buffer, casting='same_kind')
        compute_a_temp_numba(num.aK, num.KK_by_Cp, num.B_buffer, num.a_temp)
        S_current = S_n.copy()
        if timers is not None: timers['linear'] += time.perf_counter() - t0
        
        # 3. Nonlinear iteration for evaporation
        t0 = time.perf_counter()
        T_temp = hp.reconstruct_temperature_top(num.a_temp, num, geom)
        for k in range(30):
            q_evap = hp.q_evap_point(T_temp, phys)
            np.subtract(q_las, q_evap, out=num.q_diff, casting='same_kind')
            S_target = geom.dct_scale * hp.DCT_II(num.q_diff)
            S_current = 0.1 * S_target + 0.9 * S_current # Relaxation
            np.multiply(1.0, S_current, out=num.B_buffer, casting='same_kind')
            compute_a_temp_numba(num.aK, num.KK_by_Cp, num.B_buffer, num.a_temp)
            T_old = T_temp
            T_temp = hp.reconstruct_temperature_top(num.a_temp, num, geom)
            if np.max(np.abs(T_temp - T_old)) < epsilon: break
        
        num.S_n, num.q_evap_old = S_n.copy(), q_evap.astype(np.float32, copy=True)
        P_laser = np.sum(q_las) * geom.dx * geom.dy
        if timers is not None: timers['nonlinear'] += time.perf_counter() - t0

        # 4. Latent Heat Correction
        t0 = time.perf_counter()
        
        t_mesh = time.perf_counter()
        geom.update_fine_mesh(laser)
        if timers is not None: timers['lh_update_mesh'] += time.perf_counter() - t_mesh
        
        t_box = time.perf_counter()
        T_box_target, _ = hp.reconstruct_temperature_box(num.a_temp, num, geom)
        if timers is not None: timers['lh_reconstruct_box'] += time.perf_counter() - t_box

        T_box_corrected, n_iter_LH, delta_a, T_corr_final = apply_latent_heat(
            T_box_target, T_box_target, num, phys, geom, laser, timers=timers, epsilon=epsilon)
        
        num.T_corr_prev = T_corr_final
        
        t_add = time.perf_counter()
        add_arrays_numba(num.a_temp, delta_a) # Apply correction to modes
        if timers is not None: timers['lh_add_delta'] += time.perf_counter() - t_add
        
        if timers is not None: timers['latent_heat'] += time.perf_counter() - t0
        
        # Debug output
        t0 = time.perf_counter()
        if iter_step % 200 == 0:
            hp.T_to_HDF5(f"{OUT_DIR}/T_box_step_{laser.t:.5f}", T_box_target.transpose(2, 1, 0), (geom.box_x, geom.box_y, geom.box_z), geom=geom)
            hp.T_to_HDF5(f"{OUT_DIR}/T_corr_step_{laser.t:.5f}", T_corr_final.transpose(2, 1, 0), (geom.box_x, geom.box_y, geom.box_z), geom=geom)
        if timers is not None: timers['io'] += time.perf_counter() - t0
        
        return num.a_temp, T_temp, P_laser, k+1, n_iter_LH
    
    elif num.ETD == 'ETD2':
        # Simplified ETD2 implementation (no under-relaxation)
        t0 = time.perf_counter()
        q_las = hp.q_laser(geom, laser)
        S_n_next = geom.dct_scale * hp.DCT_II(q_las)
        if timers is not None: timers['source'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        np.multiply(a, num.K_phi0, out=num.aK, casting='same_kind')
        compute_a_temp_ETD2(num.aK, num.KK_phi1, num.KKK_phi2, num.S_n, S_n_next, num.a_temp)
        if timers is not None: timers['linear'] += time.perf_counter() - t0
        
        t0 = time.perf_counter()
        T_temp = hp.reconstruct_temperature_top(num.a_temp, num, geom)
        for k in range(20):
            q_evap = hp.q_evap_point(T_temp, phys)
            S_n_next = geom.dct_scale * hp.DCT_II(q_las - q_evap)
            compute_a_temp_ETD2(num.aK, num.KK_phi1, num.KKK_phi2, num.S_n, S_n_next, num.a_temp)
            T_old = T_temp
            T_temp = hp.reconstruct_temperature_top(num.a_temp, num, geom)
            if np.max(np.abs(T_temp - T_old)) < epsilon: break
        if timers is not None: timers['nonlinear'] += time.perf_counter() - t0
        
        num.S_n = S_n_next.copy()
        return num.a_temp, T_temp, np.sum(q_las)*geom.dx*geom.dy, k+1, 0

def run_simulation(phys, num, geom, laser):
    """Main simulation loop."""
    num.prepare_K_buffers(phys, geom)
    a = np.zeros((num.nz, num.ny, num.nx), dtype=np.float32)
    a[0,0,0] = phys.T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz) # Initial condition (mean T)
    
    nsteps = int(np.ceil(num.t_final / num.dt))
    T_top_hist, P_laser_hist = [], []
    
    timers = {
        'source': 0.0, 'linear': 0.0, 'nonlinear': 0.0, 'latent_heat': 0.0, 'io': 0.0,
        'lh_reconstruct_box': 0.0, 'lh_find_isotherms': 0.0, 'lh_compute_correction': 0.0, 'lh_modes_conversion': 0.0,
        'lh_update_mesh': 0.0, 'lh_add_delta': 0.0,
        'total': 0.0
    }
    
    start = time.perf_counter()
    
    for step in range(nsteps+1):
        t_step_start = time.perf_counter()
        a, T_top, P_laser, n_evap, n_LH = time_step(a, phys, num, geom, laser, timers=timers, iter_step=step)
        laser.update(num.dt)
        timers['total'] += time.perf_counter() - t_step_start
        print(f"Step {step}/{nsteps} | t={step*num.dt:.6e}s | Peak T: {np.max(T_top):.2f} K | P: {P_laser:.3f} W | Evap: {n_evap} | LH: {n_LH}")
        T_top_hist.append(T_top)
        P_laser_hist.append(P_laser)

    total_time = time.perf_counter() - start
    
    print(f"\nTotal: {total_time:.3f}s")
    print("=== Timing Recap ===")
    for k, v in timers.items():
        print(f"{k:<20}: {v:.4f} s ({v/timers['total']*100:.1f}%)")
        
    return a, T_top_hist, P_laser_hist

# ============================================================
#   MAIN
# ============================================================

if __name__ == "__main__":
    # Simulation parameters
    laser = Laser(P=200.0, r_b=6e-5, x0=0.00, y0=0.0025, v=(0.8, 0), Absorptivity=0.30)
    phys = PhysParams(rho=7850, Cp=500, k=15, T0=293.0)
    # Reduced dt for stability and t_final for quick profiling
    num = NumericalParams(dt=6e-6, t_final=0.012, nx=512, ny=256, nz=1000, ETD='ETD1')
    geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, num=num, phys=phys, laser=laser)

    # Run
    a_final, T_top_history, P_laser_history = run_simulation(phys, num, geom, laser)

    # Post-processing
    hp.save_temp_profiles(a_final, num, geom, phys, laser)
    np.savetxt(f"{OUT_DIR}/laser_power_history.csv", 
               np.column_stack([np.arange(len(P_laser_history)) * num.dt, P_laser_history]),
               header='time(s) P_laser(W)', delimiter=',', fmt='%.6e')

    # Probe temperature at a specific point
    x_probe, y_probe = 30 * laser.v[0] * num.dt + laser.x0, 30 * laser.v[1] * num.dt + laser.y0
    ix, iy = int(x_probe / geom.Lx * num.nx), int(y_probe / geom.Ly * num.ny)
    T_probe = [T[iy, ix] for T in T_top_history]
    np.savetxt("T_probe.csv", np.array(T_probe), delimiter=",")

    # Visualization
    T = T_top_history[-1]
    q_las, q_eva = hp.q_laser(geom, laser), hp.q_evap_point(T, phys)
    x_vals, z_vals, T_xz = hp.reconstruct_temperature_xz(a_final, num, geom, phys, laser, mode='meltpool')

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.contourf(geom.X*1e3, geom.Y*1e3, T, levels=50, cmap='hot')
    fig.colorbar(im0, ax=ax0, label='T (K)')
    ax0.set_title("Final Temperature Field")

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(T_xz, aspect='auto', extent=[x_vals[0]*1e3, x_vals[-1]*1e3, z_vals[0]*1e3, z_vals[-1]*1e3], origin='upper', cmap='hot')
    fig.colorbar(im1, ax=ax1, label='T (K)')
    ax1.set_title(f"XZ Slice (y={laser.y:.3f}m)")

    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.contourf(geom.X*1e3, geom.Y*1e3, q_las, levels=50, cmap='inferno')
    fig.colorbar(im2, ax=ax2, label='q_laser')
    ax2.set_title("Laser Flux")

    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.contourf(geom.X*1e3, geom.Y*1e3, q_eva, levels=50, cmap='inferno')
    fig.colorbar(im3, ax=ax3, label='q_evap')
    ax3.set_title("Evaporative Flux")

    plt.tight_layout()
    plt.show()
