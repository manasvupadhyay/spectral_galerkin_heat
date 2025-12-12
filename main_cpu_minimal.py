import numpy as np
import pyfftw
from pyfftw.interfaces.scipy_fft import dctn
import time
from numba import njit, prange
import os

pyfftw.config.NUM_THREADS = os.cpu_count()
OUT_DIR = ".out"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
#   NUMBA KERNELS
# ============================================================

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
    def __init__(self, dt, t_final, nx, ny, nz, ETD='ETD1'):
        self.dt, self.t_final = dt, t_final
        self.nx, self.ny, self.nz = nx, ny, nz
        self.ETD = ETD
        self.K = self.KK = self.KK_by_Cp = None
        self.K_phi0 = self.KK_phi1 = self.KKK_phi2 = None
        self.q_diff = self.B_buffer = self.a_temp = self.aK = self.S_n = None
        self.q_evap_old = np.zeros((ny, nx), dtype=np.float32)

# ============================================================
#   HELPER FUNCTIONS
# ============================================================

def C_coef(N, L):
    C = np.sqrt(2.0 / L) * np.ones(N)
    C[0] = np.sqrt(1.0 / L)
    return C

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

    if num.ETD == 'ETD2':
        z = -lambda_j * num.dt
        phi_0, phi_1, phi_2 = phi_functions(z)
        dt_over_rhoCp = num.dt / (phys.rho * phys.Ceff)
        Cp = C_coef(num.nz, geom.Lz)[:, None, None]
        K_phi0 = phi_0.astype(np.float32)
        KK_phi1 = (dt_over_rhoCp * phi_1 * Cp).astype(np.float32)
        KKK_phi2 = (dt_over_rhoCp * phi_2 * Cp).astype(np.float32)
        return K.astype(np.float32), KK.astype(np.float32), K_phi0, KK_phi1, KKK_phi2
    return K.astype(np.float32), KK.astype(np.float32), None, None, None

# ============================================================
#   SOLVER
# ============================================================

def time_step(a, t, phys, num, geom, epsilon=2e+1):
    if num.ETD == 'ETD1':
        q_las = q_laser(geom, t, phys)
        q_evap = shift_flux_along_x(num.q_evap_old, phys.vx * num.dt, geom)
        q_dct = DCT_II(q_las - q_evap)
        
        np.multiply(a, num.K, out=num.aK, casting='same_kind')
        S_n = geom.dct_scale * q_dct
        omega = 0.1
        
        np.multiply(geom.dct_scale, q_dct, out=num.B_buffer, casting='same_kind')
        compute_a_temp_numba(num.aK, num.KK_by_Cp, num.B_buffer, num.a_temp)
        S_current = S_n.copy()
        
        T_temp = reconstruct_temperature_top(num.a_temp, num, geom)
        k = 0
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


def run_simulation(phys, num, geom):
    print(f"Initializing {num.ETD} simulation...")
    K, KK, K_phi0, KK_phi1, KKK_phi2 = precompute_K_KK(phys, num, geom)
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
    
    nsteps = int(np.ceil(num.t_final / num.dt))
    T_top_history, P_laser_history = [], []
    start = time.perf_counter()
    
    for step in range(nsteps+1):
        t = step * num.dt
        a, T_top, P_laser, n_iter = time_step(a, t, phys, num, geom)
        print(f"Step {step}/{nsteps} | t={t:.6e}s | Peak T: {np.max(T_top):.2f} K")
        T_top_history.append(T_top)
        P_laser_history.append(P_laser)

    print(f"Total time: {time.perf_counter() - start:.3f}s")
    return a, T_top_history, P_laser_history

def reconstruct_temperature_top(a, num, geom):
    A = (geom.Cp32[:, None, None] * a).sum(axis=0)
    return (geom.recon_scale * dctn(A, type=3, norm='ortho', axes=(0, 1), workers=-1)).astype(np.float32, copy=False)

def reconstruct_temperature_xz(a, num, geom, phys, y0=None):
    if y0 is None: y0 = phys.y0
    n_idx, m_idx, p_idx = np.arange(num.ny), np.arange(num.nx), np.arange(num.nz)
    cos_n_y0 = geom.Cn * np.cos(np.pi * n_idx * y0 / geom.Ly)
    modal_pm = (a * cos_n_y0[None, :, None]).sum(axis=1) * geom.Cm[None, :]
    cos_m_x = np.cos(np.pi * m_idx[:, None] * geom.x[None, :] / geom.Lx)
    temp_p_x = modal_pm @ cos_m_x * geom.Cp[:, None]
    cos_p_z = np.cos(np.pi * p_idx[:, None] * geom.z[None, :] / geom.Lz)
    return geom.x, geom.z, (cos_p_z.T @ temp_p_x).astype(np.float32)

def save_temp_profiles(a, num, geom, phys):
    T_top = reconstruct_temperature_top(a, num, geom)
    iy_max, ix_max = np.unravel_index(np.argmax(T_top), T_top.shape)
    x_center = geom.x[ix_max]
    x_xz, z_vals, T_xz = reconstruct_temperature_xz(a, num, geom, phys, y0=phys.y0)
    ix_xz = int(np.argmin(np.abs(x_xz - x_center)))
    
    iy = int(np.argmin(np.abs(geom.y - phys.y0)))
    
    for direction, coords, profile in [
        ('x', geom.x, T_top[iy, :]),
        ('y', geom.y, T_top[:, ix_max]),
        ('z', z_vals, T_xz[:, ix_xz])
    ]:
        fname = f"{OUT_DIR}/{direction}_spectral_latent_heat.txt"
        np.savetxt(fname, np.vstack([coords, profile]).T, header=f'{direction}(m) T(K)', fmt='% .6e')
        print(f"Saved: {fname}")

# ============================================================
#   MAIN
# ============================================================

if __name__ == "__main__":
    phys = PhysParams(rho=7850, Cp=500, k=15, T0=293.0, P=200.0, Absorptivity=0.30, r_b=6e-5, x0=0.0, y0=0.0025, vx=0.8)
    num = NumericalParams(dt=6e-6, t_final=0.012, nx=512, ny=256, nz=1000, ETD='ETD1')
    geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, num=num, phys=phys)

    a_final, T_top_history, P_laser_history = run_simulation(phys, num, geom)
    save_temp_profiles(a_final, num, geom, phys)
    
    np.savetxt(f"{OUT_DIR}/laser_power_history.csv", 
               np.column_stack([np.arange(len(P_laser_history)) * num.dt, P_laser_history]),
               header='time(s) P_laser(W)', delimiter=',', fmt='%.6e')
    print("Done.")