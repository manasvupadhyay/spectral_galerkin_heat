import numpy as np
import cupy as cp
import cupyx.scipy.fft as fft
import time
import os

# Ensure output directory exists
OUT_DIR = ".out"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
#   CUPY KERNELS (GPU OPTIMIZED)
# ============================================================

# Fused kernel for spectral update: a_temp = aK + KK_by_Cp * B_scaled
# Handles broadcasting of 2D B_scaled (ny, nx) against 3D arrays (nz, ny, nx) implicitly via indexing logic if needed,
# but CuPy ElementwiseKernel handles broadcasting automatically if shapes align.
# Here B_scaled is (ny, nx) and others are (nz, ny, nx).
spectral_update_kernel = cp.ElementwiseKernel(
    'float32 aK, float32 KK_Cp, float32 B',
    'float32 a_temp',
    'a_temp = aK + KK_Cp * B',
    'spectral_update_kernel'
)

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
        # Calculate as standard float first, then cast to CuPy scalar
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
    # T is on GPU
    # Avoid division by zero or invalid math if T is low (though T should be > 0)
    # Using cp.maximum to ensure safety
    T_safe = cp.maximum(T, 100.0) 
    
    q = 0.82 * phys.DeltaH_LV * phys.Pa / cp.sqrt(2 * cp.pi * phys.R_v * T_safe) * \
        cp.exp((phys.DeltaH_LV / (phys.R_v * phys.T_boil)) * (1.0 - phys.T_boil / T_safe))
    
    # Apply cutoff
    q = cp.where(T < phys.T_liquidus, 0.0, q)
    return q.astype(cp.float32)

def shift_flux_along_x(field, shift, geom):
    # field is on GPU
    if field is None or abs(shift) < 1e-12:
        return field
    
    # 1D interpolation along X axis for each Y row.
    # Since grid is uniform, we can use map_coordinates or simple interpolation.
    # For speed on GPU, we can use simple index shifting if shift is integer multiple of dx,
    # but here shift is float. Linear interpolation is best.
    
    # Coordinate mapping: x_new = x_old - shift
    # We want value at x_grid points.
    # value at x_i comes from x_i + shift in the old field.
    
    # x indices to sample from:
    x_indices = (geom.X + shift) / geom.dx
    y_indices = geom.Y / geom.dy # Identity for Y
    
    # Use map_coordinates. Order 1 (linear) is sufficient and fast.
    # mode='constant', cval=0.0 assumes 0 flux outside domain.
    from cupyx.scipy.ndimage import map_coordinates
    shifted = map_coordinates(field, cp.stack([y_indices, x_indices]), order=1, mode='constant', cval=0.0)
    
    return shifted.astype(cp.float32)

def DCT_II(q):
    # q is (ny, nx)
    return fft.dctn(q, type=2, norm='ortho', axes=(0, 1)).astype(cp.float32)

def precompute_K_KK(phys, num, geom):
    # Compute on GPU
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
#   SOLVER
# ============================================================

def reconstruct_temperature_top(a, num, geom):
    # 1. Summation along Z (Spectral space reduction)
    # a is (nz, ny, nx), Cp32 is (nz). Broadcasting works.
    # Result A is (ny, nx)
    A = (geom.Cp32[:, None, None] * a).sum(axis=0)
    
    # 2. Inverse DCT (2D)
    # Type 3 is inverse of Type 2 (ortho)
    return (geom.recon_scale * fft.dctn(A, type=3, norm='ortho', axes=(0, 1))).astype(cp.float32)

def time_step(a, t, phys, num, geom, epsilon=2e+1):
    # All inputs are GPU arrays
    
    # 1. Fluxes
    q_las = q_laser(geom, t, phys)
    q_evap = shift_flux_along_x(num.q_evap_old, phys.vx * num.dt, geom)
    q_net = q_las - q_evap
    
    # 2. Forward DCT
    q_dct = DCT_II(q_net)
    
    # 3. Linear Step
    # aK = a * K
    cp.multiply(a, num.K, out=num.aK)
    
    # S_n = scale * q_dct
    S_n = geom.dct_scale * q_dct
    
    # B_scaled = scale * q_dct (Same as S_n initially)
    # a_temp = aK + KK_Cp * B_scaled
    # Use fused kernel
    spectral_update_kernel(num.aK, num.KK_by_Cp, S_n, num.a_temp)
    
    S_current = S_n  # Reference, not copy yet
    
    # 4. Non-linear Iteration (Evaporation)
    T_temp = reconstruct_temperature_top(num.a_temp, num, geom)
    
    omega = 0.1
    k = 0
    for k in range(30):
        q_evap_new = q_evap_point(T_temp, phys)
        
        # q_diff = q_las - q_evap_new
        cp.subtract(q_las, q_evap_new, out=num.q_diff)
        
        # S_target = scale * DCT(q_diff)
        S_target = geom.dct_scale * DCT_II(num.q_diff)
        
        # Relaxation: S_current = omega * S_target + (1-omega) * S_current
        S_current = omega * S_target + (1.0 - omega) * S_current
        
        # Update a_temp
        spectral_update_kernel(num.aK, num.KK_by_Cp, S_current, num.a_temp)
        
        T_old = T_temp
        T_temp = reconstruct_temperature_top(num.a_temp, num, geom)
        
        # Check convergence
        max_diff = cp.max(cp.abs(T_temp - T_old))
        if max_diff < epsilon:
            break
    
    # Update state
    num.S_n = S_current # Keep for next step if needed (ETD2)
    num.q_evap_old = q_evap_new if k > 0 else q_evap # Update evaporation field
    
    # Calculate power (scalar CPU)
    P_laser = float(cp.sum(q_las)) * geom.dx * geom.dy
    
    return num.a_temp, T_temp, P_laser

def run_simulation(phys, num, geom):
    print(f"Initializing GPU simulation...")
    
    # Precompute matrices
    K, KK = precompute_K_KK(phys, num, geom)
    num.K = K
    num.KK = KK
    num.KK_by_Cp = (num.KK * geom.Cp[:, None, None]).astype(cp.float32)
    
    # Allocate GPU buffers
    num.q_diff = cp.empty((num.ny, num.nx), dtype=cp.float32)
    num.a_temp = cp.empty((num.nz, num.ny, num.nx), dtype=cp.float32)
    num.aK = cp.empty((num.nz, num.ny, num.nx), dtype=cp.float32)
    
    # Initial State
    a = cp.zeros((num.nz, num.ny, num.nx), dtype=cp.float32)
    # Set T0 (DC component)
    # a[0,0,0] corresponds to the mean temperature component scaled
    # T(x) = sum(a * phi) -> T0 = a000 * phi000 * ...
    # Correct initialization for uniform T0:
    a[0,0,0] = phys.T0 * cp.sqrt(geom.Lx * geom.Ly * geom.Lz)
    
    nsteps = int(np.ceil(num.t_final / num.dt))
    
    # History lists (CPU)
    T_top_history = []
    P_laser_history = []
    
    start_time = time.perf_counter()
    
    # Time Loop
    for step in range(nsteps + 1):
        t = step * num.dt
        
        a, T_top_gpu, P_laser = time_step(a, t, phys, num, geom)
        
        if step % 100 == 0:
            peak_T = float(cp.max(T_top_gpu))
            print(f"Step {step}/{nsteps} | t={t:.6e}s | Peak T: {peak_T:.2f} K")
        
        # Transfer surface T to CPU for history
        T_top_history.append(cp.asnumpy(T_top_gpu))
        P_laser_history.append(P_laser)
        
    total_time = time.perf_counter() - start_time
    print(f"Total time: {total_time:.3f}s")
    
    return a, T_top_history, P_laser_history

def reconstruct_temperature_xz(a, num, geom, phys, y0=None):
    # Reconstruct vertical slice on GPU, then transfer
    if y0 is None: y0 = phys.y0
    
    n_idx = cp.arange(num.ny, dtype=cp.float32)
    m_idx = cp.arange(num.nx, dtype=cp.float32)
    p_idx = cp.arange(num.nz, dtype=cp.float32)
    
    # Cosine basis at y0
    cos_n_y0 = geom.Cn * cp.cos(cp.pi * n_idx * y0 / geom.Ly)
    
    # Sum over n (y-modes) -> (nz, nx)
    # a is (nz, ny, nx)
    # modal_pm[p, m] = sum_n (a[p, n, m] * cos_n_y0[n]) * Cm[m]
    # Broadcasting: a * cos_n_y0[None, :, None] -> sum axis 1
    modal_pm = (a * cos_n_y0[None, :, None]).sum(axis=1) * geom.Cm[None, :]
    
    # Reconstruct x (IDCT-like or direct sum)
    # We want T(x, z) at grid points.
    # Using matrix multiplication for reconstruction along lines is efficient if N is not huge.
    # T(x, z) = sum_p sum_m modal_pm[p, m] * cos(m*pi*x/Lx) * cos(p*pi*z/Lz) * Cp[p]
    
    # x-basis: (nx, nx) matrix
    cos_m_x = cp.cos(cp.pi * m_idx[:, None] * cp.asarray(geom.x_cpu)[None, :] / geom.Lx)
    
    # temp_p_x[p, x] = sum_m modal_pm[p, m] * cos_m_x[m, x]
    temp_p_x = modal_pm @ cos_m_x
    
    # Multiply by Cp
    temp_p_x = temp_p_x * geom.Cp[:, None]
    
    # z-basis: (nz, nz)
    cos_p_z = cp.cos(cp.pi * p_idx[:, None] * cp.asarray(geom.z_cpu)[None, :] / geom.Lz)
    
    # T_xz[z, x] = sum_p cos_p_z[p, z] * temp_p_x[p, x]
    # Transpose cos_p_z to (nz_grid, p_modes) -> (nz, nz) symmetric anyway
    T_xz = cos_p_z.T @ temp_p_x
    
    return cp.asnumpy(T_xz)

def save_temp_profiles(a, num, geom, phys):
    # Reconstruct final surface on GPU
    T_top_gpu = reconstruct_temperature_top(a, num, geom)
    T_top = cp.asnumpy(T_top_gpu)
    
    # Find hotspot
    iy_max, ix_max = np.unravel_index(np.argmax(T_top), T_top.shape)
    x_center = geom.x_cpu[ix_max]
    
    # Reconstruct XZ slice
    T_xz = reconstruct_temperature_xz(a, num, geom, phys, y0=phys.y0)
    
    # Find z-profile at x_center
    ix_xz = int(np.argmin(np.abs(geom.x_cpu - x_center)))
    
    iy = int(np.argmin(np.abs(geom.y_cpu - phys.y0)))
    
    # Save
    for direction, coords, profile in [
        ('x', geom.x_cpu, T_top[iy, :]),
        ('y', geom.y_cpu, T_top[:, ix_max]),
        ('z', geom.z_cpu, T_xz[:, ix_xz])
    ]:
        fname = f"{OUT_DIR}/{direction}_spectral_latent_heat.txt"
        np.savetxt(fname, np.vstack([coords, profile]).T, header=f'{direction}(m) T(K)', fmt='% .6e')
        print(f"Saved: {fname}")

# ============================================================
#   MAIN
# ============================================================

if __name__ == "__main__":
    # Parameters
    phys = PhysParams(rho=7850, Cp=500, k=15, T0=293.0, P=200.0, Absorptivity=0.30, r_b=6e-5, x0=0.0, y0=0.0025, vx=0.8)
    num = NumericalParams(dt=6e-6, t_final=0.012, nx=512, ny=256, nz=1000)
    geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, num=num, phys=phys)

    # Run
    a_final, T_top_history, P_laser_history = run_simulation(phys, num, geom)
    
    # Save
    save_temp_profiles(a_final, num, geom, phys)
    
    np.savetxt(f"{OUT_DIR}/laser_power_history.csv", 
               np.column_stack([np.arange(len(P_laser_history)) * num.dt, P_laser_history]),
               header='time(s) P_laser(W)', delimiter=',', fmt='%.6e')
    print("Done.")


