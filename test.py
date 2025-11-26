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


#make .out directory if not exists
OUT_DIR = ".out"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
#  GEOMETRY PARAMETERS (GPU only for 2D fields)
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
        self.dct_scale = np.float32((self.dx * self.dy / 4.0) * np.sqrt((self.nx * self.ny) / (self.Lx * self.Ly)))
        self.recon_scale = np.float32(np.sqrt(self.nx * self.ny) / np.sqrt(self.Lx * self.Ly))
        
        # Laser coefficient base
        self.laser_coef = phys.Absorptivity * 2.0 * phys.P / (np.pi * phys.r_b ** 2)

        # Numerical checks
        lambda_max_z = (phys.k / (phys.rho * phys.Ceff)) * (np.pi * num.nz / self.Lz) ** 2
        print(f"dx={self.dx:.3e}, dy={self.dy:.3e}, r_b={phys.r_b:.3e}, v_res_x={self.dx/num.dt:.3e}, vx={phys.vx}, lambda_max_z*dt={lambda_max_z*num.dt:.3e}")

# ============================================================
#   MATERIAL & LASER PARAMETERS
# ============================================================

class PhysParams:
    def __init__(self, rho, Ceff, k,
                 P, Absorptivity, r_b, x0, y0, vx,
                 DeltaH_LV=7.41e6, R_v=150.0, T0=300.0,
                 T_boil=3090.0, T_liquidus=1800.0, T_solidus=1700.0):

        self.rho = rho
        self.Ceff = Ceff
        self.k = k

        self.P = P
        self.Absorptivity = Absorptivity
        self.r_b = r_b
        self.x0 = x0
        self.y0 = y0
        self.vx = vx

        self.DeltaH_LV = DeltaH_LV
        self.R_v = R_v
        self.T0 = T0
        self.T_boil = T_boil
        self.T_liquidus = T_liquidus
        self.T_solidus = T_solidus


# ============================================================
#   NUMERICAL PARAMETERS
# ============================================================

class NumericalParams:
    def __init__(self, dt, t_final, nx, ny, nz, debug=False):
        self.dt = dt
        self.t_final = t_final
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.debug = debug

        self.K = None   # CPU arrays (nz, ny, nx)
        self.KK = None
        self.KK_by_Cp = None
        self.q_diff = None
        self.B_buffer = None
        self.a_temp = None
        self.aK = None


# ============================================================
#   COSINE NORMALIZATION COEFFICIENTS
# ============================================================

def C_coef(N, L):
    C = np.sqrt(2.0 / L) * np.ones(N)
    C[0] = np.sqrt(1.0 / L)
    return C


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
    q = 0.82 * phys.DeltaH_LV / np.sqrt(2 * np.pi * phys.R_v * T) * \
        np.exp((phys.DeltaH_LV / (phys.R_v * phys.T_boil)) * (1.0 - phys.T_boil / T))
    q[T < phys.T_boil] = 0.0
    return q.astype(np.float32)

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

    return K.astype(np.float32), KK.astype(np.float32)


# ============================================================
#   TIME STEP 
# ============================================================

def time_step(a, t, phys, num, geom, epsilon=1e-0):
    q_las = q_laser(geom, t, phys)
    q_dct = DCT_II(q_las)
    
    np.multiply(a, num.K, out=num.aK, casting='same_kind')
    np.multiply(geom.dct_scale, q_dct, out=num.B_buffer, casting='same_kind')
    compute_a_temp_numba(num.aK, num.KK_by_Cp, num.B_buffer, num.a_temp)
    
    T_temp = reconstruct_temperature_top(num.a_temp, num, geom)
    for k in range(20):
        np.subtract(q_las, q_evap_point(T_temp, phys), out=num.q_diff, casting='same_kind')
        np.multiply(geom.dct_scale, DCT_II(num.q_diff), out=num.B_buffer, casting='same_kind')
        compute_a_temp_numba(num.aK, num.KK_by_Cp, num.B_buffer, num.a_temp)
        T_old = T_temp
        T_temp = reconstruct_temperature_top(num.a_temp, num, geom)
        if np.max(np.abs(T_temp - T_old)) < epsilon:
            break
    return num.a_temp, T_temp


# ============================================================
#   RUN SIMULATION
# ============================================================

def run_simulation(phys, num, geom):
    print("Precomputing K, KK, buffers...")
    num.K, num.KK = precompute_K_KK(phys, num, geom)
    num.KK_by_Cp = (num.KK * geom.Cp[:, None, None]).astype(np.float32)
    num.q_diff = np.empty((num.ny, num.nx), dtype=np.float32)
    num.B_buffer = np.empty((num.ny, num.nx), dtype=np.float32)
    num.a_temp = np.empty((num.nz, num.ny, num.nx), dtype=np.float32)
    num.aK = np.empty((num.nz, num.ny, num.nx), dtype=np.float32)
    
    a = np.zeros((num.nz, num.ny, num.nx), dtype=np.float32)
    a[0,0,0] = phys.T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz)
    
    nsteps = int(np.ceil(num.t_final / num.dt))
    T_top_history = []
    start = time.perf_counter()
    
    for step in range(nsteps):
        t = step * num.dt
        print(f"Step {step+1}/{nsteps} | t={t:.6e}s")
        a, T_top = time_step(a, t, phys, num, geom)
        T_top_history.append(T_top)
    
    elapsed = time.perf_counter() - start
    print(f"\nTotal: {elapsed:.3f}s ({nsteps} steps, {elapsed/nsteps:.3f}s/step)")
    return a, T_top_history


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
    """Save 1D temperature profiles through the last laser position."""
    t = num.t_final if t is None else t
    x_center = phys.x0 + phys.vx * t
    
    T_top = reconstruct_temperature_top(a, num, geom)
    x_vals, y_vals = geom.X[0, :], geom.Y[:, 0]
    ix = int(np.argmin(np.abs(x_vals - x_center)))
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
#   SENSITIVITY ANALYSIS OVER nz
# ============================================================
"""
phys = PhysParams(rho=7900, Ceff=500, k=14, T0=300.0,
                  P=200.0, Absorptivity=0.30, r_b=0.00006,
                  x0=0.001, y0=0.0025, vx=0.5)

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
    num = NumericalParams(dt=0.0002, t_final=0.002, nx=512, ny=256, nz=nz, debug=True)
    geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.005, params=num)

    a_final = run_simulation(phys, num, geom)
    T_final = reconstruct_temperature_top(a_final, num, geom)  # (ny, nx)

    Tmax, W, L = meltpool_metrics(T_final, phys, geom)
    results_Tmax.append(Tmax)
    results_width.append(W * 1e3)   # mm
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
#   MAIN SCRIPT
# ============================================================

phys = PhysParams(rho=7900, Ceff=500, k=14, T0=300.0,
                  P=200.0, Absorptivity=0.30, r_b=6e-5,
                  x0=0.005, y0=0.0025, vx=0.8)

num = NumericalParams(dt=6e-6, t_final=0.012,
                      nx=512, ny=256, nz=1000)

geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, num=num, phys=phys)

a_final, T_top_history = run_simulation(phys, num, geom)
save_temp_profiles(a_final, num, geom, phys)
T_final = T_top_history[-1]

# Probing a point on top surface
# chose x probe so that temperature max is reached on a chosen time step
x_probe = 30 * phys.vx * num.dt + phys.x0 # = 100 * 0.8 * 0.0000075 = 0.0006
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





