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

        # Precompute normalization coefficients (NumPy)
        self.Cm = C_coef(self.nx, self.Lx)
        self.Cn = C_coef(self.ny, self.Ly)
        self.Cp = C_coef(self.nz, self.Lz)


        # Precompute small separable coefficients and outer products
        # Avoid allocating large dense basis matrices (Bx, By, Bz) to save memory.
        m = np.arange(self.nx)
        n = np.arange(self.ny)
        p = np.arange(self.nz)
        # Keep 1D coefficient vectors and the outer product CnCm used repeatedly
        self.Bx = None
        self.By = None
        self.Bz = None
        self.CnCm = np.outer(self.Cn, self.Cm)   # (ny, nx)


        # Informational numerical checks
        dx = self.Lx / self.nx
        dy = self.Ly / self.ny
        rb = phys.r_b
        v_res_x = dx / num.dt
        lambda_max_z = (phys.k / (phys.rho * phys.Ceff)) * (np.pi * num.nz / self.Lz) ** 2
        print(f"Numerical checks -> dx={dx:.3e}, dy={dy:.3e}, r_b={rb}, v_res_x={v_res_x:.3e}, phys.vx={phys.vx}, lambda_max_z*dt={lambda_max_z*num.dt:.3e}")

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
    """Return q_laser as a NumPy array (CPU-only).

    Evaluates the Gaussian laser heat flux on the precomputed NumPy meshgrid
    `geom.X, geom.Y` and returns a NumPy array.
    """
    x0t = phys.x0 + phys.vx * t
    rb = phys.r_b
    coeff = phys.Absorptivity * 2.0 * phys.P / (np.pi * rb ** 2)
    X = geom.X
    Y = geom.Y
    flux = coeff * np.exp(-2.0 * ((X - x0t) ** 2 + (Y - phys.y0) ** 2) / (rb ** 2))
    return flux.astype(np.float32)


def q_evap_point(T: np.ndarray, phys: PhysParams) -> np.ndarray:
    Lv, Rv, Tb = phys.DeltaH_LV, phys.R_v, phys.T_boil
    #T_safe = np.maximum(T, 1.0)
    q = 0.82 * Lv / np.sqrt(2 * np.pi * Rv * T) * np.exp((Lv / (Rv * Tb)) * (1.0 - Tb / T))
    q[T < Tb] = 0.0
    return q.astype(np.float32)

# ============================================================
#   DCT-II 2D 
# ============================================================

def DCT_II(q, geom=None):
    """2D DCT-II on surface q.
    """
    return dctn(q, type=2, norm='ortho', workers=-1).astype(np.float32, copy=False)


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

    return K, KK


# ============================================================
#   TIME STEP 
# ============================================================

def time_step(a, t, phys, num, geom, epsilon=1e-0, debug=False):
    n_iter_max = 20
    q_las = q_laser(geom, t, phys)

    # Initial modal source term (compute separable S cheaply, do not allocate full 3D)
    dx = geom.dx
    dy = geom.dy
    q_dct = DCT_II(q_las, geom=geom)

    scale = (dx * dy / 4.0) *np.sqrt((geom.nx * geom.ny) / (geom.Lx * geom.Ly))

    aK = num.aK
    np.multiply(a, num.K, out=aK, casting='same_kind')
    a_temp = num.a_temp
    # use precomputed KK_by_Cp for faster updates
    KK_by_Cp = num.KK_by_Cp
    np.multiply(scale, q_dct, out=num.B_buffer, casting='same_kind')
    compute_a_temp_numba(aK, KK_by_Cp, num.B_buffer, a_temp)

    T_temp = reconstruct_temperature_top(a_temp, num, geom)
    if debug:
        print("Iterating")
    for k in range(n_iter_max):
        q_evap = q_evap_point(T_temp, phys)          # (ny, nx)

        # data-transfer / diff (all NumPy now)
        q_diff = num.q_diff
        np.subtract(q_las, q_evap, out=q_diff, casting='same_kind')

        # forcing for iteration: compute q_diff DCT and update modal field without full S
        qd = DCT_II(q_diff, geom=geom)
        np.multiply(scale, qd, out=num.B_buffer, casting='same_kind')
        # compute a_temp in parallel using precomputed KK_by_Cp
        compute_a_temp_numba(aK, num.KK_by_Cp, num.B_buffer, a_temp)
        T_temp_old = T_temp
        T_temp = reconstruct_temperature_top(a_temp, num, geom)
        if debug:
            print("iteration : ", k)
            print("Tmax = ",np.max(np.abs(T_temp)))
        if np.max(np.abs(T_temp - T_temp_old)) < epsilon:
            break
        T_temp_old = T_temp
    return a_temp, T_temp


# ============================================================
#   RUN SIMULATION
# ============================================================

def run_simulation(phys, num, geom):
    print("Precomputing K, KK ...")
    K, KK = precompute_K_KK(phys, num, geom)
    num.K = K
    num.KK = KK
    # Precompute KK multiplied by Cp to reduce per-step work: KK_by_Cp[p,:,:] = KK[p,:,:] * Cp[p]
    num.KK_by_Cp = KK * geom.Cp[:, None, None]
    num.q_diff = np.empty((num.ny, num.nx), dtype=np.float32)
    num.B_buffer = np.empty((num.ny, num.nx), dtype=np.float64)
    num.a_temp = np.empty((num.nz, num.ny, num.nx), dtype=np.float64)
    num.aK = np.empty((num.nz, num.ny, num.nx), dtype=np.float64)
    T_top_history = []

    print("Allocating modal field a ...")
    a = np.zeros((num.nz, num.ny, num.nx), dtype=np.float64)  # (nz, ny, nx)
    a[0,0,0] = phys.T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz)

    t = 0.0
    nsteps = int(np.ceil(num.t_final / num.dt))
    start = time.perf_counter()
    for step in range(nsteps):
        print(f"Step {step+1}/{nsteps} | Time: t={t:.6e}s")
        a, T_top = time_step(a, t, phys, num, geom, debug=num.debug)
        t += num.dt
        T_top_history.append(T_top)

    elapsed = time.perf_counter() - start
    print(f"Total elapsed real time: {elapsed:.3f} s over {nsteps} steps (avg {elapsed/max(1,nsteps):.3f} s/step)")
    return a, T_top_history


# ============================================================
#   RECONSTRUCT TEMPERATURE FIELD (GPU)
# ============================================================

def reconstruct_temperature_top(a, num, geom):
    """Reconstruct the top surface temperature (ny, nx) from modal coefficients a (nz, ny, nx).
    Always returns a NumPy array (for consistency with DCT and forcing).
    """
    # 1) weight by Cp and sum over p -> a2d (ny, nx)
    A = (geom.Cp[:, None, None] * a).sum(axis=0)
    scale_top = np.float32(np.sqrt(geom.nx * geom.ny) / np.sqrt(geom.Lx * geom.Ly))
    A32 = A.astype(np.float32)
    T = scale_top * dctn(A32, type=3, norm='ortho', axes=(0, 1), workers=-1)
    if False: 
        fname = os.path.join(OUT_DIR, f"reconstruct_top_debug_t.png")
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        X = geom.X * 1e3
        Y = geom.Y * 1e3
        c = ax.contourf(X, Y, T, levels=50, cmap='viridis')
        fig.colorbar(c, ax=ax, label='Temperature (K)')
        ax.set_title('Reconstructed Top Surface (debug)')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        plt.tight_layout()
        fig.savefig(fname, dpi=150)
        #plt.show()
        plt.close(fig)

    return T.astype(np.float32, copy=False)

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
        """Save 1D temperature profiles through the last laser position.
        Produces three files in the working directory:

        Arguments:
            - a: modal coefficients (nz, ny, nx)
            - num, geom, phys: parameter objects used throughout the module
            - t: optional time to compute last laser position; defaults to num.t_final
        """
        if t is None:
                t = num.t_final

        # last laser x position
        x_center = phys.x0 + phys.vx * t
        y_center = phys.y0

        # Top surface temperature (ny, nx)
        T_top = reconstruct_temperature_top(a, num, geom)
        x_vals = geom.X[0, :]
        y_vals = geom.Y[:, 0]

        # nearest indices
        ix = int(np.argmin(np.abs(x_vals - x_center)))
        iy = int(np.argmin(np.abs(y_vals - y_center)))

        # profiles on top surface
        x_profile = T_top[iy, :]
        y_profile = T_top[:, ix]

        # x-z slice at y_center (full field) and pick column at nearest x
        x_xz, z_vals, T_xz = reconstruct_temperature_xz(a, num, geom, phys, y0=y_center, mode='full_field')
        ix_xz = int(np.argmin(np.abs(x_xz - x_center)))
        z_profile = T_xz[:, ix_xz]

        # Save files: two columns each (coordinate, temperature)
        # save to .out folder
        out_dir = ".out"
        fname_x = f"{out_dir}/x_spectral_latent_heat.txt"
        fname_y = f"{out_dir}/y_spectral_latent_heat.txt"
        fname_z = f"{out_dir}/z_spectral_latent_heat.txt"

        np.savetxt(fname_x, np.vstack([x_vals, x_profile]).T,
                             header='x(m) T_top(K)', fmt='% .6e')
        np.savetxt(fname_y, np.vstack([y_vals, y_profile]).T,
                             header='y(m) T_top(K)', fmt='% .6e')
        np.savetxt(fname_z, np.vstack([z_vals, z_profile]).T,
                             header='z(m) T_xz(K)', fmt='% .6e')

        print(f"Saved: {fname_x}, {fname_y}, {fname_z}")
        return fname_x, fname_y, fname_z


def print_timings_summary():
    """Print a detailed timing summary from the TIMINGS aggregator."""
    if not TIMINGS:
        print("No timings recorded.")
        return
    total_all = sum(v['total'] for v in TIMINGS.values())
    print('\n==== Detailed timings summary ====>')
    print(f"Total tracked time: {total_all:.6f} s")
    print(f"{'Function':40s} {'Total(s)':>10s} {'%':>6s} {'Calls':>8s} {'Avg(s)':>10s}")
    print('-' * 80)
    for name, v in sorted(TIMINGS.items(), key=lambda kv: -kv[1]['total']):
        tot = v['total']
        cnt = v['count']
        pct = (tot / total_all * 100.0) if total_all > 0 else 0.0
        avg = tot / cnt if cnt else 0.0
        print(f"{name:40s} {tot:10.6f} {pct:6.2f}% {cnt:8d} {avg:10.6f}")
    print('==== End timings ====>\n')

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

num = NumericalParams(dt=6e-6, t_final=0.0006,
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





