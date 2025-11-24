import numpy as np
from scipy.fft import dctn
import matplotlib.pyplot as plt

# Try to use CuPy (GPU). If unavailable, fall back to NumPy (CPU).
try:
    import cupy as cp
    USE_CUPY = True
except Exception:
    import numpy as cp
    USE_CUPY = False

# Compatibility helpers: convert device arrays to CPU numpy arrays.
if USE_CUPY:
    asnumpy = cp.asnumpy
    def to_cpu(x):
        try:
            return x.get()
        except Exception:
            return cp.asnumpy(x)
else:
    def asnumpy(x):
        return x
    def to_cpu(x):
        return x

# ============================================================
#  GEOMETRY PARAMETERS (GPU only for 2D fields)
# ============================================================

class GeomParams:
    def __init__(self, Lx, Ly, Lz, params):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        # GPU 2D grid (ny, nx)
        x = cp.linspace(0, Lx, params.nx, endpoint=False)
        y = cp.linspace(0, Ly, params.ny, endpoint=False)
        self.X, self.Y = cp.meshgrid(x, y, indexing="xy")


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

def q_laser(X, Y, t, phys):
    x0t = phys.x0 + phys.vx * t
    rb = phys.r_b
    return (phys.Absorptivity * 2 * phys.P / (np.pi * rb**2)) \
            * cp.exp(-2 * ((X - x0t)**2 + (Y - phys.y0)**2) / rb**2)


def q_evap_point(T: np.ndarray, phys: PhysParams) -> np.ndarray:
    Lv, Rv, Tb = phys.DeltaH_LV, phys.R_v, phys.T_boil
    #T_safe = np.maximum(T, 1.0)
    q = 0.82 * Lv / np.sqrt(2 * np.pi * Rv * T) * np.exp((Lv / (Rv * Tb)) * (1.0 - Tb / T))
    q[T < Tb] = 0.0
    return q

# ============================================================
#   DCT-II 2D (CPU)
# ============================================================

def DCT_II(q):
    q_cpu = asnumpy(q)             # shape (ny, nx)
    return dctn(q_cpu, type=2, norm='backward', workers=-1)


# ============================================================
#   PRECOMPUTE 3D K / KK (CPU)
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
#   BUILD 3D SOURCE TERM S (CPU)
# ============================================================

def forcing_S(q, phys, num, geom):
    nx, ny, nz = num.nx, num.ny, num.nz
    dx, dy = geom.Lx / nx, geom.Ly / ny

    q_dct = DCT_II(q)    # shape (ny, nx)

    Cm = C_coef(nx, geom.Lx)
    Cn = C_coef(ny, geom.Ly)
    Cp = C_coef(nz, geom.Lz)

    # Broadcast to shape (nz, ny, nx)
    S = Cp[:, None, None] * Cn[None, :, None] * Cm[None, None, :]
    S *= (dx * dy / 16) * q_dct[None, :, :]

    return S


# ============================================================
#   TIME STEP (CPU)
# ============================================================

def time_step(a, t, phys, num, geom):
    q = q_laser(geom.X, geom.Y, t, phys)  # GPU array (ny, nx)
    S = forcing_S(q, phys, num, geom)     # CPU array (nz, ny, nx)
    return num.K * a + num.KK * S          # CPU operation

def time_step(a, t, phys, num, geom, epsilon=1e-2, debug=False):
    n_iter_max = 20
    # GPU laser field
    q_las = q_laser(geom.X, geom.Y, t, phys)         # (ny, nx)
    # Initial modal source term (DCT)
    S = forcing_S(q_las, phys, num, geom)        # CPU array (nz, ny, nx)
    aK = a * num.K
    a_temp = aK + num.KK * S  # first shot with laser   
    T_temp = to_cpu(reconstruct_temperature(a_temp, num, geom))
    if debug:
        print("Iterating")
    for k in range(n_iter_max):
        q_evap = q_evap_point(T_temp, phys)          # CPU (ny, nx)
        # Update DCT source term with evaporation
        q_diff = to_cpu(q_las) - q_evap
        S = forcing_S(cp.asarray(q_diff), phys, num, geom)
        # Reconstruct 2D temperature on top
        a_temp = aK + num.KK * S 
        T_temp_old = T_temp
        T_temp = to_cpu(reconstruct_temperature(a_temp, num, geom))
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
    T_top_history = []

    print("Allocating modal field a (CPU)...")
    a = np.zeros((num.nz, num.ny, num.nx), dtype=np.float64)  # (nz, ny, nx)
    a[0,0,0] = phys.T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz)

    t = 0.0
    nsteps = int(np.ceil(num.t_final / num.dt))
    for step in range(nsteps):
        print(f"t = {t:.6f}")
        a, T_top = time_step(a, t, phys, num, geom, debug=num.debug)
        t += num.dt
        T_top_history.append(T_top)
    return a, T_top_history


# ============================================================
#   RECONSTRUCT TEMPERATURE FIELD (GPU)
# ============================================================

def reconstruct_temperature(a, num, geom):
    a_gpu = cp.asarray(a)  # Send modal coefficients to GPU

    nx, ny, nz = num.nx, num.ny, num.nz

    # --- Z sum ---
    Cp = cp.asarray(C_coef(nz, geom.Lz))           # shape (nz,)
    a2d = cp.tensordot(Cp, a_gpu, axes=(0, 0))     # shape (ny, nx)

    # --- X,Y normalization ---
    Cm = cp.asarray(C_coef(nx, geom.Lx))           # shape (nx,)
    Cn = cp.asarray(C_coef(ny, geom.Ly))           # shape (ny,)

    x_vals = geom.X[0, :]
    y_vals = geom.Y[:, 0]

    Bx = cp.cos(cp.pi * cp.arange(nx)[:, None] * x_vals[None, :] / geom.Lx)  # (nx, nx)
    By = cp.cos(cp.pi * cp.arange(ny)[:, None] * y_vals[None, :] / geom.Ly)  # (ny, ny)

    A = (Cn[:, None] * a2d) * Cm[None, :]
    T = By.T.dot(A).dot(Bx)
    return T


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
    y_coords = geom.Y.get()[:, 0]
    width = y_coords[y_mask].max() - y_coords[y_mask].min()

    # length along x
    x_mask = np.any(mask, axis=0)
    x_coords = geom.X.get()[0, :]
    length = x_coords[x_mask].max() - x_coords[x_mask].min()

    return np.max(T), width, length


for nz in nz_values:
    print(f"\n=== Running simulation for nz = {nz} ===")
    num = NumericalParams(dt=0.0002, t_final=0.002, nx=512, ny=256, nz=nz, debug=True)
    geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.005, params=num)

    a_final = run_simulation(phys, num, geom)
    T_final = reconstruct_temperature(a_final, num, geom).get()  # (ny, nx)

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
                  P=200.0, Absorptivity=0.30, r_b=0.00006,
                  x0=0.001, y0=0.0025, vx=0.8)

num = NumericalParams(dt=0.00001, t_final=0.005,
                      nx=512, ny=256, nz=1000)

geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, params=num)

a_final, T_top_history = run_simulation(phys, num, geom)
T_final = T_top_history[-1]
# Probing a point on top surface
# chose x probe so that temperature max is reached on a chosen time step
x_probe = 20 * phys.vx * num.dt # = 20 * 0.8 * 0.00005 = 0.0008
y_probe = 0.0025
ix = int(x_probe / geom.Lx * num.nx)
iy = int(y_probe / geom.Ly * num.ny)
T_probe = [T_top[iy, ix] for T_top in T_top_history]

# CPU for plotting
X = to_cpu(geom.X)
Y = to_cpu(geom.Y)
T = to_cpu(T_final)

# Compute q_laser and q_evap for final time
q_las = to_cpu(q_laser(geom.X, geom.Y, num.t_final, phys))  # CPU array
q_eva = q_evap_point(T, phys)                             # CPU array

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])  # 2 rows, 2 cols, top bigger

# --- Temperature (top, spanning both columns) ---
ax0 = fig.add_subplot(gs[0, :])
im0 = ax0.contourf(X*1e3, Y*1e3, T, levels=50, cmap='hot')
fig.colorbar(im0, ax=ax0, label='Temperature (K)')
ax0.set_title("Final Temperature Field")
ax0.set_xlabel("x (mm)")
ax0.set_ylabel("y (mm)")
ax0.set_aspect('equal')

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


