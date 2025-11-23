import numpy as np
from scipy.fft import dctn
import cupy as cp
import matplotlib.pyplot as plt

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

        self.K = None   # CPU arrays
        self.KK = None


# ============================================================
#   COSINE NORMALIZATION COEFFICIENTS
# ============================================================

def C_coef_cpu(N, L):
    C = np.sqrt(2.0 / L) * np.ones(N)
    C[0] = np.sqrt(1.0 / L)
    return C


# ============================================================
#   LASER HEAT FLUX (GPU)
# ============================================================

def q_laser(x, y, t, phys):
    x0t = phys.x0 + phys.vx * t
    rb = phys.r_b
    return (phys.Absorptivity * 2 * phys.P / (np.pi * rb**2)) \
            * cp.exp(-2 * ((x - x0t)**2 + (y - phys.y0)**2) / rb**2)


# ============================================================
#   GPU → CPU DCT-II (2D)
# ============================================================

def DCT_II(q_gpu):
    q_cpu = cp.asnumpy(q_gpu)             # (ny, nx)
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

def forcing_S(q_gpu, phys, num, geom):
    nx, ny, nz = num.nx, num.ny, num.nz
    dx, dy = geom.Lx / nx, geom.Ly / ny

    q_dct = DCT_II(q_gpu)    # CPU array (ny, nx)

    Cm = C_coef_cpu(nx, geom.Lx)
    Cn = C_coef_cpu(ny, geom.Ly)
    Cp = C_coef_cpu(nz, geom.Lz)

    # Broadcast to (nz, ny, nx)
    S = Cp[:, None, None] * Cn[None, :, None] * Cm[None, None, :]
    S *= (dx * dy / 4) * q_dct[None, :, :]

    return S


# ============================================================
#   TIME STEP (CPU)
# ============================================================

def time_step(a, t, phys, num, geom):
    q_gpu = q_laser(geom.X, geom.Y, t, phys)
    S = forcing_S(q_gpu, phys, num, geom)
    return num.K * a + num.KK * S


# ============================================================
#   RUN SIMULATION — CPU MODES, GPU 2D LASER
# ============================================================

def run_simulation(phys, num, geom):
    print("Precomputing K, KK ...")
    K, KK = precompute_K_KK(phys, num, geom)
    num.K = K
    num.KK = KK

    print("Allocating modal field a (CPU)...")
    a = np.zeros((num.nz, num.ny, num.nx), dtype=np.float64)

    # Put initial temperature in zero mode
    a[0,0,0] = phys.T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz)

    t = 0.0
    nsteps = int(np.ceil(num.t_final / num.dt))

    for step in range(nsteps):
        print(f"t = {t:.6f}")
        a = time_step(a, t, phys, num, geom)
        t += num.dt

    return a


# ============================================================
#   RECONSTRUCT TEMPERATURE FIELD (GPU)
# ============================================================

def reconstruct_temperature_gpu(a_cpu, num, geom):
    # Send the modal coefficients to GPU
    a_gpu = cp.asarray(a_cpu)

    nx, ny, nz = num.nx, num.ny, num.nz

    # --- Z sum ---
    Cp_cpu = C_coef_cpu(nz, geom.Lz)
    Cp = cp.asarray(Cp_cpu)                     # CuPy
    a2d = cp.tensordot(Cp, a_gpu, axes=(0, 0))  # (ny, nx)

    # --- X,Y normalization (convert to GPU) ---
    Cm_cpu = C_coef_cpu(nx, geom.Lx)
    Cn_cpu = C_coef_cpu(ny, geom.Ly)
    Cm = cp.asarray(Cm_cpu)
    Cn = cp.asarray(Cn_cpu)

    # Grids
    x_vals = geom.X[0, :]
    y_vals = geom.Y[:, 0]

    # Build cosine matrices
    m_idx = cp.arange(nx)[:, None]
    n_idx = cp.arange(ny)[:, None]

    Bx = cp.cos(cp.pi * m_idx * x_vals[None, :] / geom.Lx)   # (nx, nx)
    By = cp.cos(cp.pi * n_idx * y_vals[None, :] / geom.Ly)   # (ny, ny)

    # Apply normalization
    A = (Cn[:, None] * a2d) * Cm[None, :]

    # final reconstruction
    T = By.T.dot(A).dot(Bx)
    return T


# ============================================================
#   MAIN SCRIPT
# ============================================================

phys = PhysParams(rho=7900, Ceff=500, k=14, T0=300.0,
                  P=200.0, Absorptivity=0.30, r_b=0.00006,
                  x0=0.001, y0=0.0025, vx=0.5)

num = NumericalParams(dt=0.0002, t_final=0.002,
                      nx=512, ny=256, nz=4000)

geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.005, params=num)
"""
a_final = run_simulation(phys, num, geom)
T_final = reconstruct_temperature_gpu(a_final, num, geom)

# CPU for plotting
X = geom.X.get()
Y = geom.Y.get()
T = T_final.get()

plt.figure(figsize=(10, 5))
plt.contourf(X * 1e3, Y * 1e3, T, levels=50, cmap='hot')
plt.colorbar()
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title("Final temperature field")
plt.gca().set_aspect("equal")
plt.show()
"""
# ---- PARAMETERS ----
phys = PhysParams(
    rho=7900, Ceff=500, k=14, T0=300.0,
    P=200.0, Absorptivity=0.30, r_b=0.00006,
    x0=0.001, y0=0.0025, vx=0.5
)

nz_values = [50, 100, 200, 400, 800, 1200, 2000]   # choose values you can afford
results_Tmax = []
results_width = []
results_length = []


def meltpool_metrics(T, phys, geom):
    """
    Returns: maxT, meltpool width (in meters), meltpool length (meters)
    """
    mask = T > phys.T_liquidus

    if not np.any(mask):
        return np.max(T), 0.0, 0.0

    # width: span in y direction
    y_mask = np.any(mask, axis=1)  # rows where melt exists
    y_coords = geom.Y.get()[:, 0]
    width = y_coords[y_mask].max() - y_coords[y_mask].min()

    # length: span in x direction
    x_mask = np.any(mask, axis=0)  # columns where melt exists
    x_coords = geom.X.get()[0, :]
    length = x_coords[x_mask].max() - x_coords[x_mask].min()

    return np.max(T), width, length


# ----------- LOOP OVER nz ----------------
for nz in nz_values:
    print(f"\n=== Running simulation for nz = {nz} ===")

    num = NumericalParams(
        dt=0.0002, t_final=0.002,
        nx=512, ny=256, nz=nz
    )
    geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.005, params=num)

    # --- simulation ---
    a_final  = run_simulation(phys, num, geom)

    # --- reconstruct ---
    T_final = reconstruct_temperature_gpu(a_final, num, geom).get()

    # compute metrics
    Tmax, W, L = meltpool_metrics(T_final, phys, geom)
    results_Tmax.append(Tmax)
    results_width.append(W * 1e3)   # convert to mm
    results_length.append(L * 1e3)

    print(f"Tmax = {Tmax:.1f} K")
    print(f"Width = {W*1e3:.3f} mm")
    print(f"Length = {L*1e3:.3f} mm")


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