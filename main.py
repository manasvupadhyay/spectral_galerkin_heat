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
    def __init__(self, Lx, Ly, Lz, num, phys):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        # GPU 2D grid (ny, nx)
        x = cp.linspace(0, Lx, num.nx, endpoint=False)
        y = cp.linspace(0, Ly, num.ny, endpoint=False)
        self.X, self.Y = cp.meshgrid(x, y, indexing="xy")

        # A few numerical checks to ensure convergence
        # Laser stop should be resolved in x and y 
        dx = Lx / num.nx
        dy = Ly / num.ny

        rb = phys.r_b
        vx = phys.vx
        k_phys = phys.k
        rho_phys = phys.rho
        Ceff_phys = phys.Ceff

        v_res_x = dx / num.dt
        lambda_max_z = (k_phys / (rho_phys * Ceff_phys)) * (np.pi * num.nz / Lz)**2

        print(
            f"Numerical checks -> dx={dx:.3e}, dy={dy:.3e}, r_b={rb}, v_res_x={v_res_x:.3e}, "
            f"phys.vx={vx}, lambda_max_z={lambda_max_z}, lambda_max_dt(wanted >2)={(lambda_max_z * num.dt) if lambda_max_z is not None else None}"
        )

        # Now assert with safety guards (phys may not be defined during import-time checks)
        assert dx < rb / 3, "dx too large for laser spot size"
        assert dy < rb / 3, "dy too large for laser spot size"

        # Check speed resolution
        assert vx < v_res_x, "Laser speed too high for dx resolution"

        # Check good convergence in z direction (lambda * Delta t < 2)
        assert lambda_max_z * num.dt > 2.0, "Time step too large for z resolution"

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
    T_temp = to_cpu(reconstruct_temperature_top(a_temp, num, geom))
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
        T_temp = to_cpu(reconstruct_temperature_top(a_temp, num, geom))
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

def reconstruct_temperature_top(a, num, geom):
    """Reconstruct the top surface temperature (y,x) from modal coefficients `a`.

    Returns an array of shape (ny, nx). Works with CuPy or NumPy (via fallback).
    """
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


def reconstruct_temperature_xz(a, num, geom, phys, y0=None, mode='full_field'):
    """Reconstruct an x-z slice of the temperature at a given y position.
    Not optimized, as we call it only for analysis.

    Returns (x_vals, z_vals, T_xz) where T_xz has shape (nz, nx_selected)
    and x_vals, z_vals are numpy arrays in meters.
    """
    if y0 is None:
        y0 = phys.y0

    nx, ny, nz = num.nx, num.ny, num.nz

    # Precompute normalization and basis functions
    Cp = cp.asarray(C_coef(nz, geom.Lz))
    Cm = cp.asarray(C_coef(nx, geom.Lx))
    Cn = cp.asarray(C_coef(ny, geom.Ly))

    # discrete modal indices
    n_idx = cp.arange(ny)
    m_idx = cp.arange(nx)

    # evaluate cos(n*pi*y0/Ly) * Cn[n]
    cos_n_y0 = Cn * cp.cos(cp.pi * n_idx * (y0) / geom.Ly)   # shape (ny,)

    x_vals = to_cpu(geom.X[0, :])
    z_vals = np.linspace(0.0, geom.Lz, nz)

    a_gpu = cp.asarray(a)  # (nz, ny, nx)

    # For each p (z-mode) compute temperature along x at y=y0
    T_xz = []
    for p in range(nz):
        a_p = a_gpu[p, :, :]  # (ny, nx)
        # sum over n: S_m = sum_n Cn[n]*cos(n*pi*y0/Ly) * a_p[n,m]
        S_m = (cos_n_y0[:, None] * a_p).sum(axis=0)   # (nx,)
        # multiply by Cm and project on x basis
        x_cos = cp.cos(cp.pi * m_idx[:, None] * geom.X[0, :][None, :] / geom.Lx)  # (nx, nx)
        T_p_x = Cp[p] * (Cm[:, None] * S_m[:, None]).sum(axis=0)  # crude: will compute below
        # Alternative: perform explicit formula: T_p_x = Cp[p] * sum_m (Cm[m]*S_m[m]*cos(m*pi*x/Lx))
        T_p_x = Cp[p] * (Cm * S_m) @ x_cos  # (nx,)
        T_xz.append(T_p_x)

    T_xz = cp.stack(T_xz, axis=0)  # (nz, nx)

    # Convert to CPU numpy
    T_xz = to_cpu(T_xz)

    # If meltpool mode, crop x-range around laser spot
    if mode == 'meltpool':
        # Reconstruct top temperature to estimate meltpool length
        T_top = to_cpu(reconstruct_temperature_top(a, num, geom))
        mask = T_top > phys.T_liquidus
        x_coords = to_cpu(geom.X[0, :])
        if np.any(mask):
            x_mask = np.any(mask, axis=0)
            x_min_mp = x_coords[x_mask].min()
            x_max_mp = x_coords[x_mask].max()
            mp_length = x_max_mp - x_min_mp
        else:
            # fallback small length
            mp_length = geom.Lx * 0.1

        half_span = 1.5 * mp_length / 2.0
        x_center = phys.x0
        x_min = max(0.0, x_center - half_span)
        x_max = min(geom.Lx, x_center + half_span)

        # select indices
        ix_min = int(np.searchsorted(x_coords, x_min))
        ix_max = int(np.searchsorted(x_coords, x_max))
        if ix_min == ix_max:
            ix_min = max(0, ix_min - 1)
            ix_max = min(nx, ix_max + 1)

        x_sel = x_coords[ix_min:ix_max]
        T_xz = T_xz[:, ix_min:ix_max]
        return x_sel, z_vals, T_xz

    # full field
    return x_vals, z_vals, T_xz


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

num = NumericalParams(dt=0.0000075, t_final=0.005,
                      nx=512, ny=256, nz=1000)

geom = GeomParams(Lx=0.01, Ly=0.005, Lz=0.0025, num=num, phys=phys)

a_final, T_top_history = run_simulation(phys, num, geom)

T_final = T_top_history[-1]
# Probing a point on top surface
# chose x probe so that temperature max is reached on a chosen time step
x_probe = 100 * phys.vx * num.dt # = 100 * 0.8 * 0.0000075 = 0.0006
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
                   origin='lower', cmap='hot')
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


