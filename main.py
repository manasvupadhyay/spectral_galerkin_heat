"""main.py

This script implements:
- Uses scipy.fft.dctn for 2D DCT projection of heat flux field
- Vectorized temperature reconstruction using tensor contractions

"""
# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, fftshift
import os
# Clean up old files
for filename in os.listdir('.'):
    if filename.startswith(('temperature_field_t', 'temperature_field_contour_t',
                            'q_laser_field_contour_t', 'q_evap_field_contour_t')):
        os.remove(filename)

class Params:
    """Container for physical and geometric parameters."""
    def __init__(self,
                # geometry
                Lx: float, Ly: float, Lz: float,
                # material properties
                rho: float, 
                Ceff: float, # J/K
                k: float, # W/(m K)
                T0: float, # thermal conductivity
                # laser
                P: float, Absorptivity: float, r_b: float, x0: float, y0: float, vx: float,
                # evaporation
                DeltaH_LV: float = 7.41e6, # J/kg Latent heat evaporation
                R_v: float = 150.0, # J/(kg K)
                T_boil: float = 3090.0, # K
                T_liquidus: float = 1800, 
                T_solidus: float = 1700,
                debug: bool = False
                ):
        self.Lx = Lx; self.Ly = Ly; self.Lz = Lz
        self.rho = rho; self.Ceff = Ceff; self.k = k; self.T0 = T0
        self.P = P; self.Absorptivity = Absorptivity; self.r_b = r_b; self.x0 = x0; self.y0 = y0; self.vx = vx
        self.DeltaH_LV = DeltaH_LV; self.R_v = R_v; self.T_boil = T_boil; self.T_liquidus = T_liquidus; self.T_solidus = T_solidus
        self.debug = debug

class NumericalParams:
    """Container for numerical parameters."""
    def __init__(self,
                # time stepping
                dt: float,
                t_final: float,
                # spatial discretization
                nx: int, ny: int, nz: int,
                K: np.ndarray = None,
                KK: np.ndarray = None,
                phi_p0_tile: np.ndarray = None
                ):
        self.dt = dt
        self.t_final = t_final
        self.nx = nx; self.ny = ny; self.nz = nz
        self.K = K; self.KK = KK
        self.phi_p0_tile = phi_p0_tile

# -------------------------
# Eigenfunctions
# -------------------------
def phi_matrix(n, L):
    """Return matrix of size (nx, nx): phi[m, i] = phi_m(x_i)."""
    x = np.linspace(L / (2 * n), L - L / (2 * n), n)
    m = np.arange(n).reshape(-1, 1)
    pref = np.sqrt(2.0 / L)
    phi = pref * np.cos(m * np.pi * x / L)
    phi[0, :] = 1.0 / np.sqrt(L)
    return x, phi


def phi_p_zero(nz, Lz):
    """Vector of phi_p(0) values for all p."""
    phi0 = np.sqrt(2.0 / Lz) * np.ones(nz)
    phi0[0] = 1.0 / np.sqrt(Lz)
    return phi0

# -------------------------
# Heat source
# -------------------------
def q_laser_field(x: np.ndarray, y: np.ndarray, t: float, params: Params) -> np.ndarray:
    P, Absorptivity, rb = params.P, params.Absorptivity, params.r_b
    x0t = params.x0 + params.vx * t
    y0 = params.y0
    # Q_laser does not contain any absorbitivity, coul be to match other code 
    return (2 * Absorptivity*  P / (np.pi * rb ** 2)) * np.exp(-2 * ((x - x0t) ** 2 + (y - y0) ** 2) / rb ** 2)

def q_evap_point(T: np.ndarray, params: Params) -> np.ndarray:

    # Extract required parameters
    Lv = params.DeltaH_LV      # J/kg
    Rv = params.R_v                   # J/(kg·K)
    Tb = params.T_boil                         # K
    T_safe = np.maximum(T, 1.0)

    q = 0.82 * Lv / np.sqrt(2*np.pi*Rv*T_safe) * np.exp((Lv / (Rv * Tb)) * (1.0 - Tb / T_safe))
    q[T < Tb] = 0.0

    return q
# -------------------------
# Fast DCT
# -------------------------

def dct2_heatflux_scipy(q : np.ndarray, params: Params, phi_p0_tile: np.ndarray) -> np.ndarray:
    ny, nx = q.shape   # explicit ordering: q is (ny, nx)
    pref = np.sqrt(params.Lx*params.Ly)/np.sqrt(nx*ny)

    q_dct = dctn(q, type=2, norm='ortho', workers=-1)   # shape (ny, nx)
    # old: S = (phi_p0[:, None, None] * pref * q_dct[None, :, :]).astype(np.float64)
    S = (phi_p0_tile * pref * q_dct[None, :, :]).astype(np.float64)
    return S.reshape(-1, order='C')
    

# -------------------------
# Reconstruction
# -------------------------
def reconstruct_temperature_field(a: np.ndarray, nx: int, ny: int, nz: int,
                                   phi_x: np.ndarray, phi_y: np.ndarray,
                                   phi_p0: np.ndarray, phi_p0_tile: np.ndarray) -> np.ndarray:
    """
    Fully vectorized reconstruction using tensor contraction:
    T_ij = sum_{m,n,p} a_mnp * phi_y[n,i] * phi_x[m,j] * phi_p0[p]
    """
    # reshape coefficient vector a to (nz, ny, nx)
    A = a.reshape((nz, ny, nx), order='C')
    # old: B = np.tensordot(phi_p0, A, axes=(0,0))
    B = np.sum(phi_p0_tile * A, axis=0)        # shape (ny, nx)
    T = phi_y.T @ (B @ phi_x)                    # (ny, nx)
    return T


def save_fields(T_field, q_laser, q_evap, Xg, Yg, t, params):
    aspect_ratio = params.Lx / params.Ly
    plt.figure(figsize=(10, 20 / aspect_ratio))
    plt.contourf(Xg*1e3, Yg*1e3, T_field, levels=50, cmap='hot')
    plt.colorbar(label='Temperature (K)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(f'Temperature Field at z=0, t={t:.4f} s')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    np.savetxt(f"temperature_field_t{t:.4f}.txt", T_field)
    plt.savefig(f"temperature_field_contour_t{t:.4f}.png")
    
    plt.figure(figsize=(10, 20 / aspect_ratio))
    plt.contourf(Xg*1e3, Yg*1e3, q_laser, levels=50, cmap='hot')
    plt.colorbar(label='Laser Heat Flux (W/m²)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(f'Laser Heat Flux Field at t={t:.4f} s')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f"q_laser_field_contour_t{t:.4f}.png")
    
    plt.figure(figsize=(10, 20 / aspect_ratio))
    plt.contourf(Xg*1e3, Yg*1e3, q_evap, levels=50, cmap='hot')
    plt.colorbar(label='Evaporation Heat Flux (W/m²)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(f'Evaporation Heat Flux Field at t={t:.4f} s')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f"q_evap_field_contour_t{t:.4f}.png")

    plt.close()



def meltpool(T, X, Y, params,
             padding_frac=0.4, display=True, cmap='inferno', levels=80):

    melt_mask = T >= params.T_liquidus
    if not np.any(melt_mask):
        print("No meltpool found (T < T_liquidus everywhere)")
        return

    melt_x = X[melt_mask]; melt_y = Y[melt_mask]
    x_min0, x_max0 = melt_x.min(), melt_x.max()
    y_min0, y_max0 = melt_y.min(), melt_y.max()

    width  = y_max0 - y_min0
    length = x_max0 - x_min0

    x_min = max(0.0, x_min0 - padding_frac * length/2)
    x_max = min(params.Lx, x_max0 + padding_frac * length/2)
    y_min = max(0.0, y_min0 - padding_frac * width/2)
    y_max = min(params.Ly, y_max0 + padding_frac * width/2)

    if not display:
        return  width, length# Skip plotting entirely

    # --- Plot ---
    print(f"Melt-pool width  = {width*1e3:.3f} mm")
    print(f"Melt-pool length = {length*1e3:.3f} mm")
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.contourf(X*1e3, Y*1e3, T, levels=levels, cmap=cmap)
    plt.colorbar(pcm, ax=ax, label='Temperature (K)')
    ax.contour(X*1e3, Y*1e3, T, levels=[params.T_liquidus], colors='k', linewidths=2)
    ax.set_xlim(x_min*1e3, x_max*1e3)
    ax.set_ylim(y_min*1e3, y_max*1e3)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Melt-pool shape (top view)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    return width, length

# ------------------------
# Update, Time Depedency
#--------------------------

#update function for time stepping


def update_coefficients_iterative(
    a, params, num_params, X, Y,
    phi_x, phi_y, phi_p0, phi_p0_tile, t,
    epsilon=1e-10
):
    n_iter_max = 20
    q_laser = q_laser_field(X, Y, t, params)
    q_dct = dct2_heatflux_scipy(q_laser, params, phi_p0_tile)
    a_temp = a.copy()
    aK = a* num_params.K

    for k in range(n_iter_max):
        a_old = a_temp.copy()
        a_temp = aK + num_params.KK * (q_dct)
        T_temp = reconstruct_temperature_field(
            a_temp, num_params.nx, num_params.ny, num_params.nz,
            phi_x, phi_y, phi_p0, phi_p0_tile
        )
        q_evap = q_evap_point(T_temp, params)
        if params.debug:
            dx = params.Lx/num_params.nx; dy = params.Ly/num_params.ny
            print(f"iter {k:02d}: P_evap={np.sum(q_evap)*dx*dy:.6f} W")
        q_dct = dct2_heatflux_scipy(q_laser - q_evap, params, phi_p0_tile)
        if params.debug:
            print(f"  err={np.max(np.abs(a_temp-a_old)):.3e}")

        if np.max(np.abs(a_temp - a_old)) < epsilon:
            if params.debug: print(f"✓ Converged in {k+1} iter\n")
            return a_temp

    if params.debug:
        print(f"No convergence after {n_iter_max} iter\n")
    return a_temp


# -------------------------
# Main simulation runner
# -------------------------
def run_simulation(params: Params, num_params: NumericalParams):
    nx, ny, nz = num_params.nx, num_params.ny, num_params.nz
    a = np.zeros(nx * ny * nz)
    a[0] = params.T0 * np.sqrt(params.Lx * params.Ly * params.Lz)

    # precompute mode matrices
    x, phi_x = phi_matrix(nx, params.Lx)
    y, phi_y = phi_matrix(ny, params.Ly)
    phi_p0 = phi_p_zero(nz, params.Lz)
    phi_p0_tile = phi_p0[:, None, None]
    num_params.phi_p0_tile = phi_p0_tile

    # compute lambda_mnp grid in vectorized form
    # create arrays so broadcasting yields shape (nz, ny, nx) directly
    p = np.arange(nz)[:, None, None]   # (nz,1,1)
    n = np.arange(ny)[None, :, None]   # (1,ny,1)
    m = np.arange(nx)[None, None, :]   # (1,1,nx)

    lam = (m * np.pi / params.Lx) ** 2 + (n * np.pi / params.Ly) ** 2 + (p * np.pi / params.Lz) ** 2
    lam = lam.ravel(order='C')   # now lam shape (nz, ny, nx) flattened to p,n,m order
    # compute K and KK robustly
    K = np.ones_like(lam, dtype=np.float64)
    KK = np.zeros_like(lam, dtype=np.float64)
    mask = lam > 0
    alpha = params.k / (params.rho * params.Ceff)
    temp = alpha * lam * num_params.dt
    K[mask] = np.exp(-temp)[mask]
    # add security on 1 -exp term to avoid numerical issues
    expm1_neg = np.expm1(-temp) # 1 - exp(-x)
    KK[mask] = -expm1_neg[mask] / (params.rho * params.Ceff * alpha * lam[mask])
    #    KK[mask] = -expm1_neg[mask] / (params.rho * params.Ceff * lam[mask]) previously, forgot alpha leading to very low forcing of non zero modes
    
    KK[~mask] = num_params.dt / (params.rho * params.Ceff)
    
    num_params.K, num_params.KK = K, KK
    X, Y = np.meshgrid(x, y)
    print(f"[DEBUG] Max laser flux: {2*params.P/(np.pi*params.r_b**2):.3e} W/m²")
    t = 0.0

    while t < num_params.t_final - 1e-12:
        a = update_coefficients_iterative(a, params, num_params, X, Y, phi_x, phi_y, phi_p0, phi_p0_tile, t)
        t += num_params.dt

    return a, X, Y, phi_x, phi_y, phi_p0

# -------------------------
# Energy check function
# -------------------------
def energy_check(a_final, params: Params, num_params: NumericalParams,
                             phi_x: np.ndarray, phi_y: np.ndarray, phi_p0: np.ndarray):
    """
    Compute ΔE = E(t_final) - E(t=0) using modal coefficients directly.
    Analytical integration over basis functions (no uniform-T assumption).
    """
    nx, ny, nz = num_params.nx, num_params.ny, num_params.nz
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    a0 = np.zeros(nx * ny * nz)
    a0[0] = params.T0 * np.sqrt(params.Lx * params.Ly * params.Lz)

    # --- analytical integrals of φ_m(x) over [0,L]
    def phi_integral(n_modes, L):
        m = np.arange(n_modes)
        I = np.zeros_like(m, dtype=float)
        I[0] = np.sqrt(2.0 / L) * L / np.sqrt(2)  
        for i in range(1, n_modes):
            I[i] = np.sqrt(2.0 / L) * (np.sin(i * np.pi) / (i * np.pi / L))  
        return I

    Ix = phi_integral(nx, Lx)
    Iy = phi_integral(ny, Ly)
    Iz = phi_integral(nz, Lz)

    # reshape coefficients
    a_t = a_final.reshape((nz, ny, nx))
    a_0 = a0.reshape((nz, ny, nx))

    # accumulate ΔE
    total = 0.0
    for p in range(nz):
        for j in range(ny):
            for i in range(nx):
                total += (a_t[p, j, i] - a_0[p, j, i]) * Ix[i] * Iy[j] * Iz[p]

    ΔE = params.rho * params.Ceff * total

    E_in = params.P * num_params.t_final

    print("\n===== ENERGY BALANCE (ΔE from t=0) =====")
    print(f"Total laser input     E_in      = {E_in:.4e} J")
    print(f"Stored energy change  ΔE_stored = {ΔE:.4e} J")
    print(f"Fraction (ΔE/E_in)              = {ΔE / E_in:.4f}")
    print("=========================================\n")

    return E_in, ΔE

# -------------------------
# Example parameters & run
# -------------------------


params = Params(Lx =0.01, Ly=0.005, Lz=0.01,
                rho=7900, Ceff=500, k=14, T0 = 300.0,
                P=70.0, Absorptivity = 0.30, r_b=0.00006, x0=0.001, y0=0.0025, vx=0.1)

num_params = NumericalParams(dt=0.0001, t_final=0.001, nx=512, ny=512, nz=100)
"""
a_final, Xg, Yg, phi_x, phi_y, phi_p0 = run_simulation(params, num_params)

# reconstruct final temperature field at z=0
T_final = reconstruct_temperature_field(a_final, num_params.nx, num_params.ny,
                                            num_params.nz, phi_x, phi_y, phi_p0, num_params.phi_p0_tile)
width, length = meltpool(T_final , Xg, Yg, params)#, display = False)

energy_check(a_final, params, num_params,
                             phi_x, phi_y, phi_p0)
# Store T_final to file
#np.savetxt("T_final.txt", T_final)
# plot final temperature field 
"""
"""
aspect_ratio = params.Lx / params.Ly
plt.figure(figsize=(10, 20 / aspect_ratio))
plt.contourf(Xg*1e3, Yg*1e3, T_final, levels=50, cmap='hot')
plt.colorbar(label='Temperature (K)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Final Temperature Field at z=0')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()  
save_fields(T_final, np.zeros_like(T_final), np.zeros_like(T_final), Xg, Yg, num_params.t_final, params)
"""

import pandas as pd
import itertools


nz_values = [80, 90, 100, 110, 120]
nx_values = [400, 480, 512, 540, 600]
dt_values = [0.5e-4, 0.75e-4, 1e-4, 1.25e-4, 1.5e-4]

results = []

for nz, nx, dt in itertools.product(nz_values, nx_values, dt_values):
    num_params = NumericalParams(dt=dt, t_final=0.001, nx=nx, ny=nx, nz=nz)
    # Run simulation
    a_final, Xg, Yg, phi_x, phi_y, phi_p0 = run_simulation(params, num_params)
    T_final = reconstruct_temperature_field(a_final, nx, nx, nz, phi_x, phi_y, phi_p0, num_params.phi_p0_tile)
    
    # Meltpool metrics
    Tliq = params.T_liquidus
    melt_mask = T_final >= Tliq
    if np.any(melt_mask):
        melt_x = Xg[melt_mask]
        melt_y = Yg[melt_mask]
        width = (melt_y.max() - melt_y.min())*1e3  # mm
        length = (melt_x.max() - melt_x.min())*1e3 # mm
    else:
        width = 0.0
        length = 0.0
    Tmax = T_final.max()

    results.append({
        "nz": nz,
        "nx": nx,
        "dt": dt,
        "Tmax": Tmax,
        "width_mm": width,
        "length_mm": length
    })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("sensitivity_results.csv", index=False)
print("Sensitivity analysis complete. Results saved to 'sensitivity_results.csv'")

