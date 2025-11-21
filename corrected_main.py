"""
main.py

This script implements:
- Laser heating and evaporation modeling in a 3D domain
- Vectorized temperature reconstruction using modal coefficients
- DCT-based projection of heat flux
- Iterative correction for evaporation
- Meltpool analysis and sensitivity studies
"""

# -------------------------
# Imports
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn
import os
import pandas as pd
import itertools

# -------------------------
# Clean old output files
# -------------------------
for filename in os.listdir('.'):
    if filename.startswith(('temperature_field_t', 'temperature_field_contour_t',
                            'q_laser_field_contour_t', 'q_evap_field_contour_t')):
        os.remove(filename)

# -------------------------
# Parameter Classes
# -------------------------
class Params:
    """Container for physical, material, laser, and evaporation parameters."""
    def __init__(self,
                 # geometry
                 Lx: float, Ly: float, Lz: float,
                 # material properties
                 rho: float, Ceff: float, k: float, T0: float,
                 # laser
                 P: float, Absorptivity: float, r_b: float, x0: float, y0: float, vx: float,
                 # evaporation
                 DeltaH_LV: float = 7.41e6, R_v: float = 150.0, T_boil: float = 3090.0,
                 T_liquidus: float = 1800, T_solidus: float = 1700, debug: bool = False):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.rho, self.Ceff, self.k, self.T0 = rho, Ceff, k, T0
        self.P, self.Absorptivity, self.r_b = P, Absorptivity, r_b
        self.x0, self.y0, self.vx = x0, y0, vx
        self.DeltaH_LV, self.R_v, self.T_boil = DeltaH_LV, R_v, T_boil
        self.T_liquidus, self.T_solidus = T_liquidus, T_solidus
        self.debug = debug

class NumericalParams:
    """Container for numerical and modal parameters."""
    def __init__(self, dt: float, t_final: float, nx: int, ny: int, nz: int,
                 K: np.ndarray = None, KK: np.ndarray = None, phi_p0_tile: np.ndarray = None):
        self.dt, self.t_final = dt, t_final
        self.nx, self.ny, self.nz = nx, ny, nz
        self.K, self.KK = K, KK
        self.phi_p0_tile = phi_p0_tile

# -------------------------
# Eigenfunctions
# -------------------------
def phi_matrix(n, L):
    """Return grid points and cosine basis matrix."""
    x = np.linspace(L/(2*n), L-L/(2*n), n)
    m = np.arange(n).reshape(-1,1)
    pref = np.sqrt(2.0 / L)
    phi = pref * np.cos(m * np.pi * x / L)
    phi[0, :] = 1.0 / np.sqrt(L)
    return x, phi

def phi_p_zero(nz, Lz):
    """Phi_p(0) for z=0 plane."""
    # comment for test
    phi0 = np.sqrt(2.0 / Lz) * np.ones(nz)
    phi0[0] = 1.0 / np.sqrt(Lz)
    return phi0

# -------------------------
# Heat fluxes
# -------------------------
def q_laser_field(x, y, t, params: Params):
    """Gaussian laser flux at top surface (W/m²)."""
    x0t = params.x0 + params.vx * t
    return (2 * params.Absorptivity * params.P / (np.pi * params.r_b ** 2) *
            np.exp(-2 * ((x - x0t)**2 + (y - params.y0)**2) / params.r_b**2))

def q_evap_point(T, params: Params):
    """Evaporation flux at temperature T using Lv-based formula."""
    Lv, Rv, Tb = params.DeltaH_LV, params.R_v, params.T_boil
    T_safe = np.maximum(T, 1.0)
    q = 0.82 * Lv / np.sqrt(2*np.pi*Rv*T_safe) * np.exp((Lv/(Rv*Tb)) * (1.0 - Tb/T_safe))
    q[T < Tb] = 0.0
    return q

# -------------------------
# DCT projection
# -------------------------
def dct2_heatflux_scipy(q, params: Params, phi_p0_tile: np.ndarray):
    """2D DCT projection of heat flux onto modal basis."""
    #ny, nx = q.shape
    #pref = np.sqrt(params.Lx * params.Ly)  / np.sqrt(nx * ny)
    q_dct = dctn(q, type=2, norm='backward', workers=-1)
    S = (phi_p0_tile * q_dct[None,:,:]).astype(np.float64)
    return S.reshape(-1, order='C')

def dct2_heatflux_scipy_test(q, params: Params, phi_p0_tile: np.ndarray):
    """2D DCT projection of heat flux onto modal basis, with debug image output."""
    ny, nx = q.shape

    #pref = np.sqrt(params.Lx * params.Ly) / np.sqrt(nx * ny)
    q_dct = dctn(q, type=2, norm='backward', workers=-1)

    # S has shape (nz, ny, nx)
    S = (phi_p0_tile  * q_dct[None, :, :]).astype(np.float64)

    # ---- Debug visualization of S[p,:,:] for random p ----
    nz = S.shape[0]
    num_to_show = min(3, nz)     # show at most 3 slices
    p_indices = np.random.choice(nz, size=num_to_show, replace=False)

    for p in p_indices:
        plt.figure(figsize=(5,4))
        plt.imshow(S[p], origin='lower', aspect='auto')
        plt.title(f"S slice for p = {p}")
        plt.colorbar(label="value")
        plt.tight_layout()
        plt.show()

    return S.reshape(-1, order='C')

def dct2_heatflux_scipy_debug(q, params: Params, phi_p0_tile: np.ndarray, debug=True):
    """
    2D DCT projection of heat flux q(x,y) onto modal basis.
    Returns flattened modal forcing array of size (nz*ny*nx,).
    """

    ny, nx = q.shape
    pref = np.sqrt(params.Lx * params.Ly) / np.sqrt(nx * ny)
    
    # 2D DCT
    q_dct = dctn(q, type=2, norm='ortho', workers=-1)   # shape (ny, nx)
    
    if debug:
        print(f"[DCT debug] q_dct: min={q_dct.min():.4e}, max={q_dct.max():.4e}, mean={q_dct.mean():.4e}")
    
    # Multiply by phi_p0_tile
    S = (phi_p0_tile * pref * q_dct[None, :, :]).astype(np.float64)
    
    if debug:
        print(f"[DCT debug] S after phi_p0_tile: min={S.min():.4e}, max={S.max():.4e}, mean={S.mean():.4e}")
    
    S_flat = S.reshape(-1, order='C')

    if debug:
        print(f"[DCT debug] S_flat: min={S_flat.min():.4e}, max={S_flat.max():.4e}, mean={S_flat.mean():.4e}")

        # Histogram plot
        plt.figure(figsize=(7,4))
        plt.hist(S_flat, bins=50, log=True, color='orange', edgecolor='k')
        plt.xlabel("Modal coefficient value")
        plt.ylabel("Count (log scale)")
        plt.title("Histogram of flattened DCT modal coefficients")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Pause execution
        input("Press Enter to continue...")

    return S_flat

# -------------------------
# Temperature reconstruction
# -------------------------
def reconstruct_temperature_field(a, nx, ny, nz, phi_x, phi_y, phi_p0, phi_p0_tile):
    """Vectorized reconstruction T(x,y) at z=0 from modal coefficients."""
    A = a.reshape((nz, ny, nx), order='C')
    B = np.sum(phi_p0_tile * A, axis=0)
    T = phi_y.T @ (B @ phi_x)
    return T

# -------------------------
# Meltpool analysis & plotting
# -------------------------
def meltpool(T, X, Y, params, padding_frac=0.4, display=True, cmap='inferno', levels=80):
    """Compute meltpool size and optionally plot with padding."""
    melt_mask = T >= params.T_liquidus
    if not np.any(melt_mask): return 0.0, 0.0

    melt_x, melt_y = X[melt_mask], Y[melt_mask]
    x_min0, x_max0 = melt_x.min(), melt_x.max()
    y_min0, y_max0 = melt_y.min(), melt_y.max()
    width, length = y_max0 - y_min0, x_max0 - x_min0

    # Apply padding
    x_min = max(0.0, x_min0 - padding_frac*length/2)
    x_max = min(params.Lx, x_max0 + padding_frac*length/2)
    y_min = max(0.0, y_min0 - padding_frac*width/2)
    y_max = min(params.Ly, y_max0 + padding_frac*width/2)

    if display:
        print(f"Melt-pool width  = {width*1e3:.3f} mm")
        print(f"Melt-pool length = {length*1e3:.3f} mm")
        fig, ax = plt.subplots(figsize=(8,6))
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

# -------------------------
# Iterative update for time stepping
# -------------------------
def update_coefficients_iterative(a, params, num_params, X, Y, phi_x, phi_y, phi_p0, phi_p0_tile, t, epsilon=1e-10):
    """Iterative predictor-corrector update including evaporation."""
    n_iter_max = 20
    q_laser = q_laser_field(X, Y, t, params)
    q_dct = dct2_heatflux_scipy(q_laser, params, phi_p0_tile)
    a_temp = a.copy()
    aK = a * num_params.K

    for k in range(n_iter_max):
        a_old = a_temp.copy()
        a_temp = aK + num_params.KK * q_dct
        T_temp = reconstruct_temperature_field(a_temp, num_params.nx, num_params.ny, num_params.nz,
                                               phi_x, phi_y, phi_p0, phi_p0_tile)
        q_evap = q_evap_point(T_temp, params)
        if params.debug:
            dx = params.Lx / num_params.nx
            dy = params.Ly / num_params.ny
            print(f"iter {k:02d}: P_evap={np.sum(q_evap)*dx*dy:.6f} W")
        q_dct = dct2_heatflux_scipy(q_laser - q_evap, params, phi_p0_tile)

        if np.max(np.abs(a_temp - a_old)) < epsilon:
            if params.debug: print(f"✓ Converged in {k+1} iter\n")
            return a_temp

    if params.debug: print(f"No convergence after {n_iter_max} iter\n")
    return a_temp

# -------------------------
# Main simulation runner
# -------------------------
def run_simulation(params: Params, num_params: NumericalParams):
    """Run time-stepping simulation and return final modal coefficients."""
    nx, ny, nz = num_params.nx, num_params.ny, num_params.nz
    a = np.zeros(nx*ny*nz)
    a[0] = params.T0 * np.sqrt(params.Lx*params.Ly*params.Lz)

    # Precompute mode matrices
    x, phi_x = phi_matrix(nx, params.Lx)
    y, phi_y = phi_matrix(ny, params.Ly)
    phi_p0 = phi_p_zero(nz, params.Lz)
    phi_p0_tile = phi_p0[:, None, None]
    num_params.phi_p0_tile = phi_p0_tile

    p = np.arange(nz)[:, None, None]
    n = np.arange(ny)[None, :, None]
    m = np.arange(nx)[None, None, :]

    alpha = params.k / (params.rho * params.Ceff)

    lam = (m * np.pi / params.Lx) ** 2 + (n * np.pi / params.Ly) ** 2 + (p * np.pi / params.Lz) ** 2
    lam_flat = lam.ravel(order='C')

    K = np.ones_like(lam_flat, dtype=np.float64)
    mask_nonzero_lam = lam_flat > 0
    K[mask_nonzero_lam] = np.exp(-alpha * lam_flat * num_params.dt)[mask_nonzero_lam]

    # Redefine the delta function masks using the broadcastable 3D index arrays
    delta_m0 = (m == 0).astype(np.float64)
    delta_n0 = (n == 0).astype(np.float64)
    delta_p0 = (p == 0).astype(np.float64)

    # Define the C factors in their broadcastable 3D shapes
    Cm_sqrt = np.sqrt(2.0 - delta_m0)/ np.sqrt(params.Lx )
    Cn_sqrt = np.sqrt(2.0 - delta_n0)/ np.sqrt(params.Ly)
    Cp_sqrt = np.sqrt(2.0 - delta_p0)

    # S_factor_3D is computed via broadcasting the 3D arrays
    S_factor_3D = (Cm_sqrt * Cn_sqrt * Cp_sqrt)  

    # Flatten the 3D result to match the shape of lam_flat and K
    S_factor = S_factor_3D.ravel(order='C')

    KK = np.zeros_like(lam_flat, dtype=np.float64)
    mask_singularity = lam_flat == 0

    KK[mask_nonzero_lam] = S_factor[mask_nonzero_lam] * (1.0 - K[mask_nonzero_lam]) / (lam_flat[mask_nonzero_lam] * params.k)
    KK[mask_singularity] = S_factor[mask_singularity] * num_params.dt / (params.rho * params.Ceff)
    num_params.K = K
    num_params.KK = KK

    X, Y = np.meshgrid(x, y)
    t = 0.0

    while t < num_params.t_final - 1e-12:
        a = update_coefficients_iterative(a, params, num_params, X, Y, phi_x, phi_y, phi_p0, phi_p0_tile, t)
        t += num_params.dt

    return a, X, Y, phi_x, phi_y, phi_p0

# -------------------------
# Sensitivity analysis
# -------------------------

def run_sensitivity_analysis(params: Params):
    """Run grid-based sensitivity study on nz, nx=ny, dt."""
    nz_values = [80, 90, 100, 110, 120]
    nx_values = [400, 480, 512, 540, 600]
    dt_values = [0.5e-4, 0.75e-4, 1e-4, 1.25e-4, 1.5e-4]

    results = []
    for nz, nx, dt in itertools.product(nz_values, nx_values, dt_values):
        num_params = NumericalParams(dt=dt, t_final=0.001, nx=nx, ny=nx, nz=nz)
        a_final, Xg, Yg, phi_x, phi_y, phi_p0 = run_simulation(params, num_params)
        T_final = reconstruct_temperature_field(a_final, nx, nx, nz, phi_x, phi_y, phi_p0, num_params.phi_p0_tile)

        melt_mask = T_final >= params.T_liquidus
        if np.any(melt_mask):
            melt_x, melt_y = Xg[melt_mask], Yg[melt_mask]
            width = (melt_y.max() - melt_y.min())*1e3
            length = (melt_x.max() - melt_x.min())*1e3
        else:
            width, length = 0.0, 0.0

        Tmax = T_final.max()
        results.append({"nz": nz, "nx": nx, "dt": dt, "Tmax": Tmax, "width_mm": width, "length_mm": length})

    df = pd.DataFrame(results)
    df.to_csv("sensitivity_results.csv", index=False)
    print("Sensitivity analysis complete. Results saved to 'sensitivity_results.csv'")


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
# Example run (commented to avoid accidental long execution)
# -------------------------

params = Params(Lx=0.01, Ly=0.005, Lz=0.01, rho=7900, Ceff=500, k=14, T0=300.0,
                P=200.0, Absorptivity=0.30, r_b=0.00006, x0=0.001, y0=0.0025, vx=0.5, debug=False)

import pandas as pd
import matplotlib.pyplot as plt

Ln = [5, 20, 30, 35, 40, 50, 55, 60, 120]
L_Lz = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1]
Lt_max = []

results = []

for nz in Ln:
    print(f"Running simulation for Lz = {nz} ...")
    params = Params(Lx=0.01, Ly=0.005, Lz=0.02, rho=7900, Ceff=500, k=14, T0=300.0,
                P=200.0, Absorptivity=0.30, r_b=0.00006, x0=0.001, y0=0.0025, vx=0.5, debug=False)

    num_params = NumericalParams(
        dt=0.0001,
        t_final=0.001,
        nx=512,
        ny=256,
        nz=nz
    )
    
    a_final, Xg, Yg, phi_x, phi_y, phi_p0 = run_simulation(params, num_params)
    T_final = reconstruct_temperature_field(
        a_final,
        num_params.nx,
        num_params.ny,
        num_params.nz,
        phi_x,
        phi_y,
        phi_p0,
        num_params.phi_p0_tile
    )
    energy_check(a_final, params, num_params,
                             phi_x, phi_y, phi_p0)
    melt_mask = T_final >= params.T_liquidus
    if np.any(melt_mask):
        melt_x, melt_y = Xg[melt_mask], Yg[melt_mask]
        width = (melt_y.max() - melt_y.min())*1e3
        length = (melt_x.max() - melt_x.min())*1e3
    else:
        width, length = 0.0, 0.0

    Tmax = float(np.max(T_final))
    Lt_max.append(Tmax)
    results.append({"nz": nz, "Tmax": Tmax, "width": width, "length": length})
    
    print(f"  → Tmax = {Tmax:.2f} K")
    print(f"  → width = {width:.4f} mm")
    print(f"  → length = {width:.4f} ")


# Save results
df = pd.DataFrame(results)
df.to_csv("sensitivity_nz_temperature.csv", index=False)
print("\nSaved results to sensitivity_nz_temperature.csv")

# Plot
plt.figure(figsize=(7,5))
plt.plot(Ln, Lt_max, marker='o')
plt.xlabel('Lz (height in z)')
plt.ylabel('T_max (K)')
plt.title('Max Temperature vs Number of xy-modes')
plt.grid(True)
plt.tight_layout()
plt.show()


width, length = meltpool(T_final, Xg, Yg, params)

aspect_ratio = params.Lx / params.Ly
plt.figure(figsize=(10, 20 / aspect_ratio))
plt.contourf(Xg*1e3, Yg*1e3, T_final-1, levels=50, cmap='hot')
plt.colorbar(label='Temperature (K)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Final Temperature Field at z=0')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()  
"""
run_sensitivity_analysis(params)
"""
