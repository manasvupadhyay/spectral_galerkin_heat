"""
main.py

This script implements:
- Laser heating and evaporation modeling in a 3D domain
- Vectorized temperature reconstruction using modal coefficients
- DCT-based projection of heat flux
- Robust exact integrating-factor modal update and fixed-point iteration
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
        try:
            os.remove(filename)
        except Exception:
            pass

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
                 T_liquidus: float = 1800, T_solidus: float = 1700, debug: bool = True):
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
                 phi_p0_tile: np.ndarray = None):
        self.dt, self.t_final = dt, t_final
        self.nx, self.ny, self.nz = nx, ny, nz
        self.phi_p0_tile = phi_p0_tile
        # we'll store lam and lambda_j arrays for reuse
        self.lam = None
        self.lambda_j = None

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
    # avoid division by zero
    T_safe = np.maximum(T, 1.0)
    q = 0.82 * Lv / np.sqrt(2*np.pi*Rv*T_safe) * np.exp((Lv/(Rv*Tb)) * (1.0 - Tb/T_safe))
    q[T < Tb] = 0.0
    return q

# -------------------------
# DCT projection
# -------------------------
def dct2_heatflux_scipy_old(q, params: Params, phi_p0_tile: np.ndarray):
    """
    2D DCT projection of heat flux q(x,y) onto modal basis.
    Returns flattened modal forcing array of size (nz*ny*nx,).
    """
    ny, nx = q.shape
    pref = np.sqrt(params.Lx * params.Ly) / np.sqrt(nx * ny)
    q_dct = dctn(q, type=2, norm='ortho', workers=-1)   # shape (ny, nx)
    S = (phi_p0_tile * pref * q_dct[None, :, :]).astype(np.float64)
    return S.reshape(-1, order='C')


def dct2_heatflux_scipy(q, params: Params, phi_p0_tile: np.ndarray, debug=False):
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
# Numerically-stable modal exact update
# -------------------------
def modal_update_exact(a, F, lambda_j, dt, rhoC):
    """
    Vectorized exact integrating factor update:
      a_new = a_inf + (a - a_inf) * exp(-lambda*dt)
    where a_inf = F / (rhoC * lambda) (lambda>0)
    and for lambda==0, a_inf = F * dt / rhoC (limit).
    Inputs are flattened arrays (C-order) consistent with modal ordering.
    """
    a = np.asarray(a, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    lambda_j = np.asarray(lambda_j, dtype=np.float64)

    temp = lambda_j * dt
    expm = np.exp(-temp)

    # a_inf (stable)
    a_inf = np.empty_like(a)
    mask_pos = (lambda_j > 0.0)
    if np.any(mask_pos):
        a_inf[mask_pos] = F[mask_pos] / (rhoC * lambda_j[mask_pos])
    mask_zero = ~mask_pos
    if np.any(mask_zero):
        a_inf[mask_zero] = F[mask_zero] * (dt / rhoC)

    a_new = a_inf + (a - a_inf) * expm
    return a_new

# -------------------------
# Temperature reconstruction
# -------------------------
def reconstruct_temperature_field(a, nx, ny, nz, phi_x, phi_y, phi_p0, phi_p0_tile):
    """Vectorized reconstruction T(x,y) at z=0 from modal coefficients a (flattened)."""
    A = a.reshape((nz, ny, nx), order='C')
    B = np.sum(phi_p0_tile * A, axis=0)   # shape (ny, nx)
    T = phi_y.T @ (B @ phi_x)
    return T

# -------------------------
# Meltpool analysis & plotting
# -------------------------
def meltpool(T, X, Y, params, padding_frac=0.4, display=True, cmap='inferno', levels=80):
    """Compute meltpool size and optionally plot with padding."""
    melt_mask = T >= params.T_liquidus
    if not np.any(melt_mask):
        if display:
            print("No meltpool found (T < T_liquidus everywhere)")
        return 0.0, 0.0

    melt_x, melt_y = X[melt_mask], Y[melt_mask]
    x_min0, x_max0 = float(melt_x.min()), float(melt_x.max())
    y_min0, y_max0 = float(melt_y.min()), float(melt_y.max())
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
# Robust fixed-point update including evaporation
# -------------------------
def update_coefficients_fixedpoint(a, params, num_params, X, Y,
                                   phi_x, phi_y, phi_p0, phi_p0_tile,
                                   t,
                                   eps=1e-8, max_iter=30, omega=0.85, debug=False):
    """
    Fixed-point iteration in modal space that uses the exact modal update.
    - a : current modal coefficients flattened
    - returns a_new
    """
    rhoC = params.rho * params.Ceff
    nx, ny, nz = num_params.nx, num_params.ny, num_params.nz

    # lambda_j must already be computed and stored in num_params.lambda_j (shape M,)
    if num_params.lambda_j is None:
        # compute lam grid and lambda_j if not present
        p = np.arange(nz)[:, None, None]
        n_arr = np.arange(ny)[None, :, None]
        m_arr = np.arange(nx)[None, None, :]
        lam_grid = (m_arr * np.pi / params.Lx)**2 + (n_arr * np.pi / params.Ly)**2 + (p * np.pi / params.Lz)**2
        lam_flat = lam_grid.ravel(order='C')
        alpha = params.k / rhoC
        num_params.lam = lam_flat
        num_params.lambda_j = alpha * lam_flat
    lambda_j = num_params.lambda_j  # flattened

    # Laser-only projection
    q_laser = q_laser_field(X, Y, t, params)
    q_dct_laser = dct2_heatflux_scipy(q_laser, params, phi_p0_tile, debug = True)  # shape (M,)

    # initialize forcing and iterate
    # F is modal forcing (S_laser - S_evap) flattened
    F = q_dct_laser.copy()
    a_old = a.copy()

    for k in range(max_iter):
        # exact modal update with current forcing F
        a_new = modal_update_exact(a_old, F, lambda_j, num_params.dt, rhoC)

        # reconstruct temperature and evaluate evaporation
        T_new = reconstruct_temperature_field(a_new, nx, ny, nz, phi_x, phi_y, phi_p0, phi_p0_tile)
        q_evap = q_evap_point(T_new, params)     # grid (ny, nx)

        # project evaporation flux to modal space
        q_dct_evap = dct2_heatflux_scipy(q_evap, params, phi_p0_tile, debug = True)  # shape (M,)

        # new forcing
        F_new = q_dct_laser - q_dct_evap

        # under-relax forcing update
        F = omega * F_new + (1.0 - omega) * F

        diff = np.max(np.abs(a_new - a_old))

        if debug or params.debug:
            dx = params.Lx / nx; dy = params.Ly / ny
            P_evap = float(np.sum(q_evap) * dx * dy)
            print(f"[t={t:.6f}] iter {k:02d}: diff={diff:.3e}, P_evap={P_evap:.6f} W")

        if diff < eps:
            return a_new

        a_old = a_new

    # if not converged, return last iterate
    if debug or params.debug:
        print(f"[WARN] Fixed-point did not converge in {max_iter} iter (diff={diff:.3e}). Returning last iterate.")
    return a_new

# -------------------------
# Main simulation runner
# -------------------------
def run_simulation(params: Params, num_params: NumericalParams, debug=False):
    """
    Run time stepping simulation using the fixed-point modal update.
    Returns final modal coefficients and precomputed objects for reconstruction.
    """
    nx, ny, nz = num_params.nx, num_params.ny, num_params.nz
    # modal vector length
    M = nx * ny * nz

    # initial modal coefficients (flattened)
    a = np.zeros(M, dtype=np.float64)
    a[0] = params.T0 * np.sqrt(params.Lx * params.Ly * params.Lz)

    # Precompute mode matrices and phi_p0_tile
    x, phi_x = phi_matrix(nx, params.Lx)
    y, phi_y = phi_matrix(ny, params.Ly)
    phi_p0 = phi_p_zero(nz, params.Lz)
    phi_p0_tile = phi_p0[:, None, None]
    num_params.phi_p0_tile = phi_p0_tile

    # Precompute lam and lambda_j (flattened)
    p = np.arange(nz)[:, None, None]
    n_arr = np.arange(ny)[None, :, None]
    m_arr = np.arange(nx)[None, None, :]
    lam_grid = (m_arr * np.pi / params.Lx)**2 + (n_arr * np.pi / params.Ly)**2 + (p * np.pi / params.Lz)**2
    lam_flat = lam_grid.ravel(order='C')
    rhoC = params.rho * params.Ceff
    alpha = params.k / rhoC
    lambda_j = alpha * lam_flat
    num_params.lam = lam_flat
    num_params.lambda_j = lambda_j

    # prepare grid for projection and reconstruction
    X, Y = np.meshgrid(x, y)

    t = 0.0
    while t < num_params.t_final - 1e-12:
        a = update_coefficients_fixedpoint(a, params, num_params, X, Y, phi_x, phi_y, phi_p0, phi_p0_tile,
                                           t, eps=1e-8, max_iter=30, omega=0.85, debug=debug)
        t += num_params.dt

    return a, X, Y, phi_x, phi_y, phi_p0

# -------------------------
# Sensitivity analysis
# -------------------------
def run_sensitivity_analysis(params: Params):
    """Run grid-based sensitivity study on nz, nx=ny, dt."""
    nz_values = [20, 40, 60, 80, 100]
    nx_values = [200, 256, 320]
    dt_values = [0.5e-4, 1e-4]

    results = []
    for nz, nx, dt in itertools.product(nz_values, nx_values, dt_values):
        print(f"Running sensitivity run nz={nz}, nx={nx}, dt={dt}")
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

        Tmax = float(np.max(T_final))
        results.append({"nz": nz, "nx": nx, "dt": dt, "Tmax": Tmax, "width_mm": width, "length_mm": length})

    df = pd.DataFrame(results)
    df.to_csv("sensitivity_results.csv", index=False)
    print("Sensitivity analysis complete. Results saved to 'sensitivity_results.csv'")

# -------------------------
# Example run (commented to avoid accidental long execution)
# -------------------------
if __name__ == "__main__":
    params = Params(Lx=0.01, Ly=0.005, Lz=0.01, rho=7850, Ceff=500, k=15, T0=300.0,
                    P=70.0, Absorptivity=0.30, r_b=0.00006, x0=0.001, y0=0.0025, vx=0.1, debug=False)

    # small example run (safe sizes)
    num_params = NumericalParams(dt=0.0001, t_final=0.0005, nx=512, ny=512, nz=10)

    a_final, Xg, Yg, phi_x, phi_y, phi_p0 = run_simulation(params, num_params, debug=False)

    # reconstruct final temperature field at z=0
    T_final = reconstruct_temperature_field(a_final, num_params.nx, num_params.ny, num_params.nz,
                                            phi_x, phi_y, phi_p0, num_params.phi_p0_tile)

    width, length = meltpool(T_final, Xg, Yg, params)
    print(f"Final Tmax = {T_final.max():.2f} K, width={width*1e3:.3f} mm, length={length*1e3:.3f} mm")

    # quick plot
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
