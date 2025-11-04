"""main.py

Analytical / semi-analytic solver for the cuboid heat problem described in the provided LaTeX
notes.


This script implements:
- Uses scipy.fft.dctn for 2D DCT projection of heat flux field
- Vectorized temperature reconstruction using tensor contractions
"""
# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn
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
                P: float, r_b: float, x0: float, y0: float, vx: float,
                # evaporation
                DeltaH_LV: float = 6.0e6, # J/kg
                R_v: float = 150.0, # J/(kg K)
                T_boil: float = 2800.0, # K
                debug: bool = False
                ):
        self.Lx = Lx; self.Ly = Ly; self.Lz = Lz
        self.rho = rho; self.Ceff = Ceff; self.k = k; self.T0 = T0
        self.P = P; self.r_b = r_b; self.x0 = x0; self.y0 = y0; self.vx = vx
        self.DeltaH_LV = DeltaH_LV; self.R_v = R_v; self.T_boil = T_boil
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
                KK: np.ndarray = None
                ):
        self.dt = dt
        self.t_final = t_final
        self.nx = nx; self.ny = ny; self.nz = nz
        self.K = K; self.KK = KK

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
    P, rb = params.P, params.r_b
    x0t = params.x0 + params.vx * t
    y0 = params.y0
    return (2 * P / (np.pi * rb ** 2)) * np.exp(-2 * ((x - x0t) ** 2 + (y - y0) ** 2) / rb ** 2)

def q_evap_point(T: np.ndarray, params: Params) -> np.ndarray:
    """
    Evaporation heat flux at temperature T (array).
    Returns 0 where T < T_boil.
    """
    A = 0.005 / np.sqrt(2.0 * np.pi * params.R_v)
    # avoid division-by-zero or overflow issues
    T_safe = np.maximum(T, params.T_boil)
    exponent = (params.DeltaH_LV / (params.R_v * params.T_boil)) * (1.0 - params.T_boil / T_safe)
    q_evap = A * np.exp(exponent)

    # zero flux where T < T_boil
    q_evap[T < params.T_boil] = 0.0
    return q_evap


# -------------------------
# Fast DCT
# -------------------------

"""
def dct2_heatflux_scipy(q : np.ndarray, params: Params, phi_p0: np.ndarray) -> np.ndarray:
    """#2D DCT-based projection of heat flux q(x,y) at z=0 plane.
"""
    # q shape: (Nx, Ny) where Nx = nx, Ny = ny
    Nx, Ny = q.shape
    pref = np.sqrt(params.Lx*params.Ly)/np.sqrt(Nx*Ny)

    # 2D DCT-II (type=2) orthonormal
    q_dct = dctn(q, type=2, norm='ortho', workers=-1)  # here q_dct is of shape ny nx # workers=-1 uses all available cores
    print("DCT shape:", q_dct.shape)
    S = (phi_p0[:, None, None] * pref * q_dct[None, :, :]).astype(np.float64)
    print("S shape:", S.shape) # here S shape is (nz, ny, nx)
    return S.reshape(-1, order='C')
"""
def dct2_heatflux_scipy(q : np.ndarray, params: Params, phi_p0: np.ndarray) -> np.ndarray:
    ny, nx = q.shape   # explicit ordering: q is (ny, nx)
    pref = np.sqrt(params.Lx*params.Ly)/np.sqrt(nx*ny)

    q_dct = dctn(q, type=2, norm='ortho', workers=-1)   # shape (ny, nx)
    # Build S only for p=0: S shape will be (nz, ny, nx) but nonzero only at p=0
    nz = phi_p0.size
    S = np.zeros((nz, ny, nx), dtype=np.float64)
    S[0, :, :] = (phi_p0[0] * pref * q_dct).astype(np.float64)
    return S.reshape(-1, order='C')



# -------------------------
# Reconstruction
# -------------------------
def reconstruct_temperature_field(a: np.ndarray, nx: int, ny: int, nz: int, phi_x: np.ndarray, phi_y: np.ndarray, phi_p0: np.ndarray) -> np.ndarray:
    """
    Fully vectorized reconstruction using tensor contraction:
    T_ij = sum_{m,n,p} a_mnp * phi_y[n,i] * phi_x[m,j] * phi_p0[p]
    """
    # reshape coefficient vector a to (nz, ny, nx)
    A = a.reshape((nz, ny, nx), order='C')
    # correct contraction
    B = np.tensordot(phi_p0, A, axes=(0,0))        # shape (ny, nx)
    T = phi_y.T @ (B @ phi_x)                    # (ny, nx)
    return T

# evaluate q_laser - q_evap at z=0 plane
def evaluate_heat_source(a: np.ndarray, params: Params, Xg: np.ndarray, Yg: np.ndarray, num_params: NumericalParams, t:float, phi_x: np.ndarray, phi_y: np.ndarray, phi_p0: np.ndarray) -> np.ndarray:
    T_field = reconstruct_temperature_field(a, num_params.nx, num_params.ny,
                                            num_params.nz, phi_x, phi_y, phi_p0)
    q_laser = q_laser_field(Xg,Yg,t,params)
    q_evap = q_evap_point(T_field,params)

    if params.debug :
        save_fields(T_field, q_laser, q_evap, Xg, Yg, t, params)
    return q_laser - q_evap

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

#update function for time stepping
def update_coefficients(a, params, num_params, X, Y, phi_x, phi_y, phi_p0, t):
    # reconstruct field at z=0
    q_field = evaluate_heat_source(a, params, X, Y, num_params, t, phi_x, phi_y, phi_p0)
    q_dct = dct2_heatflux_scipy(q_field, params,  phi_p0)
    return a * num_params.K + num_params.KK * q_dct

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

    # compute lambda_mnp grid in vectorized form
    # create arrays so broadcasting yields shape (nz, ny, nx) directly
    p = np.arange(nz)[:, None, None]   # (nz,1,1)
    n = np.arange(ny)[None, :, None]   # (1,ny,1)
    m = np.arange(nx)[None, None, :]   # (1,1,nx)

    lam = (m * np.pi / params.Lx) ** 2 + (n * np.pi / params.Ly) ** 2 + (p * np.pi / params.Lz) ** 2
    # check if no floating point issues in lam
    if np.any(lam < 0):
        raise ValueError("Negative eigenvalue encountered in lambda computation.")
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
    KK[mask] = -expm1_neg[mask] / (params.rho * params.Ceff * lam[mask])
    # check for any NaN or Inf in KK (should not happen, modes would vanish numerically)
    if np.any(np.isnan(KK)) or np.any(np.isinf(KK)):
        raise ValueError("Numerical issue encountered in KK computation.")
    KK[~mask] = num_params.dt / (params.rho * params.Ceff)
    
    num_params.K, num_params.KK = K, KK
    X, Y = np.meshgrid(x, y)
    print(f"exp(-alpha*lambda_max*dt) = {np.exp(-alpha*np.max(lam)*num_params.dt):.2e}")
    print(f"Prefactor in DCT: {np.sqrt(params.Lx*params.Ly)/np.sqrt(num_params.nx*num_params.ny):.3e}")
    print(f"Max laser flux: {2*params.P/(np.pi*params.r_b**2):.3e} W/m²")
    t = 0.0

    while t < num_params.t_final - 1e-12:
        a = update_coefficients(a, params, num_params, X, Y, phi_x, phi_y, phi_p0, t)
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


params = Params(Lx =0.01, Ly=0.005, Lz=0.2,
                rho=7900, Ceff=500, k=25, T0 = 300.0,
                P=50.0, r_b=0.00006, x0=0.005, y0=0.0025, vx=0.0004)

num_params = NumericalParams(dt=0.1, t_final=10, nx=512, ny=512, nz=8)

a_final, Xg, Yg, phi_x, phi_y, phi_p0 = run_simulation(params, num_params)

# reconstruct final temperature field at z=0
T_final = reconstruct_temperature_field(a_final, num_params.nx, num_params.ny,
                                            num_params.nz, phi_x, phi_y, phi_p0)

energy_check(a_final, params, num_params,
                             phi_x, phi_y, phi_p0)
# Store T_final to file
np.savetxt("T_final.txt", T_final)
# plot final temperature field 
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




