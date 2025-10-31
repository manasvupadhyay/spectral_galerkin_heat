"""main.py

Analytical / semi-analytic solver for the cuboid heat problem described in the provided LaTeX
notes.


This script implements:
- Uses scipy.fft.dctn for 2D DCT projection of heat flux field
- Vectorized temperature reconstruction using tensor contractions
"""

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
                rho: float, Ceff: float, k: float, # thermal conductivity
                # laser
                P: float, r_b: float, x0: float, y0: float, vx: float,
                # evaporation
                DeltaH_LV: float = 6.0e6, # J/kg
                R_v: float = 150.0, # J/(kg K)
                T_boil: float = 2800.0, # K
                debug: bool = False
                ):
        self.Lx = Lx; self.Ly = Ly; self.Lz = Lz
        self.rho = rho; self.Ceff = Ceff; self.k = k
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
def dct2_heatflux_scipy(q : np.ndarray, params: Params, phi_p0: np.ndarray) -> np.ndarray:
    """2D DCT-based projection of heat flux q(x,y) at z=0 plane."""
    # q shape: (Nx, Ny) where Nx = nx, Ny = ny
    Nx, Ny = q.shape
    pref = np.sqrt(params.Lx*params.Ly)/np.sqrt(Nx*Ny)

    # 2D DCT-II (type=2) orthonormal
    q_dct = dctn(q, type=2, norm='ortho', workers=-1)  # here q_dct is of shape ny nx # workers=-1 uses all available cores
    print("DCT shape:", q_dct.shape)
    S = (phi_p0[:, None, None] * pref * q_dct[None, :, :]).astype(np.float64)
    print("S shape:", S.shape) # here S shape is (nz, ny, nx)
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
    a[0] = 300.0 * np.sqrt(params.Lx * params.Ly * params.Lz)

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
    lam = lam.ravel(order='C')   # now lam shape (nz, ny, nx) flattened to p,n,m order

    K = np.ones_like(lam)
    KK = np.zeros_like(lam)
    mask = lam > 0
    alpha = params.k / (params.rho * params.Ceff)
    exp_term = np.exp(-alpha * lam[mask] * num_params.dt)
    K[mask] = exp_term
    KK[mask] = (1 - exp_term) / (params.rho * params.Ceff * lam[mask])
    KK[~mask] = num_params.dt / (params.rho * params.Ceff)
    num_params.K, num_params.KK = K, KK
    X, Y = np.meshgrid(x, y)
    t = 0.0
    while t < num_params.t_final - 1e-12:
        a = update_coefficients(a, params, num_params, X, Y, phi_x, phi_y, phi_p0, t)
        t += num_params.dt

    return a, X, Y, phi_x, phi_y, phi_p0

# -------------------------
# Example parameters & run
# -------------------------



params = Params(Lx =0.005, Ly=0.001, Lz=0.01,
                rho=7900, Ceff=500, k=15,
                P=40000.0, r_b=0.00006, x0=0.0005, y0=0.0005, vx=0.2) 

num_params = NumericalParams(dt=0.0001, t_final=0.01, nx=128, ny=128, nz=100)

a_final, Xg, Yg, phi_x, phi_y, phi_p0 = run_simulation(params, num_params)

# reconstruct final temperature field at z=0
T_final = reconstruct_temperature_field(a_final, num_params.nx, num_params.ny,
                                            num_params.nz, phi_x, phi_y, phi_p0)
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




