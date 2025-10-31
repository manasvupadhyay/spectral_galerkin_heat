"""main.py

Analytical / semi-analytic solver for the cuboid heat problem described in the provided LaTeX
notes.


This script implements:
- 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn
from typing import List, Tuple
# delete previous files
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
                rho: float, Ceff: float, k: float,
                # laser
                P: float, r_b: float, x0: float, y0: float, vx: float,
                # evaporation
                DeltaH_LV: float = 6.0e6, # J/kg
                R_v: float = 150.0, # J/(kg K)
                T_boil: float = 2800.0 # K
                ):
        self.Lx = Lx; self.Ly = Ly; self.Lz = Lz
        self.rho = rho; self.Ceff = Ceff; self.k = k
        self.P = P; self.r_b = r_b; self.x0 = x0; self.y0 = y0; self.vx = vx
        self.DeltaH_LV = DeltaH_LV; self.R_v = R_v; self.T_boil = T_boil

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
def phi_1d(m: int, x: np.ndarray, L: float) -> np.ndarray:
    """Normalized 1D Neumann cosine eigenfunction on [0,L].
    m=0 -> constant 1/sqrt(L). For m>=1 -> sqrt(2/L)*cos(m*pi*x/L).
    """
    if m == 0:
        return np.full_like(x, 1.0 / np.sqrt(L))
    return np.sqrt(2.0 / L) * np.cos(m * np.pi * x / L)

def phi_p_at_zero(p: int, Lz: float) -> float:   #will be used often at z=0
    """Value of normalized Neumann eigenfunction at z=0."""
    return 1.0 / np.sqrt(Lz) if p == 0 else np.sqrt(2.0 / Lz)

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
# Modes enumerator
# -------------------------
def make_modes(M:int, N:int, P:int) -> List[Tuple[int,int,int]]:
    return [(m, n, p) for p in range(P) for n in range(N) for m in range(M)]

# -------------------------
# Fast DCT
# -------------------------
def dct2_heatflux_scipy(q : np.ndarray, params: Params, num_params : NumericalParams) -> np.ndarray:
    """
    Fast SciPy DCT-II-based routine.
    Returns a flattened vector of length M*N*P (ordered p, n, m) matching make_modes().
    """
    # q shape: (Nx, Ny) where Nx = nx, Ny = ny
    Nx, Ny = q.shape
    prefactor = np.sqrt(params.Lx*params.Ly)/np.sqrt(Nx*Ny)

    # 2D DCT-II (type=2) orthonormal
    q_dct = dctn(q, type=2, norm='ortho', workers=-1)   # shape (Nx, Ny); axis 0 -> m, axis 1 -> n
    print("DCT shape:", q_dct.shape)
    # allocate S in shape (P, N, M) so that flatten(C-order) yields iterate p,n,m
    S_pnm = np.empty((num_params.nz, Ny, Nx), dtype=np.float64)
    for p in range(num_params.nz):
        S_pnm[p] = phi_p_at_zero(p, params.Lz) * prefactor * q_dct.T
    return S_pnm.ravel(order='C')

# -------------------------
# Reconstruction
# -------------------------
def reconstruct_temperature_field(a: np.ndarray, modes: List[Tuple[int,int,int]], params: Params, phi_x: List[np.ndarray], phi_y: List[np.ndarray])->np.ndarray:
    nx, ny = len(phi_x[0]), len(phi_y[0])
    T = np.zeros((ny, nx))
    for ai, (m, n, p) in zip(a, modes):
        T += ai * np.outer(phi_y[n], phi_x[m]) * phi_p_at_zero(p, params.Lz)
    return T

# evaluate q_laser - q_evap at z=0 plane
def evaluate_heat_source(a: np.ndarray, modes: List[Tuple[int,int,int]], params: Params, Xg: np.ndarray, Yg: np.ndarray, num_params: NumericalParams, t:float)->np.ndarray:
    T_field = reconstruct_temperature_field(a,modes,params,Xg,Yg)
    # save intermediate image of T plot and text file for debugging
    aspect_ratio = params.Lx / params.Ly
    plt.figure(figsize=(10, 20 / aspect_ratio))
    plt.contourf(Xg*1e3, Yg*1e3, T_field, levels=50, cmap='hot')
    plt.colorbar(label='Temperature (K)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(f'Temperature Field at z=0, t={t:.4f} s')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f"temperature_field_contour_t{t:.4f}.png")
    plt.close()
    np.savetxt(f"temperature_field_t{t:.4f}.txt", T_field)
    # evaluate heat sources + save images for debugging
    q_laser = q_laser_field(Xg,Yg,t,params)
    plt.figure(figsize=(10, 20 / aspect_ratio))
    plt.contourf(Xg*1e3, Yg*1e3, q_laser, levels=50, cmap='hot')
    plt.colorbar(label='Laser Heat Flux (W/m²)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(f'Laser Heat Flux Field at t={t:.4f} s')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f"q_laser_field_contour_t{t:.4f}.png")
    plt.close()
    q_evap = q_evap_point(T_field,params)
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
    return q_laser - q_evap

#update function for time stepping
def update_coefficients(a: np.ndarray, modes: List[Tuple[int,int,int]], params: Params, num_params: NumericalParams, Xg: np.ndarray, Yg: np.ndarray, 
                        t: float) -> np.ndarray:
    q_field = evaluate_heat_source(a,modes,params,Xg,Yg,num_params,t)
    print("Heat source field shape:", q_field.shape)
    q_dct = dct2_heatflux_scipy(q_field.T,params, num_params)
    a_new = np.zeros_like(a)
    a_new = a * num_params.K + num_params.KK * q_dct
    return a_new

# -------------------------
# Main simulation runner
# -------------------------
def run_simulation(params: Params, num_params: NumericalParams):
    nx, ny = num_params.nx, num_params.ny
    modes = make_modes(nx, ny, num_params.nz)
    a = np.zeros(len(modes))
    a[0] = 300.0 * np.sqrt(params.Lx * params.Ly * params.Lz)

    # Precompute coefficients
    K = np.zeros(len(modes))
    KK = np.zeros(len(modes))
    for i, (m, n, p) in enumerate(modes):
        lam = (m * np.pi / params.Lx)**2 + (n * np.pi / params.Ly)**2 + (p * np.pi / params.Lz)**2
        if lam == 0:
            K[i] = 1.0
            KK[i] = num_params.dt / (params.rho * params.Ceff)
        else:
            exp_term = np.exp(-params.k / (params.rho * params.Ceff) * lam * num_params.dt)
            K[i] = exp_term
            KK[i] = (1 - exp_term) / (params.rho * params.Ceff * lam)
    num_params.K, num_params.KK = K, KK

    # Grid + precomputed eigenmodes
    x = np.linspace(params.Lx / (2 * nx), params.Lx - params.Lx / (2 * nx), nx)
    y = np.linspace(params.Ly / (2 * ny), params.Ly - params.Ly / (2 * ny), ny)
    Xg, Yg = np.meshgrid(x, y)
    phi_x = [phi_1d(m, x, params.Lx) for m in range(nx)]
    phi_y = [phi_1d(n, y, params.Ly) for n in range(ny)]
    t = 0.0
    while t < num_params.t_final - 1e-12:
        a = update_coefficients(a, modes, params, num_params, Xg, Yg, t)
        t += num_params.dt

    return a, modes, Xg, Yg, phi_x, phi_y

# -------------------------
# Example parameters & run
# -------------------------



params = Params(Lx =0.005, Ly=0.001, Lz=0.01,
                rho=7900, Ceff=500, k=15,
                P=50.0, r_b=0.00006, x0=0.0005, y0=0.0005, vx=0.004) 

num_params = NumericalParams(dt=0.001, t_final=0.01, nx=128, ny=128, nz=8)

a_final, modes, Xg, Yg, phi_x, phi_y = run_simulation(params, num_params)

# reconstruct final temperature field at z=0
T_final = reconstruct_temperature_field(a_final, modes, params, phi_x, phi_y)
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




