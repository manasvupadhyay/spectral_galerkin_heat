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
for filename in os.listdir('.'):
    if filename.startswith('temperature_field_t') and filename.endswith('.txt'):
        os.remove(filename)
    if filename.startswith('temperature_field_contour_t') and filename.endswith('.png'):
        os.remove(filename)
    if filename.startswith('q_laser_field_contour_t') and filename.endswith('.png'):
        os.remove(filename)
    if filename.startswith('q_evap_field_contour_t') and filename.endswith('.png'):
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
    return [(m,n,p) for p in range(P) for n in range(N) for m in range(M)]

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
    q_dct = dctn(q, type=2, norm='ortho')   # shape (Nx, Ny); axis 0 -> m, axis 1 -> n
    print("DCT shape:", q_dct.shape)
    # allocate S in shape (P, N, M) so that flatten(C-order) yields iterate p,n,m
    S_pnm = np.empty((num_params.nz, Ny, Nx), dtype=np.float64)
    for p in range(num_params.nz):
        phi_p0 = phi_p_at_zero(p, params.Lz)
        # all m,n modes share same q_dct; multiply by phi_p(0) and prefactor
        S_mn = phi_p0 * prefactor * q_dct
        # store transposed into S_pnm[p, n, m] (so p,n,m ordering)
        S_pnm[p, :, :] = S_mn.T  # S_mn shape (Nx, Ny) -> transpose to (Ny, Nx)
    # flatten in C-order to get vector consistent with make_modes
    return S_pnm.ravel(order='C')  # length Mx*Ny*P_modes

# -------------------------
# Reconstruction
# -------------------------
def reconstruct_temperature_field(a: np.ndarray, modes: List[Tuple[int,int,int]], params: Params, Xg: np.ndarray, Yg: np.ndarray)->np.ndarray:
    T = np.zeros_like(Xg)
    for ai, (m,n,p) in zip(a, modes):
        T += ai * phi_1d(m,Xg.flatten(),params.Lx).reshape(Xg.shape)*phi_1d(n,Yg.flatten(),params.Ly).reshape(Yg.shape)*phi_p_at_zero(p,params.Lz)
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
                        dt: float, t: float) -> np.ndarray:
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
    modes = make_modes(num_params.nx, num_params.ny, num_params.nz)
    a = np.zeros(len(modes)) # a contains the expansion coefficients
    # initialize a to get 300 k (initial temperature is zero everywhere)
    a[0] = 300.0 * np.sqrt(params.Lx*params.Ly*params.Lz) # initial temperature 300K everywhere (normalized !!)
    # time stepping parameters
    dt = num_params.dt
    t_final = num_params.t_final

    # prepare vectors for time update of coefficients
    # Kmnp = exp(-k/(rho*Ceff)*lambda_mnp*dt)
    # Kkmnp = (1 - Kmnp)/(rho*Ceff*lambda_mnp)
    K = np.zeros(len(modes))
    KK = np.zeros(len(modes))
    for idx, (m,n,p) in enumerate(modes):
        lambda_mnp = ( (m*np.pi/params.Lx)**2 + 
                      (n*np.pi/params.Ly)**2 + 
                      (p*np.pi/params.Lz)**2 )
        if lambda_mnp == 0.0:
            # avoid division by zero for the (0,0,0) mode 
            # This will be handled later, for now the mean of T remains constant
            K[idx] = 1.0
            KK[idx] = dt / (params.rho * params.Ceff)
        else:
            K[idx] = np.exp(-params.k/(params.rho*params.Ceff)*lambda_mnp*dt)
            KK[idx] = (1 - np.exp(-params.k/(params.rho*params.Ceff)*lambda_mnp*dt))/(params.rho*params.Ceff*lambda_mnp)
    # vectorize K and KK
    num_params.K = K
    num_params.KK = KK

    # prepare a grid for DCT 2D evaluations
    dx = params.Lx / nx # grid spacing in x
    dy = params.Ly / ny
    x = np.linspace(dx/2,params.Lx - dx/2,nx)
    y = np.linspace(dy/2,params.Ly - dy/2,ny)
    Xg, Yg = np.meshgrid(x,y) # grid at cell centers,  Xg and Yg are of shape (nx,ny)
    t = 0.0
    while t < t_final - 1e-12:
        a = update_coefficients(a, modes, params, num_params, Xg, Yg, dt, t)
        t += dt

    return a, modes, Xg, Yg

# -------------------------
# Example parameters & run
# -------------------------



params = Params(Lx =0.005, Ly=0.001, Lz=0.01,
                rho=7900, Ceff=500, k=15,
                P=50.0, r_b=0.00006, x0=0.0005, y0=0.0005, vx=0.004) 

num_params = NumericalParams(dt=0.001, t_final=0.01, nx=128, ny=128, nz=8)

a_final, modes, Xg, Yg = run_simulation(params, num_params)

# reconstruct final temperature field at z=0
T_final = reconstruct_temperature_field(a_final, modes, params, Xg, Yg)   
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




