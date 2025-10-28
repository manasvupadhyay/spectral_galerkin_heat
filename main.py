"""main.py

Analytical / semi-analytic solver for the cuboid heat problem described in the provided LaTeX
notes.


This script implements:
- 
"""
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
from typing import List, Tuple

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
                DeltaH_LV: float = 2.26e6, # J/kg (example for water)
                R_v: float = 461.5, # J/(kg K) (vapor gas constant)
                T_boil: float = 373.15 # K
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
                nx: int, ny: int, nz: int
                ):
        self.dt = dt
        self.t_final = t_final
        self.nx = nx; self.ny = ny; self.nz = nz

# define eigenfunctions
def phi_1d(m: int, x: np.ndarray, L: float) -> np.ndarray:
    """Normalized 1D Neumann cosine eigenfunction on [0,L].
    m=0 -> constant 1/sqrt(L). For m>=1 -> sqrt(2/L)*cos(m*pi*x/L).
    """
    if m == 0:
        return np.full_like(x, 1.0 / np.sqrt(L))
    return np.sqrt(2.0 / L) * np.cos(m * np.pi * x / L)

def phi_p_at_zero(p: int, Lz: float) -> float:   #will be used often at z=0
    """Value of normalized 1D Neumann cosine eigenfunction at z=0."""
    return 1.0 / np.sqrt(Lz) if p == 0 else np.sqrt(2.0 / Lz)

# heat source functions

def q_laser_field(x: np.ndarray, y: np.ndarray, t: float, params: Params) -> np.ndarray:
    P, rb = params.P, params.r_b
    x0t = params.x0 + params.vx * t
    y0 = params.y0
    return (2 * P / (np.pi * rb ** 2)) * np.exp(-2 * ((x - x0t) ** 2 + (y - y0) ** 2) / rb ** 2)


def q_evap_point(T: np.ndarray, params: Params) -> np.ndarray:
    A = 0.005 / np.sqrt(2.0 * np.pi * params.R_v)
    exponent = (params.DeltaH_LV / (params.R_v * params.T_boil)) * (1.0 - params.T_boil / T)
    return A * np.exp(exponent)

# function for reconstruction of temperature field on z=0 plane
def reconstruct_temperature_field(a: np.ndarray, modes: List[Tuple[int,int,int]], params: Params, Xg: np.ndarray, Yg: np.ndarray, num_params: NumericalParams, t:float)->np.ndarray:
    T = np.zeros_like(Xg)
    for ai, (m,n,p) in zip(a, modes):
        T += ai * phi_1d(m,Xg.flatten(),params.Lx).reshape(Xg.shape)*phi_1d(n,Yg.flatten(),params.Ly).reshape(Yg.shape)*phi_p_at_zero(p,params.Lz)
    return T

# evaluate q_laser - q_evap at z=0 plane
def evaluate_heat_source(a: np.ndarray, modes: List[Tuple[int,int,int]], params: Params, Xg: np.ndarray, Yg: np.ndarray, num_params: NumericalParams, t:float)->np.ndarray:
    T_field = reconstruct_temperature_field(a,modes,params,Xg,Yg,num_params,t)
    q_laser = q_laser_field(Xg,Yg,t,params)
    q_evap = q_evap_point(T_field,params)
    return q_laser - q_evap 


def dct2_heatflux_fftw(q : np.ndarray, params: Params) -> np.ndarray:
    Nx, Ny = q.shape
    # prefactor for the field
    prefactor = 2*params.P/(np.pi*params.r_b**2)*np.sqrt((params.Lx*params.Ly)/(Nx*Ny))
    # perform DCT-II using pyfftw
    input = pyfftw.empty_aligned((Nx, Ny), dtype='float64') # create aligned array for pyfftw
    out = pyfftw.empty_aligned((Nx, Ny), dtype='float64')
    input[:] = q
    fft_object = pyfftw.builders.dctn(input, type=2, norm='ortho', threads=-1) # define FFTW DCT-II object
    out[:] = fft_object() # execute DCT-II
    return prefactor * out

def make_modes(M:int,N:int, P:int)->List[Tuple[int,int,int]]:
    return [(m,n,p) for p in range(P) for n in range(N) for m in range(M)]

#update function for time stepping
def update_coefficients(a: np.ndarray, modes: List[Tuple[int,int,int]], params: Params, num_params: NumericalParams, Xg: np.ndarray, Yg: np.ndarray, 
                        dt: float, t: float) -> np.ndarray:
    q_field = evaluate_heat_source(a,modes,params,Xg,Yg,num_params,t)
    q_dct = dct2_heatflux_fftw(q_field,params)
    a_new = np.zeros_like(a)
    for i, (m,n,p) in enumerate(modes):
        lambda_mnp = ( (m*np.pi/params.Lx)**2 + (n*np.pi/params.Ly)**2 + (p*np.pi/params.Lz)**2 )
        exp_factor = np.exp(-params.k/(params.rho*params.Ceff)*lambda_mnp*dt)
        a_new[i] = a[i]*exp_factor + (1 - exp_factor)/(params.rho*params.Ceff*lambda_mnp) * q_dct[i]
    return a_new

# solving

params = Params(Lx =0.001,
                Ly=0.001,
                Lz=0.001,
                rho=7800,
                Ceff=500,
                k=50,
                P=50.0,
                r_b=0.00006,# 0.06mm
                x0=0.0005,
                y0=0.0005,
                vx=0.0001)

num_params = NumericalParams(dt=0.05,
                            t_final=5.0,
                            nx=128, # spatial grid points in x, corresponds to number of DCT modes
                            ny=128,
                            nz=8)

# simulation function 
def run_simulation(params: Params, num_params: NumericalParams):
    nx, ny, nz = num_params.nx, num_params.ny, num_params.nz
    modes = make_modes(num_params.m_max, num_params.n_max, num_params.p_max)
    a = np.zeros(len(modes)) # a contains the expansion coefficients
    a[0]=300.0 # initial temperature 300K everywhere

    dt = num_params.dt
    t_final = num_params.t_final
    t=0.0

    # prepare a grid for DCT 2D evaluations
    dx = params.Lx / nx # grid spacing in x
    dy = params.Ly / ny
    x = np.linspace(dx/2,params.Lx - dx/2,nx)
    y = np.linspace(dy/2,params.Ly - dy/2,ny)
    Xg,Yg = np.meshgrid(x,y)

    while t < t_final:
        a = update_coefficients(a, modes, params, num_params, Xg, Yg, dt, t)
        t += dt

    return a, modes, Xg, Yg