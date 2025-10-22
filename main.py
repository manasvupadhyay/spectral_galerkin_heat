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
                # eigenfunction truncation limits
                m_max: int, n_max: int, p_max: int,
                # spatial discretization
                nx: int, ny: int, nz: int
                ):
        self.dt = dt
        self.t_final = t_final
        self.m_max = m_max; self.n_max = n_max; self.p_max = p_max
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

# Projection of heat source onto eigenmodes


# utilities

def make_modes(M:int,N:int,P:int)->List[Tuple[int,int,int]]:
    return [(m,n,p) for p in range(P) for n in range(N) for m in range(M)]

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
M,N,P = 10,10,10
modes = make_modes(M,N,P)
a = np.zeros(len(modes))
a[0]=300.0 # initial temperature 300K everywhere
nx, ny = 128, 128 # spatial discretization (2**n for FFTW)
x = np.linspace(0,params.Lx,nx)
y = np.linspace(0,params.Ly,ny)
Xg,Yg = np.meshgrid(x,y)


dt = 0.05
t=0.0