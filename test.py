"""
minimal_heat_solver.py

Minimal script for 3D heat diffusion using a spectral method (modal expansion).
Focuses on core time-stepping with a fixed, constant heat flux (no evaporation).
"""

import numpy as np
from scipy.fft import dctn

# -------------------------
# Parameter Classes
# -------------------------
class Params:
    """Container for essential physical and material parameters."""
    def __init__(self, Lx: float, Ly: float, Lz: float,
                 rho: float, Ceff: float, k: float, T0: float,
                 P: float, Absorptivity: float, r_b: float):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.rho, self.Ceff, self.k, self.T0 = rho, Ceff, k, T0
        # Laser parameters used only for the fixed source term
        self.P, self.Absorptivity, self.r_b = P, Absorptivity, r_b

class NumericalParams:
    """Container for numerical and modal parameters."""
    def __init__(self, dt: float, t_final: float, nx: int, ny: int, nz: int):
        self.dt, self.t_final = dt, t_final
        self.nx, self.ny, self.nz = nx, ny, nz
        # K and KK will be set later in run_simulation
        self.K = None
        self.KK = None
        self.phi_p0_tile = None

# -------------------------
# Eigenfunctions
# -------------------------
def phi_matrix(n, L):
    """Return grid points and cosine basis matrix."""
    x = np.linspace(L/(2*n), L-L/(2*n), n)
    m = np.arange(n).reshape(-1,1)
    # The normalization constant is implicitly handled here
    pref = np.sqrt(2.0 / L)
    phi = pref * np.cos(m * np.pi * x / L)
    phi[0, :] = 1.0 / np.sqrt(L) # Correct normalization for m=0
    return x, phi

def phi_p_zero(nz, Lz):
    """Phi_p(0) for z=0 plane (The value of the eigenfunction at z=0)."""
    phi0 = np.sqrt(2.0 / Lz) * np.ones(nz)
    phi0[0] = 1.0 / np.sqrt(Lz)
    return phi0

# -------------------------
# Heat fluxes (Fixed Source)
# -------------------------
def q_laser_field(x, y, t, params: Params):
    """Gaussian laser flux at top surface (W/m²). For minimal case, let's fix it at t=0."""
    x0t = 0.0 # Fixed laser position for simplicity
    return (2 * params.Absorptivity * params.P / (np.pi * params.r_b ** 2) *
            np.exp(-2 * (x**2 + y**2) / params.r_b**2))

def dct_projection(q, params: Params, phi_p0_tile: np.ndarray):
    """2D DCT projection of fixed heat flux onto modal basis."""
    
    # 1. 2D DCT of the spatial surface flux (q has shape (ny, nx))
    q_dct = dctn(q, type=2, norm='backward', workers=-1)
    
    # 2. Normalization factor from the spatial integral/discretization
    nx, ny = q.shape[1], q.shape[0]
    # For DCT-II with norm='backward', the total scaling includes L_x*L_y and sqrt(N_x*N_y)
    # The DCT library choice handles some scaling; we add the physical Lx*Ly factor.
    # The original script had: pref = np.sqrt(params.Lx * params.Ly) / np.sqrt(nx * ny) 
    # Let's simplify this by applying the geometric factor needed to match the basis normalization.
    pref = np.sqrt(params.Lx * params.Ly) / (nx * ny) # Simplification for normalization consistency

    # 3. Project onto the z-modes by multiplying by Phi_p(0)
    # S has shape (nz, ny, nx)
    S = (phi_p0_tile * pref * q_dct[None, :, :]).astype(np.float64)
    
    return S.ravel(order='C')

# -------------------------
# Temperature reconstruction
# -------------------------
def reconstruct_temperature_field(a, nx, ny, nz, phi_x, phi_y, phi_p0_tile):
    """Vectorized reconstruction T(x,y) at z=0 from modal coefficients."""
    A = a.reshape((nz, ny, nx), order='C')
    # Use the pre-tiled phi_p0_tile for the sum over z-modes
    B = np.sum(phi_p0_tile * A, axis=0)
    T = phi_y.T @ (B @ phi_x)
    return T

# -------------------------
# Main simulation runner
# -------------------------
def run_simulation(params: Params, num_params: NumericalParams):
    """Run time-stepping simulation with fixed source and return final coefficients."""
    nx, ny, nz = num_params.nx, num_params.ny, num_params.nz
    
    # Initial state: a[0] = T0 * sqrt(Lx*Ly*Lz) (Constant T0 modal coefficient)
    a = np.zeros(nx*ny*nz, dtype=np.float64)
    a[0] = params.T0 * np.sqrt(params.Lx*params.Ly*params.Lz)

    # 1. Precompute mode matrices
    x, phi_x = phi_matrix(nx, params.Lx)
    y, phi_y = phi_matrix(ny, params.Ly)
    phi_p0 = phi_p_zero(nz, params.Lz)
    phi_p0_tile = phi_p0[:, None, None]
    num_params.phi_p0_tile = phi_p0_tile
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # 2. Precompute K and KK (Normalization calculated in ONE PLACE)
    p = np.arange(nz)[:, None, None]
    n = np.arange(ny)[None, :, None]
    m = np.arange(nx)[None, None, :]
    
    alpha = params.k / (params.rho * params.Ceff)
    lam = (m * np.pi / params.Lx) ** 2 + (n * np.pi / params.Ly) ** 2 + (p * np.pi / params.Lz) ** 2
    lam_flat = lam.ravel(order='C')

    # K (Homogeneous Decay Factor)
    K = np.ones_like(lam_flat, dtype=np.float64)
    mask_nonzero_lam = lam_flat > 0
    K[mask_nonzero_lam] = np.exp(-alpha * lam_flat * num_params.dt)[mask_nonzero_lam]

    # C Factors (Normalization)
    delta_m0 = (m == 0).astype(np.float64)
    delta_n0 = (n == 0).astype(np.float64)
    delta_p0 = (p == 0).astype(np.float64)

    Cm_sqrt = np.sqrt(2.0 - delta_m0) / np.sqrt(params.Lx)
    Cn_sqrt = np.sqrt(2.0 - delta_n0) / np.sqrt(params.Ly)
    Cp_sqrt = np.sqrt(2.0 - delta_p0) / np.sqrt(params.Lz)
    
    # S_factor_3D is the product C_m * C_n * C_p (The full normalization constant)
    S_factor_3D = Cm_sqrt * Cn_sqrt * Cp_sqrt
    S_factor = S_factor_3D.ravel(order='C')
    
    # KK (Fixed Source Forcing Factor)
    KK = np.zeros_like(lam_flat, dtype=np.float64)
    mask_singularity = lam_flat == 0

    # Non-singular case (lambda > 0)
    KK[mask_nonzero_lam] = S_factor[mask_nonzero_lam] * (1.0 - K[mask_nonzero_lam]) / (lam_flat[mask_nonzero_lam] * params.k)
    # Singular case (lambda = 0)
    KK[mask_singularity] = S_factor[mask_singularity] * num_params.dt / (params.rho * params.Ceff)
    
    num_params.K = K
    num_params.KK = KK

    # 3. Precompute the constant source term projection
    q_surface = q_laser_field(X, Y, t=0.0, params=params)
    q_dct_flat = dct_projection(q_surface, params, phi_p0_tile)

    # 4. Time Stepping Loop (Simple explicit update)
    t = 0.0
    while t < num_params.t_final - 1e-12:
        # Time step: a(t+dt) = a(t) * K + KK * Q_source
        a = a * num_params.K + num_params.KK * q_dct_flat
        t += num_params.dt

    # 5. Final Reconstruction
    T_final = reconstruct_temperature_field(a, nx, ny, nz, phi_x, phi_y, phi_p0_tile)
    
    return a, T_final

# -------------------------
# Example Execution
# -------------------------
if __name__ == '__main__':
    # Define parameters (example values)
    params = Params(Lx=0.01, Ly=0.005, Lz=0.01, rho=7900, Ceff=500, k=14, T0=300.0,
                    P=200.0, Absorptivity=0.30, r_b=0.00006)

    num_params = NumericalParams(
        dt=0.0001,
        t_final=0.001,
        nx=512, # Reduced size for fast minimal run
        ny=512,
        nz=16
    )

    print(f"Running minimal simulation with T0={params.T0} K, dt={num_params.dt} s, t_final={num_params.t_final} s.")

    a_final, T_final = run_simulation(params, num_params)
    
    print(f"\nSimulation finished.")
    print(f"Final max temperature Tmax = {T_final.max():.2f} K")

    # Optional visualization
    # import matplotlib.pyplot as plt
    # x, _ = phi_matrix(num_params.nx, params.Lx)
    # y, _ = phi_matrix(num_params.ny, params.Ly)
    # X, Y = np.meshgrid(x * 1e3, y * 1e3) # Convert to mm
    # plt.figure()
    # plt.contourf(X, Y, T_final, levels=50, cmap='inferno')
    # plt.colorbar(label='Temperature (K)')
    # plt.xlabel('x (mm)')
    # plt.ylabel('y (mm)')
    # plt.title('Minimal Final Temperature Field (z=0)')
    # plt.gca().set_aspect('equal')
    # plt.show()