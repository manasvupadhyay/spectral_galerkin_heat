import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn
import os

# =========================================================
# Clean previous outputs
# =========================================================
for f in os.listdir('.'):
    if f.startswith(('temperature_field_', 'q_laser_field_', 'q_evap_field_')):
        os.remove(f)

# =========================================================
# Parameter containers
# =========================================================
class Params:
    def __init__(self, Lx, Ly, Lz, rho, Ceff, k, P, r_b, x0, y0, vx,
                 DeltaH_LV=6.0e6, R_v=150.0, T_boil=2800.0):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.rho, self.Ceff, self.k = rho, Ceff, k
        self.P, self.r_b, self.x0, self.y0, self.vx = P, r_b, x0, y0, vx
        self.DeltaH_LV, self.R_v, self.T_boil = DeltaH_LV, R_v, T_boil


class NumericalParams:
    def __init__(self, dt, t_final, nx, ny, nz):
        self.dt, self.t_final = dt, t_final
        self.nx, self.ny, self.nz = nx, ny, nz
        self.K = None
        self.KK = None


# =========================================================
# Eigenfunctions
# =========================================================
def phi_matrix(nx, L):
    """Return matrix of size (nx, nx): phi[m, i] = phi_m(x_i)."""
    x = np.linspace(L / (2 * nx), L - L / (2 * nx), nx)
    m = np.arange(nx).reshape(-1, 1)
    pref = np.sqrt(2.0 / L)
    phi = pref * np.cos(m * np.pi * x / L)
    phi[0, :] = 1.0 / np.sqrt(L)
    return x, phi


def phi_p_zero(nz, Lz):
    """Vector of phi_p(0) values for all p."""
    phi0 = np.sqrt(2.0 / Lz) * np.ones(nz)
    phi0[0] = 1.0 / np.sqrt(Lz)
    return phi0


# =========================================================
# Heat sources
# =========================================================
def q_laser_field(X, Y, t, params):
    rb, P = params.r_b, params.P
    x0t = params.x0 + params.vx * t
    exp_term = np.exp(-2 * ((X - x0t) ** 2 + (Y - params.y0) ** 2) / rb**2)
    return (2 * P / (np.pi * rb**2)) * exp_term


def q_evap_point(T, params):
    A = 0.005 / np.sqrt(2.0 * np.pi * params.R_v)
    T_safe = np.maximum(T, params.T_boil)
    exponent = (params.DeltaH_LV / (params.R_v * params.T_boil)) * (1.0 - params.T_boil / T_safe)
    q_evap = A * np.exp(exponent)
    q_evap[T < params.T_boil] = 0.0
    return q_evap


# =========================================================
# DCT-based projection
# =========================================================
def dct2_heatflux(q, params, num_params, phi_p0):
    """
    Vectorized: 2D DCT + broadcast along z modes.
    Returns flattened S_pnm array in p,n,m order.
    """
    Nx, Ny = q.shape
    pref = np.sqrt(params.Lx * params.Ly) / np.sqrt(Nx * Ny)

    # 2D DCT-II (orthonormal) once
    q_dct = dctn(q, type=2, norm='ortho', workers=-1)

    # broadcast phi_p0[:, None, None]
    S = (phi_p0[:, None, None] * pref * q_dct[None, :, :]).astype(np.float64)

    # flatten p,n,m order directly
    return S.reshape(-1, order='C')


# =========================================================
# Temperature reconstruction (vectorized)
# =========================================================
def reconstruct_temperature_field(a, nx, ny, nz, phi_x, phi_y, phi_p0):
    """
    Fully vectorized reconstruction using tensor contraction:
    T_ij = sum_{m,n,p} a_mnp * phi_y[n,i] * phi_x[m,j] * phi_p0[p]
    """
    # reshape coefficient vector a to (nz, ny, nx)
    A = a.reshape((nz, ny, nx), order='C')
    # tensor contraction: sum over m,n,p
    # result shape (ny, nx)
    T = np.tensordot(phi_p0, A, axes=(0, 0))           # (ny, nx)
    T = np.tensordot(phi_y, T, axes=(0, 0))            # (ny, nx)
    T = np.tensordot(phi_x, T, axes=(0, 0)).T          # (ny, nx)
    return T


# =========================================================
# Coefficient update
# =========================================================
def update_coefficients(a, params, num_params, X, Y, phi_x, phi_y, phi_p0, t):
    # reconstruct field at z=0
    T_field = reconstruct_temperature_field(a, num_params.nx, num_params.ny,
                                            num_params.nz, phi_x, phi_y, phi_p0)
    q_field = q_laser_field(X, Y, t, params) - q_evap_point(T_field, params)
    q_dct = dct2_heatflux(q_field, params, num_params, phi_p0)
    return a * num_params.K + num_params.KK * q_dct


# =========================================================
# Main simulation
# =========================================================
def run_simulation(params, num_params):
    nx, ny, nz = num_params.nx, num_params.ny, num_params.nz
    a = np.zeros(nx * ny * nz)
    a[0] = 300.0 * np.sqrt(params.Lx * params.Ly * params.Lz)

    # precompute mode matrices
    x, phi_x = phi_matrix(nx, params.Lx)
    y, phi_y = phi_matrix(ny, params.Ly)
    phi_p0 = phi_p_zero(nz, params.Lz)

    # compute lambda_mnp grid in vectorized form
    m = np.arange(nx)[:, None, None]
    n = np.arange(ny)[None, :, None]
    p = np.arange(nz)[None, None, :]

    lam = (m * np.pi / params.Lx) ** 2 + (n * np.pi / params.Ly) ** 2 + (p * np.pi / params.Lz) ** 2
    lam = np.moveaxis(lam, 2, 0).ravel(order='C')

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
    while t < num_params.t_final - 1e-14:
        a = update_coefficients(a, params, num_params, X, Y, phi_x, phi_y, phi_p0, t)
        t += num_params.dt
    return a, X, Y, phi_x, phi_y, phi_p0


# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    params = Params(Lx=0.005, Ly=0.001, Lz=0.01,
                    rho=7900, Ceff=500, k=15,
                    P=50.0, r_b=6e-5,
                    x0=5e-4, y0=5e-4, vx=0.004)

    num_params = NumericalParams(dt=0.001, t_final=0.01, nx=128, ny=128, nz=8)

    a_final, X, Y, phi_x, phi_y, phi_p0 = run_simulation(params, num_params)

    T_final = reconstruct_temperature_field(a_final, num_params.nx, num_params.ny,
                                            num_params.nz, phi_x, phi_y, phi_p0)
    np.savetxt("T_final.txt", T_final)

    aspect = params.Lx / params.Ly
    plt.figure(figsize=(10, 10 / aspect))
    plt.contourf(X * 1e3, Y * 1e3, T_final, levels=50, cmap="hot")
    plt.colorbar(label="Temperature (K)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Final Temperature Field at z=0")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()
