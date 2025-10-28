# corrected_main.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn
from typing import List, Tuple

# Optional fast-backend using pyfftw's scipy-compat interface (if installed)
try:
    import pyfftw
    import pyfftw.interfaces.scipy_fftpack as fftw_scipypack
    HAVE_PYFFTW = True
except Exception:
    HAVE_PYFFTW = False

# -------------------------
# Parameter containers
# -------------------------
class Params:
    def __init__(self,
                 Lx: float, Ly: float, Lz: float,
                 rho: float, Ceff: float, k: float,
                 P: float, r_b: float, x0: float, y0: float, vx: float,
                 DeltaH_LV: float = 2.26e6,
                 R_v: float = 461.5,
                 T_boil: float = 373.15):
        self.Lx = Lx; self.Ly = Ly; self.Lz = Lz
        self.rho = rho; self.Ceff = Ceff; self.k = k
        self.P = P; self.r_b = r_b; self.x0 = x0; self.y0 = y0; self.vx = vx
        self.DeltaH_LV = DeltaH_LV; self.R_v = R_v; self.T_boil = T_boil

class NumericalParams:
    def __init__(self, dt: float, t_final: float, nx: int, ny: int, nz: int,
                 K: np.ndarray = None, KK: np.ndarray = None):
        self.dt = dt; self.t_final = t_final
        self.nx = nx; self.ny = ny; self.nz = nz
        self.K = K; self.KK = KK

# -------------------------
# Eigenfunctions
# -------------------------
def phi_1d(m: int, x: np.ndarray, L: float) -> np.ndarray:
    """Neumann cosine eigenfunction normalized on [0, L]."""
    if m == 0:
        return np.full_like(x, 1.0 / np.sqrt(L))
    return np.sqrt(2.0 / L) * np.cos(m * np.pi * x / L)

def phi_p_at_zero(p: int, Lz: float) -> float:
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
    A = 0.005 / np.sqrt(2.0 * np.pi * params.R_v)
    exponent = (params.DeltaH_LV / (params.R_v * params.T_boil)) * (1.0 - params.T_boil / T)
    return A * np.exp(exponent)

# -------------------------
# Modes enumerator
# -------------------------
def make_modes(M:int, N:int, P:int) -> List[Tuple[int,int,int]]:
    return [(m,n,p) for p in range(P) for n in range(N) for m in range(M)]

# -------------------------
# Fast DCT
# -------------------------
def dct2_heatflux_scipy(q: np.ndarray, params: Params, P_modes: int) -> np.ndarray:
    """
    Fast SciPy DCT-II-based routine.
    Returns a flattened vector of length M*N*P (ordered p, n, m) matching make_modes().
    """
    # q shape: (Mx, Ny) where Mx = nx, Ny = ny
    Nx, Ny = q.shape
    prefactor = 2*params.P/(np.pi*params.r_b**2)*np.sqrt((params.Lx*params.Ly)/(Nx*Ny))

    # 2D DCT-II (type=2) orthonormal
    q_dct = dctn(q, type=2, norm='ortho')   # shape (Mx, Ny); axis 0 -> m, axis 1 -> n
    # allocate S in shape (P, N, M) so that flatten(C-order) yields iterate p,n,m (m fastest)
    S_pnm = np.empty((P_modes, Ny, Nx), dtype=np.float64)
    for p in range(P_modes):
        phi_p0 = phi_p_at_zero(p, params.Lz)
        # all m,n modes share same q_dct; multiply by phi_p(0) and prefactor
        S_mn = phi_p0 * prefactor * q_dct
        # special-case for (m,n)=(0,0): factor 2P/(pi r_b^2) instead of 4P... => multiply by 0.5
        S_mn0 = S_mn.copy()
        S_mn0[0,0] *= 0.5
        # store transposed into S_pnm[p, n, m] (so p,n,m ordering)
        S_pnm[p, :, :] = S_mn0.T  # S_mn shape (Mx, Ny) -> transpose to (Ny, Mx)
    # flatten in C-order to get vector consistent with make_modes
    return S_pnm.ravel(order='C')  # length Mx*Ny*P_modes


# -------------------------
# Reconstruction
# -------------------------
def reconstruct_temperature_field(a: np.ndarray, modes: List[Tuple[int,int,int]],
                                  params: Params, Xg: np.ndarray, Yg: np.ndarray) -> np.ndarray:
    """
    Reconstruct T(x,y,z=0) from modal coefficients a.
    Vectorized and correct ordering of modes.
    """
    # Xg, Yg shapes: (ny, nx) because we'll create meshgrid with x, y (see run_simulation)
    ny, nx = Xg.shape
    T = np.zeros_like(Xg, dtype=np.float64)

    # Precompute phi_1d for all m and for x-array, and for n and y-array to speed up
    max_m = max(m for (m,_,_) in modes) if modes else 0
    max_n = max(n for (_,n,_) in modes) if modes else 0
    max_p = max(p for (_,_,p) in modes) if modes else 0

    # precompute basis in x and y
    x_coords = Xg[0, :]   # Xg's row 0 holds x-values (since meshgrid(x, y) yields Xg with shape (ny, nx))
    y_coords = Yg[:, 0]   # Yg's col 0 holds y-values

    Phi_x = [phi_1d(m, x_coords, params.Lx) for m in range(max_m + 1)]  # each length nx
    Phi_y = [phi_1d(n, y_coords, params.Ly) for n in range(max_n + 1)]  # each length ny

    # sum contributions
    for ai, (m,n,p) in zip(a, modes):
        # phi_x shape (nx,), phi_y shape (ny,)
        # build outer product phi_y[:,None] * phi_x[None,:] to match T shape (ny, nx)
        T += ai * (Phi_y[n][:, None] * Phi_x[m][None, :]) * phi_p_at_zero(p, params.Lz)

    return T

# -------------------------
# Update coefficients
# -------------------------
def update_coefficients(a: np.ndarray, modes: List[Tuple[int,int,int]],
                        params: Params, num_params: NumericalParams,
                        Xg: np.ndarray, Yg: np.ndarray, t: float) -> np.ndarray:
    """
    Evaluate q_field, compute modal forcing vector (flattened p,n,m), and update a using
    a_new = K * a + KK * S (element-wise).
    """
    # reconstruct T at current a
    T_field = reconstruct_temperature_field(a, modes, params, Xg, Yg)
    q_field = q_laser_field(Xg, Yg, t, params) - q_evap_point(T_field, params)  # shape (ny, nx)

    # note: our DCT implementations expect (Mx, Ny) = (nx, ny) orientation,
    # but Xg/Yg are (ny, nx) so transpose q_field
    q_field_for_dct = q_field.T  # now shape (nx, ny) consistent with earlier assumption

    S_flat = dct2_heatflux_fftw(q_field_for_dct, params, P_modes=num_params.nz)  # length M*N*P

    # a, K, KK are all length len(modes)
    a_new = a * num_params.K + num_params.KK * S_flat
    return a_new

# -------------------------
# Simulation driver
# -------------------------
def run_simulation(params: Params, num_params: NumericalParams):
    nx, ny, nz = num_params.nx, num_params.ny, num_params.nz
    modes = make_modes(nx, ny, nz)  # ordering p, n, m
    Ncoeff = len(modes)

    # initial coefficients
    a = np.zeros(Ncoeff, dtype=np.float64)
    # initial temperature 300K -> corresponds to m=n=p=0 mode coefficient = 300 * sqrt(Lx*Ly*Lz?) 
    # Here we keep same approach as before: set a[0] = 300
    a[0] = 300.0

    dt = num_params.dt
    t_final = num_params.t_final

    # precompute K and KK aligned with modes ordering (p,n,m)
    K = np.zeros(Ncoeff, dtype=np.float64)
    KK = np.zeros(Ncoeff, dtype=np.float64)
    for idx, (m,n,p) in enumerate(modes):
        lambda_mnp = ((m * np.pi / params.Lx) ** 2 +
                      (n * np.pi / params.Ly) ** 2 +
                      (p * np.pi / params.Lz) ** 2)
        if lambda_mnp == 0.0:
            # avoid division by zero for the (0,0,0) mode: in many physical problems this mode is handled differently
            K[idx] = 1.0
            KK[idx] = 0.0
        else:
            K[idx] = np.exp(-params.k / (params.rho * params.Ceff) * lambda_mnp * dt)
            KK[idx] = (1.0 - K[idx]) / (params.rho * params.Ceff * lambda_mnp)
    num_params.K = K
    num_params.KK = KK

    # build grid at cell centers (x has length nx, y has length ny)
    dx = params.Lx / nx
    dy = params.Ly / ny
    x = np.linspace(dx/2, params.Lx - dx/2, nx)
    y = np.linspace(dy/2, params.Ly - dy/2, ny)

    # IMPORTANT: meshgrid(x, y) -> Xg, Yg shapes (ny, nx)
    Xg, Yg = np.meshgrid(x, y)  # default indexing='xy' -> Xg shape (ny, nx)
    # note: when we feed data to the DCT functions we transpose to (nx, ny) as needed

    t = 0.0
    while t < t_final - 1e-12:
        a = update_coefficients(a, modes, params, num_params, Xg, Yg, t)
        t += dt

    return a, modes, Xg, Yg

# -------------------------
# Example parameters & run
# -------------------------

params = Params(Lx =0.001, Ly=0.001, Lz=0.001,
                rho=7800, Ceff=500, k=50,
                P=50.0, r_b=0.00006, x0=0.0005, y0=0.0005, vx=0.0001)

num_params = NumericalParams(dt=0.05, t_final=0.20, nx=128, ny=128, nz=8)

a_final, modes, Xg, Yg = run_simulation(params, num_params)

# reconstruct and plot
T_final = reconstruct_temperature_field(a_final, modes, params, Xg, Yg)
plt.figure(figsize=(6,5))
plt.contourf(Xg*1e3, Yg*1e3, T_final, levels=50, cmap='hot')
plt.colorbar(label='Temperature (K)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Final Temperature Field at z=0')
plt.show()
