import numpy as np
import pyfftw
from numba import njit, prange
import os
from types import SimpleNamespace
import helpers as hp

pyfftw.config.NUM_THREADS = os.cpu_count()
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
#  NUMBA KERNELS
# ============================================================

@njit(parallel=True, fastmath=True)
def compute_a_temp_numba(aK, KK_by_Cp, B_scaled, a_temp_out):
    """Update spectral coefficients for ETD1 scheme."""
    nz = aK.shape[0]
    for p in prange(nz):
        a_temp_out[p, :, :] = aK[p, :, :] + KK_by_Cp[p, :, :] * B_scaled

@njit(parallel=True, fastmath=True)
def add_source_term_numba(a_temp, KK, Q_modes):
    """
    Add volumetric source term to temperature modes.
    a_temp += KK * Q_modes
    """
    nz = a_temp.shape[0]
    for p in prange(nz):
        for i in range(a_temp.shape[1]):
            for j in range(a_temp.shape[2]):
                a_temp[p, i, j] += KK[p, i, j] * Q_modes[p, i, j]

# ============================================================
#  CLASSES
# ============================================================

class Laser:
    def __init__(self, P, r_b, x0, y0, v, Absorptivity):
        self.P = P
        self.r_b = r_b
        self.x0, self.y0 = x0, y0
        self.v = np.array(v, dtype=np.float64)
        self.Absorptivity = Absorptivity
        self.x, self.y, self.t = x0, y0, 0.0
        
    def update(self, dt):
        """Update laser position."""
        self.x += self.v[0] * dt
        self.y += self.v[1] * dt
        self.t += dt

class GeomParams:
    def __init__(self, Lx, Ly, Lz, num, phys, laser):
        self.Lx, self.Ly, self.Lz = float(Lx), float(Ly), float(Lz)
        self.nx, self.ny, self.nz = num.nx, num.ny, num.nz
        self.dx, self.dy, self.dz = Lx/num.nx, Ly/num.ny, Lz/num.nz

        # Global mesh coordinates (Cell-Centered to match the definition of DCT-II)
        self.x = ((np.arange(self.nx) + 0.5) * self.dx).astype(np.float32)
        self.y = ((np.arange(self.ny) + 0.5) * self.dy).astype(np.float32)
        self.z = ((np.arange(self.nz) + 0.5) * self.dz).astype(np.float32)
        x_np, y_np, z_np = self.x, self.y, self.z
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        self.X, self.Y = self.X.astype(np.float32), self.Y.astype(np.float32)

        # Normalization coefficients
        self.Cm = hp.C_coef(self.nx, self.Lx)
        self.Cn = hp.C_coef(self.ny, self.Ly)
        self.Cp = hp.C_coef(self.nz, self.Lz)
        self.Cp32 = self.Cp.astype(np.float32)
        
        # Scaling factors for DCT/IDCT
        self.dct_scale = np.float32((self.dx * self.dy) * np.sqrt((self.nx * self.ny) / (self.Lx * self.Ly)))
        self.recon_scale = np.float32(np.sqrt(self.nx * self.ny) / np.sqrt(self.Lx * self.Ly))

        # Precomputed cosine bases for reconstruction
        self.cos_mx = np.cos(np.pi * np.arange(self.nx)[:, None] * x_np[None, :] / self.Lx).astype(np.float32)
        self.cos_ny = np.cos(np.pi * np.arange(self.ny)[:, None] * y_np[None, :] / self.Ly).astype(np.float32)
        self.cos_pz = np.cos(np.pi * np.arange(self.nz)[:, None] * z_np[None, :] / self.Lz).astype(np.float32)

        self.laser_coef = laser.Absorptivity * 2.0 * laser.P / (np.pi * laser.r_b ** 2)
        
        # Fine mesh setup for latent heat correction
        self.refinement = 4
        self.Lx_box, self.Ly_box, self.Lz_box = 0.7e-3, 0.2e-3, 0.04e-3
        # To be implemented later, some time ni the future, check that melpool fits in box
        self.dx_fine, self.dy_fine, self.dz_fine = self.dx/self.refinement, self.dy/self.refinement, self.dz/self.refinement
        
        self.nx_fine_total = int(np.ceil(self.Lx / self.dx_fine))
        self.ny_fine_total = int(np.ceil(self.Ly / self.dy_fine))
        self.nz_fine_total = int(np.ceil(self.Lz_box / self.dz_fine))
        
        x_fine = ((np.arange(self.nx_fine_total) + 0.5) * self.dx_fine).astype(np.float32)
        y_fine = ((np.arange(self.ny_fine_total) + 0.5) * self.dy_fine).astype(np.float32)
        z_fine = ((np.arange(self.nz_fine_total) + 0.5) * self.dz_fine).astype(np.float32)
        self.x_fine = x_fine
        self.y_fine = y_fine
        self.z_fine = z_fine
        
        print("Precomputing fine cosine bases...")
        m, n, p = np.arange(self.nx), np.arange(self.ny), np.arange(self.nz)
        self.Bx_fine_full = (self.Cm[:, None] * np.cos(np.pi * m[:, None] * x_fine[None, :] / self.Lx)).astype(np.float32)
        self.By_fine_full = (self.Cn[:, None] * np.cos(np.pi * n[:, None] * y_fine[None, :] / self.Ly)).astype(np.float32)
        self.Bz_fine_full = (self.Cp[:, None] * np.cos(np.pi * p[:, None] * z_fine[None, :] / self.Lz)).astype(np.float32)
        
        # Box dimensions in fine grid points
        self.nx_box = int(np.ceil(self.Lx_box / self.dx_fine))
        self.ny_box = int(np.ceil(self.Ly_box / self.dy_fine))
        self.nz_box = self.nz_fine_total
        
        # Preallocated arrays for fine mesh box
        self.Bx_fine = np.zeros((self.nx, self.nx_box), dtype=np.float32)
        self.By_fine = np.zeros((self.ny, self.ny_box), dtype=np.float32)
        self.Bz_fine = self.Bz_fine_full[:, :self.nz_box]
        self.box_x = np.zeros(self.nx_box, dtype=np.float32)
        self.box_y = np.zeros(self.ny_box, dtype=np.float32)
        self.box_z = np.linspace(0.0, self.Lz_box, self.nz_box, dtype=np.float32)
        self.fine_mesh_initialized = False
        self.ix_laser_box = max(0, min(self.nx_box - 1, int(round(0.5 * (self.nx_box - 1)))))
        self.check_resolution(phys, laser, num)

    def check_resolution(self, phys, laser, num):
        """Check if spatial and temporal resolutions are sufficient."""
        lambda_max_z = (phys.k / (phys.rho * phys.Ceff)) * (np.pi * num.nz / self.Lz) ** 2
        dx_rb, dy_rb = self.dx / laser.r_b, self.dy / laser.r_b
        v_mag = np.linalg.norm(laser.v)
        v_crit = v_mag / (10 * self.dx / num.dt) if v_mag > 0 else 0
        
        print(f"Resolution: dx/rb={dx_rb:.2f}, dy/rb={dy_rb:.2f}, v_crit={v_crit:.2f}")
        if dx_rb > 0.4 or dy_rb > 0.4: print("WARNING: Spatial resolution insufficient!")
        if v_crit > 1.0: print("WARNING: Laser moves too fast for time step!")

    def update_fine_mesh(self, laser):
        """Update fine mesh box so the laser sits in the middle."""
        target_ix = int(round(0.5 * (self.nx_box - 1)))
        laser_ix_global = int(round(np.clip(laser.x / self.dx_fine, 0.0, self.nx_fine_total - 1)))
        ix_start = laser_ix_global - target_ix
        max_ix_start = max(0, self.nx_fine_total - self.nx_box)
        if ix_start < 0:
            ix_start = 0
        elif ix_start > max_ix_start:
            ix_start = max_ix_start
        ix_end = ix_start + self.nx_box

        self.box_x[:] = self.x_fine[ix_start:ix_end]
        self.Bx_fine.fill(0.0)
        self.Bx_fine[:, :] = self.Bx_fine_full[:, ix_start:ix_end]
        self.ix_laser_box = laser_ix_global - ix_start
        self.ix_laser_box = max(0, min(self.ix_laser_box, self.nx_box - 1))

        half_box = self.ny_box // 2
        laser_iy_global = int(round(np.clip(laser.y / self.dy_fine, 0.0, self.ny_fine_total - 1)))
        iy_start = laser_iy_global - half_box
        max_iy_start = max(0, self.ny_fine_total - self.ny_box)
        if iy_start < 0:
            iy_start = 0
        elif iy_start > max_iy_start:
            iy_start = max_iy_start
        iy_end = iy_start + self.ny_box

        self.box_y[:] = self.y_fine[iy_start:iy_end]
        self.By_fine.fill(0.0)
        self.By_fine[:, :] = self.By_fine_full[:, iy_start:iy_end]

        self.dV_fine = self.dx_fine * self.dy_fine * self.dz_fine
        self.fine_mesh_initialized = True

class PhysParams:
    def __init__(self, rho, Cp, k, L_f=267700.0, DeltaH_LV=7.41e6, R_v=150.774, T0=293.0, 
                 Pa=101325.0, T_boil=3090.0, T_liquidus=1800.0, T_solidus=1700.0):
        self.rho, self.Cp, self.k = rho, Cp, k
        self.L_f, self.Ceff = L_f, Cp
        self.DeltaH_LV, self.R_v, self.T0, self.Pa = DeltaH_LV, R_v, T0, Pa
        self.T_boil, self.T_liquidus, self.T_solidus = T_boil, T_liquidus, T_solidus
        print(f"Material: Cp={Cp}, L_f={L_f}, T_S={T_solidus}, T_L={T_liquidus}")

class NumericalParams:
    def __init__(self, dt, t_final, nx, ny, nz):
        self.dt, self.t_final = dt, t_final
        self.nx, self.ny, self.nz = nx, ny, nz
        self.iter =  0
        self.Q_latent_buffer = None

    def prepare_K_buffers(self, phys, geom):
        """Precompute spectral propagators and allocate buffers."""
        print(f"Precomputing K, KK... ")
        self.K, self.KK, self.K_phi0, self.KK_phi1, self.KKK_phi2 = precompute_K_KK(phys, self, geom)
        self.KK_by_Cp = (self.KK * geom.Cp[:, None, None]).astype(np.float32) # Projected on x,y plane
        self.KK_vol = self.KK.astype(np.float32)
        
        # Allocate working arrays
        self.q_diff = np.empty((self.ny, self.nx), dtype=np.float32)
        self.B_buffer = np.empty((self.ny, self.nx), dtype=np.float32)
        self.a_temp = np.empty((self.nz, self.ny, self.nx), dtype=np.float32)
        self.aK = np.empty((self.nz, self.ny, self.nx), dtype=np.float32)
        self.q_evap_old = np.zeros((self.ny, self.nx), dtype=np.float32)
        # ZYX layout for contiguous X-scanning
        self.Q_latent_buffer = np.zeros((geom.nz_box, geom.ny_box, geom.nx_box), dtype=np.float32)

# ============================================================
#   FUNCTIONS
# ============================================================

def precompute_K_KK(phys, num, geom):
    """Compute spectral propagators (K, KK) for heat equation."""
    m, n, p = np.arange(num.nx)[None, None, :], np.arange(num.ny)[None, :, None], np.arange(num.nz)[:, None, None]
    mu = (m * np.pi / geom.Lx)**2 + (n * np.pi / geom.Ly)**2 + (p * np.pi / geom.Lz)**2
    lambda_j = (phys.k / (phys.rho * phys.Ceff)) * mu

    K = np.exp(-lambda_j * num.dt)
    K[0, 0, 0] = 1.0
    KK = np.zeros_like(K)
    mask = lambda_j > 0
    KK[mask] = (1 - K[mask]) / (phys.rho * phys.Ceff * lambda_j[mask])
    KK[~mask] = num.dt / (phys.rho * phys.Ceff) # Limit for lambda -> 0

    K_phi0, KK_phi1, KKK_phi2 = None, None, None
      
    return K.astype(np.float32), KK.astype(np.float32), K_phi0, KK_phi1, KKK_phi2

def time_step(a, phys, num, geom, laser, epsilon=2e+1, iter_step=0):
    # 1. Compute source terms (Laser + Evaporation)
    q_las = hp.q_laser(geom, laser)
    q_evap = hp.shift_flux(num.q_evap_old, (laser.v[0]*num.dt, laser.v[1]*num.dt), geom)
    q_dct = hp.DCT_II(q_las - q_evap)
    
    # 2. Linear step (ETD1)
    np.multiply(a, num.K, out=num.aK, casting='same_kind')
    S_n = geom.dct_scale * q_dct
    np.multiply(geom.dct_scale, q_dct, out=num.B_buffer, casting='same_kind') 
    compute_a_temp_numba(num.aK, num.KK_by_Cp, num.B_buffer, num.a_temp) # Updated coefficient, temporary
    S_current = S_n.copy()

    # 3. Latent Heat Correction (Volumetric Source) - MOVED BEFORE EVAPORATION
    geom.update_fine_mesh(laser)
    
    # Compute volumetric source
    num.Q_latent_buffer.fill(0.0)
    hp.compute_latent_heat_source(num.Q_latent_buffer, (geom.box_x, geom.box_y, geom.box_z), phys, laser, geom, num)

    # Convert to modes
    Q_modes = hp.box_field_to_modes(num.Q_latent_buffer, geom)

    # Add to temperature modes
    # Update base state aK so that evaporation loop sees the latent heat
    add_source_term_numba(num.aK, num.KK_vol, Q_modes)
    # Update current a_temp so that the start of evaporation loop sees it
    add_source_term_numba(num.a_temp, num.KK_vol, Q_modes)
    
    # 4. Nonlinear iteration for evaporation
    T_temp = hp.reconstruct_temperature_top(num.a_temp, num, geom)
    for k in range(30):
        q_evap = hp.q_evap_point(T_temp, phys)
        np.subtract(q_las, q_evap, out=num.q_diff, casting='same_kind')
        S_target = geom.dct_scale * hp.DCT_II(num.q_diff)
        S_current = 0.1 * S_target + 0.9 * S_current # Relaxation
        np.multiply(1.0, S_current, out=num.B_buffer, casting='same_kind')
        compute_a_temp_numba(num.aK, num.KK_by_Cp, num.B_buffer, num.a_temp)
        T_old = T_temp
        T_temp = hp.reconstruct_temperature_top(num.a_temp, num, geom)
        if np.max(np.abs(T_temp - T_old)) < epsilon: break
    
    num.q_evap_old = q_evap.astype(np.float32, copy=True)
    P_laser = np.sum(q_las) * geom.dx * geom.dy
    
    # Debug output
    if iter_step % 200 == 0:
        T_box, box_coords = hp.reconstruct_temperature_box(num.a_temp, num, geom)
        hp.save_field_to_hdf5(
            f"{OUT_DIR}/T_box_step_{laser.t:.5f}",
            T_box, # Already (nz, ny, nx)
            box_coords,
            value_name="Temperature",
            geom=geom,
        )
        hp.save_field_to_hdf5(
            f"{OUT_DIR}/Q_latent_step_{laser.t:.5f}",
            num.Q_latent_buffer, # Already (nz, ny, nx)
            box_coords,
            value_name="LatentHeat",
            geom=geom,
        )
    
    return num.a_temp, T_temp, P_laser, k+1

def run_simulation(phys, num, geom, laser):
    """Main simulation loop."""
    num.prepare_K_buffers(phys, geom)
    a = np.zeros((num.nz, num.ny, num.nx), dtype=np.float32)
    a[0,0,0] = phys.T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz) # Initial condition (mean T)
    
    nsteps = int(np.ceil(num.t_final / num.dt))
    T_top_hist, P_laser_hist = [], []
    
    for step in range(nsteps + 1):
        a, T_top, P_laser, n_evap = time_step(a, phys, num, geom, laser, iter_step=step)
        laser.update(num.dt)
        print(f"Step {step}/{nsteps} | t={step*num.dt:.6e}s | T_laser: {T_top[int(laser.y / geom.dy), int(laser.x / geom.dx)]:.2f} K  | P: {P_laser:.3f} W | Evap: {n_evap} ")
        T_top_hist.append(T_top)
        P_laser_hist.append(P_laser)

    return a, T_top_hist, P_laser_hist

# ============================================================
#   MAIN
# ============================================================


if __name__ == "__main__":
    # Simulation parameters
    Lx, Ly, Lz = 0.01, 0.005, 0.0025
    nx, ny, nz = 512, 256, 2000
    dt = 6.0e-6
    t_final = 1.2e-2
    # Material properties
    rho = 7850.0
    Cp = 500.0
    k = 15.0
    phys = PhysParams(rho, Cp, k, T0=293.0)
    
    # Laser
    P = 200.0
    r_b = 60e-6
    v = [0.8, 0.0, 0.0]
    Absorptivity = 0.3
    # x0=2.0e-3, y0=0.0025 (centered in Y)
    laser = Laser(P, r_b, 0.0, 0.0025, v, Absorptivity)
    
    num = NumericalParams(dt, t_final, nx, ny, nz)
    geom = GeomParams(Lx, Ly, Lz, num, phys, laser)
    
    a, T_hist, P_hist = run_simulation(phys, num, geom, laser)

    T_volume = hp.reconstruct_temperature_volume(a, num, geom)
    volume_base = f"{OUT_DIR}/T_volume_final"
    hp.save_field_to_hdf5(
        volume_base,
        T_volume.transpose(2, 1, 0),
        (geom.x, geom.y, geom.z),
        value_name="Temperature",
        geom=geom,
    )

    laser_snapshot = SimpleNamespace(
        x=laser.x - laser.v[0] * num.dt,
        y=laser.y - laser.v[1] * num.dt,
        x0=laser.x0,  # Add missing attributes
        y0=laser.y0,
        r_b=laser.r_b,
        v=laser.v
    )
    hp.save_temp_profiles_fine(a, num, geom, laser=laser_snapshot, center="laser")
    hp.save_temp_profiles(a, num, geom, phys, laser=laser_snapshot, center="laser")
