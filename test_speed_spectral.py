import numpy as np
import pyfftw
import os
from types import SimpleNamespace
from dataclasses import dataclass

import helpers as hp
# Import the new CPU kernels
import implementations.physics.spectral_cpu_kernels as kernels

pyfftw.config.NUM_THREADS = os.cpu_count()
OUT_DIR = "out"


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
        self.laser_coef = self.Absorptivity * 2.0 * self.P / (np.pi * self.r_b ** 2)
        
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

@dataclass
class SpectralSolverState:
    """
    Encapsulates solver-specific buffers and precomputed grids needed for the spectral method.
    Keeps global GeomParams/NumParams clean from implementation details.
    """
    # Spectral Propagators
    K: np.ndarray = None
    KK: np.ndarray = None
    KK_by_Cp: np.ndarray = None
    
    # State Arrays
    a: np.ndarray = None       # Current temperature modes (nz, ny, nx)
    aK: np.ndarray = None      # Decayed temperature modes (nz, ny, nx)
    a_temp: np.ndarray = None  # Temporary working array (nz, ny, nx)
    
    # Buffers
    q_evap_old: np.ndarray = None
    q_diff: np.ndarray = None
    B_buffer: np.ndarray = None
    
    # Latent Heat Specifics
    Q_latent_buffer: np.ndarray = None
    isotherm_cache: list = None
    
        
    # Fine Grid / Aliasing structures
    # Box Coordinate Arrays (Changing every step)
    box_x: np.ndarray = None
    box_y: np.ndarray = None
    box_z: np.ndarray = None
    
    # Active Box Basis Subsets (Changing every step)
    Bx_fine: np.ndarray = None 
    By_fine: np.ndarray = None
    Bz_fine: np.ndarray = None
    
    # Precomputed Full Fine Bases (Static, but needed for slicing)
    Bx_fine_full: np.ndarray = None
    By_fine_full: np.ndarray = None
    Bz_fine_full: np.ndarray = None
    
    # Box State Descriptors
    ix_laser_box: int = 0
    nx_box: int = 0
    ny_box: int = 0
    nz_box: int = 0
    
    fine_mesh_initialized: bool = False

    def prepare_reconstruction_basis(self, geom):
        # Global mesh coordinates (Cell-Centered to match the definition of DCT-II)
        dx, dy, dz = geom.dx, geom.dy, geom.dz
        nx, ny, nz = geom.nx, geom.ny, geom.nz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
        self.x = ((np.arange(nx) + 0.5) * dx).astype(np.float32)
        self.y = ((np.arange(ny) + 0.5) * dy).astype(np.float32)
        self.z = ((np.arange(nz) + 0.5) * dz).astype(np.float32)
        x_np, y_np, z_np = self.x, self.y, self.z
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        self.X, self.Y = self.X.astype(np.float32), self.Y.astype(np.float32)
        """Precompute reconstruction bases and normalization coefficients."""
        print("Precomputing reconstruction bases...")
        # Normalization coefficients
        self.Cm = hp.C_coef(nx, Lx)
        self.Cn = hp.C_coef(ny, Ly)
        self.Cp = hp.C_coef(nz, Lz)
        self.Cp32 = self.Cp.astype(np.float32)
        
        # Scaling factors for DCT/IDCT
        self.dct_scale = np.float32((dx * dy) * np.sqrt((nx * ny) / (Lx * Ly)))
        self.recon_scale = np.float32(np.sqrt(nx * ny) / np.sqrt(Lx * Ly))

        # Precomputed cosine bases for reconstruction
        self.cos_mx = np.cos(np.pi * np.arange(nx)[:, None] * x_np[None, :] / Lx).astype(np.float32)
        self.cos_ny = np.cos(np.pi * np.arange(ny)[:, None] * y_np[None, :] / Ly).astype(np.float32)
        self.cos_pz = np.cos(np.pi * np.arange(nz)[:, None] * z_np[None, :] / Lz).astype(np.float32)

        # Fine mesh setup for latent heat correction
        self.refinement = 4
        self.Lx_box, self.Ly_box, self.Lz_box = 0.7e-3, 0.2e-3, 0.04e-3
        self.dx_fine, self.dy_fine, self.dz_fine = dx/self.refinement, dy/self.refinement, dz/self.refinement
        
        self.nx_fine_total = int(np.ceil(Lx / self.dx_fine))
        self.ny_fine_total = int(np.ceil(Ly / self.dy_fine))
        self.nz_fine_total = int(np.ceil(self.Lz_box / self.dz_fine))
        
        x_fine = ((np.arange(self.nx_fine_total) + 0.5) * self.dx_fine).astype(np.float32)
        y_fine = ((np.arange(self.ny_fine_total) + 0.5) * self.dy_fine).astype(np.float32)
        z_fine = ((np.arange(self.nz_fine_total) + 0.5) * self.dz_fine).astype(np.float32)
        self.x_fine = x_fine
        self.y_fine = y_fine
        self.z_fine = z_fine
        
        print("Precomputing fine cosine bases...")
        m, n, p = np.arange(nx), np.arange(ny), np.arange(nz)
        self.Bx_fine_full = (self.Cm[:, None] * np.cos(np.pi * m[:, None] * x_fine[None, :] / Lx)).astype(np.float32)
        self.By_fine_full = (self.Cn[:, None] * np.cos(np.pi * n[:, None] * y_fine[None, :] / Ly)).astype(np.float32)
        self.Bz_fine_full = (self.Cp[:, None] * np.cos(np.pi * p[:, None] * z_fine[None, :] / Lz)).astype(np.float32)
        
        # Box dimensions in fine grid points
        self.nx_box = int(np.ceil(self.Lx_box / self.dx_fine))
        self.ny_box = int(np.ceil(self.Ly_box / self.dy_fine))
        self.nz_box = self.nz_fine_total
        
        # Preallocated arrays for fine mesh box
        self.Bx_fine = np.zeros((nx, self.nx_box), dtype=np.float32)
        self.By_fine = np.zeros((ny, self.ny_box), dtype=np.float32)
        self.Bz_fine = self.Bz_fine_full[:, :self.nz_box]
        self.box_x = np.zeros(self.nx_box, dtype=np.float32)
        self.box_y = np.zeros(self.ny_box, dtype=np.float32)
        self.box_z = np.linspace(0.0, self.Lz_box, self.nz_box, dtype=np.float32)
        self.fine_mesh_initialized = False
        self.ix_laser_box = max(0, min(self.nx_box - 1, int(round(0.5 * (self.nx_box - 1)))))

    def prepare_K_buffers(self, phys, geom, num):
        """Precompute spectral propagators and allocate buffers.
        """

        print(f"Precomputing K, KK... ")
        self.K, self.KK = hp.precompute_K_KK(phys, num, geom)
        self.KK_by_Cp = (self.KK * self.Cp[:, None, None]).astype(np.float32) # Projected on x,y plane
        nx, ny, nz = num.nx, num.ny, num.nz
        # Allocate working arrays
        self.q_diff = np.empty((ny, nx), dtype=np.float32)
        self.B_buffer = np.empty((ny, nx), dtype=np.float32)
        self.a_temp = np.empty((nz, ny, nx), dtype=np.float32)
        self.aK = np.empty((nz, ny, nx), dtype=np.float32)
        self.q_evap_old = np.zeros((ny, nx), dtype=np.float32)
        self.Q_latent_buffer = None
        # ZYX layout for contiguous X-scanning
        self.Q_latent_buffer = np.zeros((self.nz_box, self.ny_box, self.nx_box), dtype=np.float32)

# ============================================================
#   FUNCTIONS
# ============================================================


def time_step(a, phys, num, geom, laser, SsState, epsilon=2e+1, iter_step=0):
    # 1. Compute source terms (Laser + Evaporation)
    q_las = hp.q_laser(SsState, laser)
    
    # Initialize buffers if not exist
    if not hasattr(SsState, 'q_evap_buffer'):
        SsState.q_evap_buffer = np.zeros_like(q_las)
        
    q_evap = hp.shift_flux(SsState.q_evap_old, (laser.v[0]*num.dt, laser.v[1]*num.dt), geom)
    q_dct = hp.DCT_II(q_las - q_evap)
    
    # 2. Linear step (ETD1)
    np.multiply(a, SsState.K, out=SsState.aK, casting='same_kind')
    S_n = SsState.dct_scale * q_dct
    np.multiply(SsState.dct_scale, q_dct, out=SsState.B_buffer, casting='same_kind') 
    
    # [KERNEL SUBCALL] Replaced hp.compute_a_temp_numba
    kernels.update_modes_etd1(SsState.aK, SsState.KK_by_Cp, SsState.B_buffer, SsState.a_temp)
    
    S_current = S_n.copy()

    # 3. Latent Heat Correction (Volumetric Source) - MOVED BEFORE EVAPORATION
    hp.update_fine_mesh(SsState, laser) 
    
    # Compute volumetric source
    SsState.Q_latent_buffer.fill(0.0)
    
    # [KERNEL Call] High-level orchestrator for latent heat
    kernels.compute_latent_heat_source(SsState.Q_latent_buffer, phys, laser, geom, num, SsState)

    # Convert to modes
    Q_modes = hp.box_field_to_modes(SsState.Q_latent_buffer, SsState)

    # Add to temperature modes
    kernels.add_source_term_modes(SsState.aK, SsState.KK, Q_modes)
    # Update current a_temp so that the start of evaporation loop sees it
    kernels.add_source_term_modes(SsState.a_temp, SsState.KK, Q_modes)
    
    # 4. Nonlinear iteration for evaporation
    T_temp = hp.reconstruct_temperature_top(SsState.a_temp, SsState)
    for k in range(30):
        
        # compute_evaporation_flux (IN-PLACE)
        # We use the buffer created earlier
        kernels.compute_evaporation_flux(
            T_temp, SsState.q_evap_buffer,
            phys.Pa, phys.R_v, phys.T_boil, phys.DeltaH_LV, phys.R_v, phys.T_liquidus
        )
        q_evap = SsState.q_evap_buffer # Alias for readability
        
        np.subtract(q_las, q_evap, out=SsState.q_diff, casting='same_kind')
        S_target = SsState.dct_scale * hp.DCT_II(SsState.q_diff)
        S_current = 0.1 * S_target + 0.9 * S_current # Relaxation
        np.multiply(1.0, S_current, out=SsState.B_buffer, casting='same_kind')
        
        # [KERNEL SUBCALL]
        kernels.update_modes_etd1(SsState.aK, SsState.KK_by_Cp, SsState.B_buffer, SsState.a_temp)
        
        T_old = T_temp
        T_temp = hp.reconstruct_temperature_top(SsState.a_temp, SsState)
        if np.max(np.abs(T_temp - T_old)) < epsilon: break
    
    SsState.q_evap_old = q_evap.astype(np.float32, copy=True)
    P_laser = np.sum(q_las) * geom.dx * geom.dy
    
    # Debug output
    if iter_step % 200 == 0:
        T_box, box_coords = hp.reconstruct_temperature_box(SsState.a_temp, SsState)
        hp.save_field_to_hdf5(
            f"{OUT_DIR}/T_box_step_{laser.t:.5f}",
            T_box, # Already (nz, ny, nx)
            box_coords,
            value_name="Temperature",
            geom=geom,
        )
        hp.save_field_to_hdf5(
            f"{OUT_DIR}/Q_latent_step_{laser.t:.5f}",
            SsState.Q_latent_buffer, # Already (nz, ny, nx)
            box_coords,
            value_name="LatentHeat",
            geom=geom,
        )
    
    return SsState.a_temp, T_temp, P_laser, k+1

def run_simulation(phys, num, geom, laser, SsState):
    """Main simulation loop."""
    
    a = np.zeros((num.nz, num.ny, num.nx), dtype=np.float32)
    a[0,0,0] = phys.T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz) # Initial condition (mean T)
    
    nsteps = int(np.ceil(num.t_final / num.dt))
    T_top_hist, P_laser_hist = [], []
    
    for step in range(nsteps + 1):
        a, T_top, P_laser, n_evap = time_step(a, phys, num, geom, laser, SsState, iter_step=step)
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
    nx, ny, nz =  512, 256, 1000
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
    SsState = SpectralSolverState()
    SsState.prepare_reconstruction_basis(geom)
    SsState.prepare_K_buffers(phys, geom, num)
    hp.check_resolution(laser, num, geom)
    # Running simulation
    a, T_hist, P_hist = run_simulation(phys, num, geom, laser, SsState)

    T_volume = hp.reconstruct_temperature_volume(a, SsState)
    volume_base = f"{OUT_DIR}/T_volume_final"
    hp.save_field_to_hdf5(
        volume_base,
        T_volume.transpose(2, 1, 0),
        (SsState.x, SsState.y, SsState.z),
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
    hp.save_temp_profiles_fine(a, num, geom, SsState, laser=laser_snapshot, center="laser")
    hp.save_temp_profiles(a, num, geom, SsState, laser=laser_snapshot, center="laser")