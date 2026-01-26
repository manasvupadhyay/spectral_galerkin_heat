import numpy as np
import pyfftw
import os
from types import SimpleNamespace


import implementations.physics.spectral_cpu_kernels as kernels
import utils.spectral_helpers as spec_hp
import utils.helpers as hp
import implementations.file_io.spectral_fs_io as spectral_fs_io
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



# ============================================================
#   FUNCTIONS
# ============================================================


def time_step(a, phys, num, geom, laser, SsState, epsilon=2e+1, iter_step=0):
    # 1. Compute source terms (Laser + Evaporation)
    # [FIX 1] Use Kernel for Laser Flux (Unpack SsState.X/Y and Laser params)
    q_las = kernels.compute_gaussian_laser_flux(
        SsState.X, SsState.Y, 
        laser.x, laser.y, 
        laser.r_b, laser.laser_coef
    )
    
    # Initialize buffers if not exist
    if not hasattr(SsState, 'q_evap_buffer'):
        SsState.q_evap_buffer = np.zeros_like(q_las)
        
    q_evap = hp.shift_flux(SsState.q_evap_old, (laser.v[0]*num.dt, laser.v[1]*num.dt), geom)
    
    # [FIX] Use spec_hp.DCT_II
    q_dct = spec_hp.DCT_II(q_las - q_evap)
    
    # 2. Linear step (ETD1)
    np.multiply(a, SsState.K, out=SsState.aK, casting='same_kind')
    S_n = SsState.dct_scale * q_dct
    np.multiply(SsState.dct_scale, q_dct, out=SsState.B_buffer, casting='same_kind') 
    
    # [KERNEL SUBCALL] Replaced hp.compute_a_temp_numba
    kernels.update_modes_etd1(SsState.aK, SsState.KK_by_Cp, SsState.B_buffer, SsState.a_temp)
    
    S_current = S_n.copy()

    # 3. Latent Heat Correction
    # [FIX] Use spec_hp.update_fine_mesh (assuming you moved it there as planned)
    spec_hp.update_fine_mesh(SsState, laser) 
    SsState.Q_latent_buffer.fill(0.0)
    kernels.compute_latent_heat_source(SsState.Q_latent_buffer, phys, laser, geom, num, SsState)

    # [FIX 2] Use Kernel for Box Projection
    Q_modes = kernels.project_box_to_modes(SsState.Q_latent_buffer, SsState)

    kernels.add_source_term_modes(SsState.aK, SsState.KK, Q_modes)
    kernels.add_source_term_modes(SsState.a_temp, SsState.KK, Q_modes)
    
    # 4. Nonlinear iteration for evaporation
    # [FIX 3] Use kernel for Surface Reconstruction
    T_temp = kernels.reconstruct_surface_temperature(SsState.a_temp, SsState)
    
    for k in range(30):
        kernels.compute_evaporation_flux(
            T_temp, SsState.q_evap_buffer,
            phys.Pa, phys.R_v, phys.T_boil, phys.DeltaH_LV, phys.R_v, phys.T_liquidus
        )
        q_evap = SsState.q_evap_buffer 
        
        np.subtract(q_las, q_evap, out=SsState.q_diff, casting='same_kind')
        # [FIX] Use spec_hp.DCT_II
        S_target = SsState.dct_scale * spec_hp.DCT_II(SsState.q_diff)
        S_current = 0.1 * S_target + 0.9 * S_current 
        np.multiply(1.0, S_current, out=SsState.B_buffer, casting='same_kind')
        
        kernels.update_modes_etd1(SsState.aK, SsState.KK_by_Cp, SsState.B_buffer, SsState.a_temp)
        
        T_old = T_temp
        # [FIX 3] Use kernel for Surface Reconstruction (Inside Loop)
        T_temp = kernels.reconstruct_surface_temperature(SsState.a_temp, SsState)
        
        if np.max(np.abs(T_temp - T_old)) < epsilon: break
    
    SsState.q_evap_old = q_evap.astype(np.float32, copy=True)
    P_laser = np.sum(q_las) * geom.dx * geom.dy
    
    # Debug output
    if iter_step % 200 == 0:
        T_box, box_coords = kernels.reconstruct_temperature_box(SsState.a_temp, SsState)
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
    nx, ny, nz =  10, 10, 10 # 512, 256, 1000
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
    SsState = spec_hp.SpectralSolverState()
    SsState.prepare_reconstruction_basis(geom)
    SsState.prepare_K_buffers(phys, geom, num)
    # Run Simulation
    a, T_hist, P_hist = run_simulation(phys, num, geom, laser, SsState)

    # Save full 3D field for Paraview
    print("Reconstructing full volume...")
    T_volume = spec_hp.reconstruct_temperature_volume(a, SsState)
    volume_base = f"{OUT_DIR}/T_volume_final"
    
    # [FIX] Use fs_io.save_field_to_hdf5
    spectral_fs_io.save_field_to_hdf5(
        volume_base, 
        T_volume, 
        (SsState.x, SsState.y, SsState.z), 
        value_name="Temperature", 
        geom=geom, 
        verbose=True
    )

    laser_snapshot = SimpleNamespace(
        x=laser.x - laser.v[0] * num.dt,
        y=laser.y - laser.v[1] * num.dt,
        x0=laser.x0,  # Add missing attributes
        y0=laser.y0,
        r_b=laser.r_b,
        v=laser.v
    )
    spec_hp.save_temp_profiles(a, num, geom, SsState, laser=laser_snapshot, center="laser")
