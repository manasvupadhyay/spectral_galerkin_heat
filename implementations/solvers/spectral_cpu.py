import numpy as np
import implementations.physics.spectral_cpu_kernels as kernels
import utils.spectral_helpers as spec_hp
import utils.helpers as hp

class SpectralSolverCPU:
    def __init__(self, context):
        self.context = context
        self.state = None
        self.laser = getattr(context, 'laser', None)
        
    def initialize(self):
        geom = self.context.geom
        num = self.context.num
        mat = self.context.mat  # Use mat instead of phys
        laser = getattr(self.context, 'laser', None)

        # Initialize spectral solver state
        self.state = spec_hp.SpectralSolverState()
        self.state.prepare_reconstruction_basis(geom)
        self.state.prepare_K_buffers(mat, geom, num)

        # Initial condition: mean T in mode (0,0,0)
        self.state.a = np.zeros((num.nz, num.ny, num.nx), dtype=np.float32)
        # Use T0 if present, else default to 293.0
        T0 = getattr(mat, 'T0', 293.0)
        self.state.a[0,0,0] = T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz)

        return self.state

    def step(self, t, dt, state):
        # Advance the spectral solution by one time step, following test_speed_spectral.py logic
        geom = self.context.geom
        num = self.context.num
        mat = self.context.mat
        laser_path = getattr(self.context, 'laser_path', None)
        laser_params = getattr(self.context, 'laser', None)
        SsState = self.state
        a = SsState.a
        # Fetch laser state at current time
        laser_state = laser_path.get_state(t, dt)
        x = laser_state.x
        y = laser_state.y
        power = laser_state.power
        is_on = laser_state.is_on
        # Use configured radius and absorptivity from laser_params if available
        r_b = laser_params.radius
        absorptivity = laser_params.absorptivity
        laser_coef = absorptivity * 2.0 * power / (np.pi * r_b ** 2) if is_on else 0.0

        # 1. Compute source terms (Laser + Evaporation)
        q_las = kernels.compute_gaussian_laser_flux(
            SsState.X, SsState.Y,
            x, y,
            r_b, laser_coef
        )
        if not hasattr(SsState, 'q_evap_buffer') or SsState.q_evap_buffer is None:
            SsState.q_evap_buffer = np.zeros_like(q_las)

        # Use velocity from laser_state (from GCodeLaserPath)
        v_x, v_y = laser_state.v if hasattr(laser_state, 'v') else (0.0, 0.0)

            
        q_evap = hp.shift_flux(SsState.q_evap_old, (v_x*num.dt, v_y*num.dt), geom)
        q_dct = spec_hp.DCT_II(q_las - q_evap)

        # 2. Linear step (ETD1)
        np.multiply(a, SsState.K, out=SsState.aK, casting='same_kind')
        S_n = SsState.dct_scale * q_dct
        np.multiply(SsState.dct_scale, q_dct, out=SsState.B_buffer, casting='same_kind')
        kernels.update_modes_etd1(SsState.aK, SsState.KK_by_Cp, SsState.B_buffer, SsState.a_temp)
        S_current = S_n.copy()

        # 3. Latent Heat Correction
        # Use a dummy laser object for update_fine_mesh and latent heat routines
        class DummyLaser:
            pass
        dummy_laser = DummyLaser()
        dummy_laser.x = x
        dummy_laser.y = y
        dummy_laser.r_b = r_b
        dummy_laser.laser_coef = laser_coef
        spec_hp.update_fine_mesh(SsState, dummy_laser)
        SsState.Q_latent_buffer.fill(0.0)
        kernels.compute_latent_heat_source(SsState.Q_latent_buffer, mat, dummy_laser, geom, num, SsState)
        Q_modes = kernels.project_box_to_modes(SsState.Q_latent_buffer, SsState)
        kernels.add_source_term_modes(SsState.aK, SsState.KK, Q_modes)
        kernels.add_source_term_modes(SsState.a_temp, SsState.KK, Q_modes)

        # 4. Nonlinear iteration for evaporation
        T_temp = kernels.reconstruct_surface_temperature(SsState.a_temp, SsState)
        for k in range(30):
            kernels.compute_evaporation_flux(
                T_temp, SsState.q_evap_buffer,
                mat.Pa if hasattr(mat, 'Pa') else 101325.0,
                mat.R_v if hasattr(mat, 'R_v') else 150.774,
                mat.T_boil if hasattr(mat, 'T_boil') else 3090.0,
                mat.DeltaH_LV if hasattr(mat, 'DeltaH_LV') else 7.41e6,
                mat.R_v if hasattr(mat, 'R_v') else 150.774,
                mat.T_liquidus if hasattr(mat, 'T_liquidus') else 1800.0
            )
            q_evap = SsState.q_evap_buffer
            np.subtract(q_las, q_evap, out=SsState.q_diff, casting='same_kind')
            S_target = SsState.dct_scale * spec_hp.DCT_II(SsState.q_diff)
            S_current = 0.1 * S_target + 0.9 * S_current
            np.multiply(1.0, S_current, out=SsState.B_buffer, casting='same_kind')
            kernels.update_modes_etd1(SsState.aK, SsState.KK_by_Cp, SsState.B_buffer, SsState.a_temp)
            T_old = T_temp
            T_temp = kernels.reconstruct_surface_temperature(SsState.a_temp, SsState)
            if np.max(np.abs(T_temp - T_old)) < 2e+1:
                break

        SsState.q_evap_old = q_evap.astype(np.float32, copy=True)
        P_laser = np.sum(q_las) * geom.dx * geom.dy

        # Optionally, return metrics for logging/diagnostics
        metrics = {
            'T_surface': T_temp,
            'P_laser': P_laser,
            'n_evap_iter': k+1
        }

        # Update state for next step
        SsState.a = SsState.a_temp.copy()
        return SsState, metrics
