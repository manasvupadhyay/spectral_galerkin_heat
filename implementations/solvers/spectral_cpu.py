import numpy as np
import implementations.physics.spectral_cpu_kernels as kernels

class SpectralSolverCPU:
    """
    SpectralSolverCPU implements a spectral method for solving the heat equation on the CPU.
    It manages the solver state, initialization, and time-stepping logic, including laser source,
    latent heat, and evaporation effects. The solver is designed for modularity and performance.
    """
    def __init__(self, context):
        """
        Initialize the SpectralSolverCPU with the simulation context.
        Args:
            context: SimulationContext containing geometry, material, laser, and numerical parameters.
        """
        self.context = context
        self.state = None
        
    def initialize(self):
        """
        Set up the spectral solver state, allocate buffers, and set the initial condition.
        Returns:
            SpectralSolverState: The initialized solver state object.
        """
        geom = self.context.geom
        num = self.context.num
        mat = self.context.mat  

        # Initialize spectral solver state
        self.state = kernels.SpectralSolverState()
        self.state.prepare_reconstruction_basis(geom)
        self.state.prepare_K_buffers(mat, geom, num)

        # Initial condition: mean T in mode (0,0,0)
        self.state.a = np.zeros((num.nz, num.ny, num.nx), dtype=np.float32)
        # Use T0 if present, else default to 293.0
        T0 = getattr(mat, 'T0', 293.0)
        self.state.a[0,0,0] = T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz)

        return self.state

    def step(self, t, dt):
        """
        Advance the spectral solution by one time step using ETD1 and nonlinear evaporation correction.
        Handles laser source, latent heat, and evaporation effects.
        Args:
            t (float): Current simulation time.
            dt (float): Time step size.
            state (SpectralSolverState): Current solver state.
        Returns:
            tuple: (updated SpectralSolverState, metrics dict)
        """
        # Unpack all needed context attributes at the top for clarity
        context = self.context
        geom = context.geom
        num = context.num
        mat = context.mat
        laser_path = context.laser_path
        laser_params = context.laser
        SsState = self.state
        a = SsState.a

        # Fetch laser state at current time
        laser_state = laser_path.get_state(t, dt)
        x = laser_state.x
        y = laser_state.y
        power = laser_state.power
        is_on = laser_state.is_on
        r_b = laser_params.radius
        absorptivity = laser_params.absorptivity
        laser_coef = absorptivity * 2.0 * power / (np.pi * r_b ** 2) if is_on else 0.0

        # 1. Compute source terms (Laser + Evaporation)
        q_las = kernels.compute_gaussian_laser_flux(
            SsState.X, SsState.Y,
            x, y,
            r_b, laser_coef
        )
        v_x, v_y = laser_state.v if hasattr(laser_state, 'v') else (0.0, 0.0)
        q_evap = kernels.shift_flux(SsState.q_evap_old, (v_x*num.dt, v_y*num.dt), geom)
        q_dct = kernels.DCT_II(q_las - q_evap)

        # 2. Linear step (ETD1)
        np.multiply(a, SsState.K, out=SsState.aK, casting='same_kind')
        S_n = SsState.dct_scale * q_dct
        np.multiply(SsState.dct_scale, q_dct, out=SsState.B_buffer, casting='same_kind')
        kernels.update_modes_etd1(SsState.aK, SsState.KK_by_Cp, SsState.B_buffer, SsState.a_temp)
        S_current = S_n.copy()

        # 3. Latent Heat Correction
        kernels.update_fine_mesh(SsState, x, y) # can be moved easily to kernels
        SsState.Q_latent_buffer.fill(0.0)
        kernels.compute_latent_heat_source(SsState.Q_latent_buffer, mat, x, y, num, SsState)
        Q_modes = kernels.project_box_to_modes(SsState.Q_latent_buffer, SsState)
        kernels.add_source_term_modes(SsState.aK, SsState.KK, Q_modes)
        # TODO Check if next line is necessary
        kernels.add_source_term_modes(SsState.a_temp, SsState.KK, Q_modes)

        # 4. Nonlinear iteration for evaporation
        T_temp = kernels.reconstruct_surface_temperature(SsState.a_temp, SsState)
        for k in range(30):
            kernels.compute_evaporation_flux(
                T_temp, SsState.q_evap_buffer, mat.Pa, mat.R_v, mat.T_boil, 
                mat.DeltaH_LV, mat.R_v, mat.T_liquidus
            )
            q_evap = SsState.q_evap_buffer
            np.subtract(q_las, q_evap, out=SsState.q_diff, casting='same_kind')
            S_target = SsState.dct_scale * kernels.DCT_II(SsState.q_diff)
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
            'T_surface_max': np.max(T_temp),
            'P_laser': P_laser,
            'n_evap_iter': k+1
        }

        # Update state for next step
        SsState.a = SsState.a_temp.copy()
        return SsState, metrics
