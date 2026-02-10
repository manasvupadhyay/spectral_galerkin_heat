import cupy as cp
import implementations.physics.spectral_gpu_kernels as kernels

class SpectralSolverGPU:
    """
    SpectralSolverGPU implements a spectral method for solving the heat equation on the GPU.
    It manages the solver state, initialization, and time-stepping logic, including laser source,
    latent heat, and evaporation effects. The solver is designed for modularity and performance.
    """
    def __init__(self, context):
        """
        Initialize the SpectralSolverGPU with the simulation context.
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
        self.state.a = cp.zeros((num.nz, num.ny, num.nx), dtype=cp.float32)
        # Use T0 if present, else default to 293.0
        T0 = getattr(mat, 'T0', 293.0)
        self.state.a[0,0,0] = cp.asarray(T0 * (geom.Lx * geom.Ly * geom.Lz) ** 0.5, dtype=cp.float32)

        # Ensure all state arrays are on GPU (CuPy) TO DO - Ensure in kernels directly
        for attr in ["aK", "a_temp", "K", "KK", "KK_by_Cp", "q_evap_old", "q_evap_buffer", "q_diff", "B_buffer", "Q_latent_buffer"]:
            arr = getattr(self.state, attr, None)
            if arr is not None and not isinstance(arr, cp.ndarray):
                setattr(self.state, attr, cp.asarray(arr, dtype=cp.float32))

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

        # Fetch laser state at current time
        laser_state = laser_path.get_state(t, dt)
        laser_coef = laser_params.absorptivity * 2.0 * laser_state.power / (cp.pi * laser_params.radius ** 2) 

        # 1. Compute source terms (Laser + Evaporation)
        q_las = kernels.compute_gaussian_laser_flux(
            SsState.X, SsState.Y,
            laser_state.x, laser_state.y,
            laser_params.radius, laser_coef
        )
        q_evap = kernels.shift_flux(SsState.q_evap_old, (laser_state.v[0]*num.dt, laser_state.v[1]*num.dt), geom)
        q_dct = kernels.DCT_II(q_las - q_evap)

        # 2. Linear step (ETD1)
        # In-place decay: a = a * K
        cp.multiply(SsState.a, SsState.K, out=SsState.a)
        
        S_n = SsState.dct_scale * q_dct
        cp.multiply(SsState.dct_scale, q_dct, out=SsState.B_buffer)
        
        # update_modes_etd1 now takes separate KK and Cp_broadcast instead of KK_by_Cp
        # and uses in-place decayed 'a' (passed as first arg)
        kernels.update_modes_etd1(SsState.a, SsState.KK, SsState.Cp32_broadcast, SsState.B_buffer, SsState.a_temp)
        S_current = S_n.copy()

        # 3. Latent Heat Correction (Iterative)
        kernels.update_fine_mesh(SsState, laser_state.x, laser_state.y)
        kernels.prepare_latent_history(SsState, laser_state.x, laser_state.y)
        
        # Initial Guess: Use previous Q aligned
        cp.copyto(SsState.Q_latent_buffer, SsState.Q_prev_aligned)
        
        # Allocation for new Q estimate
        Q_new = cp.empty_like(SsState.Q_latent_buffer)
        
        alpha_lat = 0.4
        for _ in range(5):
            # 1. Project current Q guess to modes
            Q_modes = kernels.project_box_to_modes(SsState.Q_latent_buffer, SsState)
            
            # 2. Construct temporary state `a_temp`
            # Start with (Decayed 'a' + Surface Sources)
            kernels.update_modes_etd1(SsState.a, SsState.KK, SsState.Cp32_broadcast, SsState.B_buffer, SsState.a_temp)
            # Add Volumetric Source (current Latent Heat guess)
            kernels.add_source_term_modes(SsState.a_temp, SsState.KK, Q_modes)
            
            # 3. Reconstruct Temperature T(Q)
            T_box, _ = kernels.reconstruct_temperature_box(SsState.a_temp, SsState)
            
            # 4. Compute new Q(T)
            kernels.compute_latent_source_only(Q_new, T_box, SsState, mat, num.dt)
            
            # 5. Relaxation: Q_buffer = alpha * Q_new + (1 - alpha) * Q_buffer
            cp.multiply(SsState.Q_latent_buffer, (1.0 - alpha_lat), out=SsState.Q_latent_buffer)
            cp.add(SsState.Q_latent_buffer, alpha_lat * Q_new, out=SsState.Q_latent_buffer)

        # Finalize: Add converged latent heat source to base state `a` so Evaporation loop sees it
        Q_modes_final = kernels.project_box_to_modes(SsState.Q_latent_buffer, SsState)
        kernels.add_source_term_modes(SsState.a, SsState.KK, Q_modes_final)
        
        # Commit history
        kernels.update_latent_history(SsState, T_box, SsState.Q_latent_buffer, laser_state.x, laser_state.y)
        
        # 4. Nonlinear iteration for evaporation
        T_temp = kernels.reconstruct_surface_temperature(SsState.a_temp, SsState)
        for k in range(30):
            T_old = T_temp
            # Compute evaporation flux based on current surface temperature guess
            kernels.compute_evaporation_flux(T_temp, SsState.q_evap_buffer, mat.Pa, mat.R_v, mat.T_boil, 
                mat.DeltaH_LV, mat.R_v, mat.T_liquidus)
            cp.subtract(q_las,SsState.q_evap_buffer, out=SsState.q_diff, casting='same_kind')
            S_target = SsState.dct_scale * kernels.DCT_II(SsState.q_diff)
            S_current = 0.1 * S_target + 0.9 * S_current
            cp.multiply(1.0, S_current, out=SsState.B_buffer, casting='same_kind')

            kernels.update_modes_etd1(SsState.a, SsState.KK, SsState.Cp32_broadcast, SsState.B_buffer, SsState.a_temp)
            T_temp = kernels.reconstruct_surface_temperature(SsState.a_temp, SsState)
            if cp.max(cp.abs(T_temp - T_old)) < 2e+1:
                break

        SsState.q_evap_old[:] = SsState.q_evap_buffer
        P_laser = cp.sum(q_las) * geom.dx * geom.dy

        # Optionally, return metrics for logging/diagnostics
        metrics = {
            'T_surface_max': cp.max(T_temp),
            'P_laser': P_laser,
            'n_evap_iter': k+1
        }

        # Update state for next step
        SsState.a = SsState.a_temp.copy()
        return SsState, metrics
