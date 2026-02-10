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

        # 3. Combined Nonlinear Iteration (Latent Heat + Evaporation)
        kernels.update_fine_mesh(SsState, laser_state.x, laser_state.y)        
        # Save the decayed state (independent of source terms
        
        # Initial guess for T (using current a_temp from linear step)
        # T_temp used for convergence check of Surface T
        T_temp = kernels.reconstruct_surface_temperature(SsState.a_temp, SsState)
        
        for k in range(30):
            T_old = T_temp
            
            # A. Compute Latent Heat Source (based on current a_temp from previous iter)
            # update_history=False: We only commit the history state after convergence
            SsState.Q_latent_buffer.fill(0.0)
            kernels.compute_latent_heat_source(
                SsState.Q_latent_buffer, mat, laser_state.x, laser_state.y, 
                num, SsState, update_history=False
            )
            
            # B. Compute Evaporation Source (based on current Surface T)
            kernels.compute_evaporation_flux(T_temp, SsState.q_evap_buffer, mat.Pa, mat.R_v, mat.T_boil, 
                mat.DeltaH_LV, mat.R_v, mat.T_liquidus)
            
            # Update spectral boundary source (Laser - Evap)
            cp.subtract(q_las, SsState.q_evap_buffer, out=SsState.q_diff, casting='same_kind')
            S_target = SsState.dct_scale * kernels.DCT_II(SsState.q_diff)
            # Relax the Source term update
            S_current = 0.1 * S_target + 0.9 * S_current
            cp.multiply(1.0, S_current, out=SsState.B_buffer, casting='same_kind')

            # C. Construct New Temperature State (a_temp)
            # Add Volumetric Source (Latent Heat) to base state
            # a_decayed -> a_decayed + KK * Q_latent
            kernels.add_source_term_modes(SsState.a, SsState.KK, kernels.project_box_to_modes(SsState.Q_latent_buffer, SsState))
            
            # Add Boundary Source (Evaporation/Laser) to finish ETD1
            # (a_decayed + KK*Q) -> (a_decayed + KK*Q) + KK*Cp*B_buffer -> a_temp
            kernels.update_modes_etd1(SsState.a, SsState.KK, SsState.Cp32_broadcast, SsState.B_buffer, SsState.a_temp)

            # D. Check Convergence
            T_temp = kernels.reconstruct_surface_temperature(SsState.a_temp, SsState)
            
            # Require at least 2 iterations to stabilize LH + Evap coupling
            if k > 2 and cp.max(cp.abs(T_temp - T_old)) < 20:
                break
        
        # 4. Finalize Latent Heat History
        # We must run this once more with update_history=True to save the converged state for the next step
        # Ideally, we call it with the EXACT same T field as the last iteration
        kernels.compute_latent_heat_source(
            SsState.Q_latent_buffer, mat, laser_state.x, laser_state.y, 
            num, SsState, update_history=True
        )

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
