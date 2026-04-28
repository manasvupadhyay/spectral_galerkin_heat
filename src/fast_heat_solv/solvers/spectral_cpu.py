import numpy as np
import fast_heat_solv.physics.spectral_cpu_kernels as kernels
from fast_heat_solv.core.parameters import SimulationContext
from fast_heat_solv.solvers.base import HeatSolver
from fast_heat_solv.core.laser import LaserState, LaserPath
from typing import Optional

class SpectralSolverCPU(HeatSolver):
    """
    SpectralSolverCPU implements a spectral method for solving the heat equation on the CPU.
    It manages the solver state, initialization, and time-stepping logic, including laser source,
    latent heat, and evaporation effects. The solver is designed for modularity and performance.
    """
    def __init__(self, context: Optional[SimulationContext] = None):
        """
        Optionally attach a SimulationContext at construction time.
        The context can also be provided (or overridden) later via
        ``initialize(context)``.
        """
        self.context: Optional[SimulationContext] = context
        self.state: Optional[kernels.SpectralSolverState] = None
        self.mixing_omega: np.float32 = np.float32(0.1)
        self.convergence_tol: np.float32 = np.float32(1e-4)
        self.max_picard_iter: int = 30
        self.track_picard_history: bool = False
        self.picard_history = []

    def initialize(self, context: Optional[SimulationContext] = None):
        """
        Set up the spectral solver state, allocate buffers, and set the initial condition.

        Args:
            context: If provided, replaces the stored SimulationContext.

        Returns:
            SpectralSolverState: The initialized solver state object.
        """
        if context is not None:
            self.context = context
        if self.context is None:
            raise RuntimeError("SimulationContext must be provided either at construction or in initialize().")

        geom = self.context.geom
        num = self.context.num
        mat = self.context.mat  

        # Initialize spectral solver state
        self.state = kernels.SpectralSolverState(mat, geom, num)

        # Initial condition: mean T in mode (0,0,0)
        self.state.a = np.zeros((num.nz, num.ny, num.nx), dtype=np.float32)
        # Use T0 if present, else default to 293.0
        T0 = getattr(mat, 'T0', 293.0)
        self.state.a[0,0,0] = T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz)

        return self.state

    def step(self, t, dt):
        """
        Advance the spectral solution by one time step using ETD1 with a
        fixed-point iteration for all nonlinear forcing terms.

        Args:
            t (float): Current simulation time.
            dt (float): Time step size.
        Returns:
            tuple: (updated SpectralSolverState, metrics dict)
        """
        # --- Unpack context ---
        context = self.context
        geom = context.geom
        num = context.num
        mat = context.mat
        laser_path: LaserPath = context.laser_path
        laser_params = context.laser
        SsState = self.state
        grid = SsState.grid
        buffers = SsState.buffers
        fm = SsState.fine_mesh

        # --- Laser state ---
        laser_state: LaserState = laser_path.get_state(t, dt)
        power = laser_state.power
        is_on = laser_state.is_on
        v_x, v_y = laser_state.v
        absorptivity = laser_params.absorptivity
        laser_coef = (absorptivity * 2.0 * power
                      / (np.pi * laser_params.radius ** 2)) if is_on else 0.0

        # ================================================================
        # 1. Compute laser flux (constant – does not depend on T)
        # ================================================================
        q_las = kernels.compute_gaussian_laser_flux(
            grid.x, grid.y, laser_state.x, laser_state.y,
            laser_params.radius, laser_coef
        )
        P_laser = np.sum(q_las) * geom.dx * geom.dy

        # ================================================================
        # 2. Apply exponential propagator:  θ̃ = E · θ   (Algo line 7)
        # ================================================================
        np.multiply(SsState.a, SsState.K, out=SsState.a, casting='same_kind')

        # ================================================================
        # 3. Prepare latent-heat history (once per step)
        # ================================================================
        if fm:
            fm.update(laser_state)
            buffers.a_temp[:] = SsState.a
            kernels.initialize_latent_heat_if_needed(SsState)
            kernels.shift_latent_heat_history(fm, laser_state, num)

        if self.track_picard_history:
            self.picard_history = []

        # ================================================================
        # 4. Initial guesses for forcing components
        # ================================================================
        # Top surface: warm-start with shifted previous evaporation
        q_evap_shifted = kernels.shift_flux(
            buffers.q_evap_old, (v_x * dt, v_y * dt), geom)
        S_top = grid.dct_scale * kernels.DCT_II(q_las - q_evap_shifted)

        # Latent heat: warm-start with shifted Q from previous step
        Q_latent = None
        if fm:
            Q_latent = np.zeros_like(buffers.Q_latent_buffer)
            if fm.Q_prev is not None:
                Q_latent[:] = fm.Q_prev

        # Bottom convection
        h_conv = getattr(mat, 'h_conv', 0.0)
        T0 = np.float32(getattr(mat, 'T0', 293.0))
        S_bot = np.zeros_like(buffers.q_diff) if h_conv > 0 else None

        # Build initial a_temp = θ̃ + Q_mnp · F  with all forcing guesses
        kernels.update_modes_etd1(
            SsState.a, SsState.KK, grid.Cp32_broadcast,
            S_top, buffers.a_temp)
        if fm and fm.T_prev is not None and Q_latent is not None:
            kernels.add_source_term_modes(
                buffers.a_temp, SsState.KK,
                kernels.project_box_to_modes(Q_latent, SsState))
        if h_conv > 0:
            T_bottom = kernels.reconstruct_bottom_temperature(
                buffers.a_temp, SsState)
            q_conv = np.float32(-h_conv) * (T_bottom - T0)
            S_bot[:] = grid.dct_scale * kernels.DCT_II(q_conv)
            kernels.add_bottom_surface_source(
                buffers.a_temp, SsState.KK,
                grid.Cp32_broadcast_bottom, S_bot)

        # ================================================================
        # Pre-allocate arrays to avoid per-iteration allocations
        # ================================================================
        a_old = np.empty_like(buffers.a_temp)
        a_raw = np.empty_like(buffers.a_temp)
        residual_curr = np.empty_like(buffers.a_temp)
        n_elements = np.float32(buffers.a_temp.size)

        # ================================================================
        # Hoist linear/constant forcing terms
        # ================================================================
        S_las = grid.dct_scale * kernels.DCT_II(q_las)

        S_bot_raw = None
        if h_conv > 0:
            T_bottom = kernels.reconstruct_bottom_temperature(buffers.a_temp, SsState)
            q_conv = np.float32(-h_conv) * (T_bottom - T0)
            S_bot_raw = grid.dct_scale * kernels.DCT_II(q_conv)

        # ================================================================
        # 5. Fixed-point iteration
        # ================================================================
        for iter_k in range(self.max_picard_iter):
            np.copyto(a_old, buffers.a_temp)

            T_surface = kernels.reconstruct_surface_temperature(a_old, SsState)

            kernels.compute_evaporation_flux(
                T_surface, buffers.q_evap_buffer,
                mat.Pa, mat.R_v, mat.T_boil,
                mat.DeltaH_LV, mat.R_v, mat.T_liquidus)
            S_evap = grid.dct_scale * kernels.DCT_II(buffers.q_evap_buffer)
            S_top_raw = S_las - S_evap

            Q_latent_raw = None
            if fm and fm.T_prev is not None:
                np.copyto(buffers.a_temp, a_old)
                buffers.Q_latent_buffer.fill(0.0)
                kernels.compute_latent_heat_source(
                    buffers.Q_latent_buffer, mat, num, SsState)
                Q_latent_raw = buffers.Q_latent_buffer

            kernels.update_modes_etd1(
                SsState.a, SsState.KK, grid.Cp32_broadcast,
                S_top_raw, buffers.a_temp)
            if Q_latent_raw is not None:
                kernels.add_source_term_modes(
                    buffers.a_temp, SsState.KK,
                    kernels.project_box_to_modes(Q_latent_raw, SsState))
            if S_bot_raw is not None:
                kernels.add_bottom_surface_source(
                    buffers.a_temp, SsState.KK,
                    grid.Cp32_broadcast_bottom, S_bot_raw)

            np.copyto(a_raw, buffers.a_temp)
            np.subtract(a_raw, a_old, out=residual_curr)

            omega = np.float32(self.mixing_omega)
            # if iter_k == 0:
            #     np.copyto(buffers.a_temp, a_raw)
            # else:
            # np.multiply(a_raw, omega, out=buffers.a_temp)
            # buffers.a_temp += (np.float32(1.0) - omega) * a_old

            np.multiply(a_raw, omega, out=buffers.a_temp)
            buffers.a_temp += (np.float32(1.0) - omega) * a_old

            rms_diff = np.sqrt(np.vdot(residual_curr, residual_curr) / n_elements)
            rms_old = np.sqrt(np.vdot(a_old, a_old) / n_elements)
            true_rel_err = rms_diff / max(rms_old, np.float32(1e-9))

            if self.track_picard_history:
                rho_k = None
                if iter_k > 0 and self.picard_history:
                    prev_rms = self.picard_history[-1]["rms_diff"]
                    if prev_rms > 0:
                        rho_k = float(rms_diff / prev_rms)
                self.picard_history.append({
                    "iter": iter_k,
                    "rms_diff": float(rms_diff),
                    "rho": rho_k,
                    "true_rel_err": float(true_rel_err),
                })

            if true_rel_err < self.convergence_tol:
                break

        # Restore missing variables needed below
        T_temp = kernels.reconstruct_surface_temperature(buffers.a_temp, SsState)
        Q_latent = Q_latent_raw

        # ================================================================
        # 6. Commit converged state
        # ================================================================
        buffers.q_evap_old[:] = buffers.q_evap_buffer
        SsState.a = buffers.a_temp.copy()

        # ================================================================
        # 7. Update latent-heat history with converged temperature
        # ================================================================
        if fm:
            kernels.update_latent_heat_history(SsState)
            if fm.Q_prev is None:
                fm.Q_prev = np.zeros_like(buffers.Q_latent_buffer)
            if Q_latent is not None:
                fm.Q_prev[:] = Q_latent[:]

        metrics = {
            'T_surface_max': np.max(T_temp),
            'P_laser': P_laser,
            'n_evap_iter': iter_k + 1
        }
        return SsState, metrics

    def set_state(self, temperature_field: np.ndarray) -> None:
        """
        Overwrite the internal spectral coefficients from a spatial temperature field.

        Converts the cell-centred temperature array into spectral (DCT) modes
        so that the solver can continue stepping from the injected state.

        Args:
            temperature_field: 3-D NumPy array of shape ``(nz, ny, nx)``.
        """
        if self.state is None or self.context is None:
            raise RuntimeError("Solver must be initialized before calling set_state().")
        geom = self.context.geom
        T = np.asarray(temperature_field, dtype=np.float32)
        # Forward DCT-II (ortho) converts spatial field to ortho-normalised coefficients.
        # The solver's internal modes use a scaling of sqrt(dx*dy*dz) relative to the
        # standard ortho DCT coefficients.
        scale = np.sqrt(np.float32(geom.dx * geom.dy * geom.dz))
        self.state.a = kernels.DCT_II(T) * scale

    def finalize(self) -> None:
        """
        Clean up resources (CPU memory, thread pools) if necessary.
        """
        pass
