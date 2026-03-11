import numpy as np
import cupy as cp
import implementations.physics.spectral_gpu_kernels as kernels
from core.parameters import SimulationContext
from interfaces.solver import HeatSolver
from interfaces.laser import LaserState, LaserPath
from typing import Optional, Any, Tuple, Dict

class SpectralSolverGPU(HeatSolver):
    """
    SpectralSolverGPU implements a spectral method for solving the heat equation on the GPU.
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

    def initialize(self, context: SimulationContext) -> Any:
        """
        Set up the spectral solver state, allocate buffers, and set the initial condition.

        Args:
            context: If provided, replaces the stored SimulationContext.

        Returns:
            SpectralSolverState: The initialized solver state object.
        """
        self.context = context

        geom = self.context.geom
        num = self.context.num
        mat = self.context.mat

        # Initialize spectral solver state
        self.state = kernels.SpectralSolverState(mat, geom, num)

        # Initial condition: mean T in mode (0,0,0)
        self.state.a = cp.zeros((num.nz, num.ny, num.nx), dtype=cp.float32)
        # Use T0 if present, else default to 293.0
        T0 = getattr(mat, 'T0', 293.0)
        self.state.a[0, 0, 0] = cp.float32(T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz))

        return self.state

    def step(self, t: float, dt: float) -> Tuple[Any, Dict[str, float]]:
        """
        Advance the spectral solution by one time step using ETD1 and nonlinear evaporation correction.
        Handles laser source, latent heat, and evaporation effects.
        Args:
            t (float): Current simulation time.
            dt (float): Time step size.
        Returns:
            tuple: (updated SpectralSolverState, metrics dict)
        """
        context = self.context
        geom = context.geom
        num = context.num
        mat = context.mat
        laser_path: LaserPath = context.laser_path
        laser_params = context.laser
        SsState = self.state
        grid = SsState.grid
        buffers = SsState.buffers

        # Fetch laser state at current time
        laser_state: LaserState = laser_path.get_state(t, dt)
        power = laser_state.power
        is_on = laser_state.is_on
        v_x, v_y = laser_state.v
        absorptivity = laser_params.absorptivity
        laser_coef = absorptivity * 2.0 * power / (np.pi * laser_params.radius ** 2) if is_on else 0.0

        # 1. Compute source terms (Laser + Evaporation)
        q_las = kernels.compute_gaussian_laser_flux(
            grid.x, grid.y,
            laser_state.x, laser_state.y,
            laser_params.radius, laser_coef
        )
        q_evap = kernels.shift_flux(buffers.q_evap_old, (v_x * num.dt, v_y * num.dt), geom)
        q_dct = kernels.DCT_II(q_las - q_evap)

        # 2. Linear step (ETD1)
        # In-place decay: a = a * K
        cp.multiply(SsState.a, SsState.K, out=SsState.a)

        S_n = grid.dct_scale * q_dct
        cp.multiply(grid.dct_scale, q_dct, out=buffers.B_buffer)

        # First guess for a_temp (base state without latent heat)
        kernels.update_modes_etd1(SsState.a, SsState.KK, grid.Cp32_broadcast, buffers.B_buffer, buffers.a_temp)
        S_current = S_n.copy()

        # 3. Latent Heat Correction
        if SsState.fine_mesh:
            SsState.fine_mesh.update(laser_state)

            buffers.Q_latent_buffer.fill(0.0)
            # Compute latent heat source on fine mesh from current a_temp estimate
            kernels.compute_latent_heat_source(buffers.Q_latent_buffer, mat, laser_state, num, SsState)
            # Add latent heat source to decayed a (before surface sources)
            # a_temp will be recomputed in the evaporation loop from the updated a
            kernels.add_source_term_modes(SsState.a, SsState.KK, kernels.project_box_to_modes(buffers.Q_latent_buffer, SsState))

        # 4. Nonlinear iteration for evaporation
        T_temp = kernels.reconstruct_surface_temperature(buffers.a_temp, SsState)
        for k in range(30):
            T_old = T_temp
            # Compute evaporation flux based on current surface temperature guess
            kernels.compute_evaporation_flux(T_temp, buffers.q_evap_buffer, mat.Pa, mat.T_boil,
                mat.DeltaH_LV, mat.R_v, mat.T_liquidus)
            cp.subtract(q_las, buffers.q_evap_buffer, out=buffers.q_diff, casting='same_kind')
            S_target = grid.dct_scale * kernels.DCT_II(buffers.q_diff)
            S_current = 0.1 * S_target + 0.9 * S_current
            cp.multiply(1.0, S_current, out=buffers.B_buffer, casting='same_kind')

            kernels.update_modes_etd1(SsState.a, SsState.KK, grid.Cp32_broadcast, buffers.B_buffer, buffers.a_temp)
            T_temp = kernels.reconstruct_surface_temperature(buffers.a_temp, SsState)
            if cp.max(cp.abs(T_temp - T_old)) < 2e+1:
                break

        # Avoid reallocating when possible — copy into preallocated buffer
        buffers.q_evap_old[:] = buffers.q_evap_buffer
        P_laser = cp.sum(q_las) * geom.dx * geom.dy

        # 5. Bottom convective heat loss (one-shot, no iteration needed)
        h_conv = getattr(mat, 'h_conv', 0.0)
        if h_conv > 0:
            T_bottom = kernels.reconstruct_bottom_temperature(buffers.a_temp, SsState)
            T0 = cp.float32(getattr(mat, 'T0', 293.0))
            # Negative sign: convection removes heat from z=0 boundary
            q_conv = cp.float32(-h_conv) * (T_bottom - T0)
            q_conv_dct = kernels.DCT_II(q_conv)
            B_bottom = grid.dct_scale * q_conv_dct
            kernels.add_bottom_surface_source(
                buffers.a_temp, SsState.KK,
                grid.Cp32_broadcast_bottom, B_bottom
            )

        # Optionally, return metrics for logging/diagnostics
        metrics = {
            'T_surface_max': cp.max(T_temp),
            'P_laser': P_laser,
            'n_evap_iter': k + 1
        }

        # Update state for next step
        SsState.a = buffers.a_temp.copy()
        return SsState, metrics

    def set_state(self, temperature_field: np.ndarray) -> None:
        """
        Overwrite the internal spectral coefficients from a spatial temperature field.

        Converts the cell-centred temperature array into spectral (DCT) modes
        so that the solver can continue stepping from the injected state.

        Args:
            temperature_field: 3-D array of shape ``(nz, ny, nx)``.
                Accepts NumPy or CuPy arrays.
        """
        if self.state is None or self.context is None:
            raise RuntimeError("Solver must be initialized before calling set_state().")
        geom = self.context.geom
        T = cp.asarray(temperature_field, dtype=cp.float32)
        scale = cp.float32(np.sqrt(np.float64(geom.dx * geom.dy * geom.dz)))
        self.state.a = kernels.DCT_II(T) * scale

    def finalize(self) -> None:
        """
        Clean up GPU resources if necessary.
        """
        pass
