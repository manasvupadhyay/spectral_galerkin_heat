import numpy as np
import implementations.physics.spectral_cpu_kernels as kernels
from core.parameters import SimulationContext
from interfaces.solver import HeatSolver
from interfaces.laser import LaserState, LaserPath
from typing import Optional

class SpectralSolverCPULinear(HeatSolver):
    """
    SpectralSolverCPULinear implements a strictly linear spectral method 
    for solving the heat equation on the CPU. It disables latent heat 
    and evaporation, applying only the laser source.
    Useful for testing against exact linear analytical solutions.
    """
    def __init__(self, context: Optional[SimulationContext] = None):
        """
        Optionally attach a SimulationContext at construction time.
        The context can also be provided (or overridden) later via
        ``initialize(context)``.
        """
        self.context: Optional[SimulationContext] = context
        self.state: Optional[kernels.SpectralSolverState] = None
        
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
        Advance the spectral solution by one time step linearly.
        Only the laser source is applied. No latent heat or evaporation.
        
        Args:
            t (float): Current simulation time.
            dt (float): Time step size.
            
        Returns:
            tuple: (updated SpectralSolverState, metrics dict)
        """
        # Unpack context attributes
        context = self.context
        geom = context.geom
        mat = context.mat
        laser_params = context.laser
        laser_path: LaserPath = context.laser_path
        
        SsState = self.state
        grid = SsState.grid
        buffers = SsState.buffers

        # Fetch laser state at current time
        laser_state: LaserState = laser_path.get_state(t, dt)
        power = laser_state.power
        is_on = laser_state.is_on
        absorptivity = laser_params.absorptivity
        laser_coef = absorptivity * 2.0 * power / (np.pi * laser_params.radius ** 2) if is_on else 0.0

        # 1. Compute linear source term (Laser only)
        q_las = kernels.compute_gaussian_laser_flux(
            grid.x, grid.y, laser_state.x, laser_state.y,
            laser_params.radius, laser_coef
        )
        
        q_dct = kernels.DCT_II(q_las)

        # 2. Linear step (ETD1)
        # Decay term: a * exp(-K*dt) -> a
        np.multiply(SsState.a, SsState.K, out=SsState.a, casting='same_kind')
        
        # Evaluate source term in spectral space: S_n = C * q_dct
        S_n = grid.dct_scale * q_dct
        np.multiply(1.0, S_n, out=buffers.B_buffer, casting='same_kind')
        
        # First guess for a_temp: a + S_n * (1 - exp(-K*dt)) / K -> a_temp
        kernels.update_modes_etd1(SsState.a, SsState.KK, grid.Cp32_broadcast, buffers.B_buffer, buffers.a_temp)

        # Reconstruct surface temperature just for metrics
        T_temp = kernels.reconstruct_surface_temperature(buffers.a_temp, SsState)
        
        P_laser = np.sum(q_las) * geom.dx * geom.dy

        # Metrics for logging/diagnostics
        metrics = {
            'T_surface_max': np.max(T_temp),
            'P_laser': P_laser,
            'n_evap_iter': 0
        }

        # Update state for next step
        SsState.a = buffers.a_temp.copy()
        return SsState, metrics

    def set_state(self, temperature_field: np.ndarray) -> None:
        """
        Overwrite the internal spectral coefficients from a spatial temperature field.
        """
        if self.state is None or self.context is None:
            raise RuntimeError("Solver must be initialized before calling set_state().")
        geom = self.context.geom
        T = np.asarray(temperature_field, dtype=np.float32)
        scale = np.sqrt(np.float32(geom.dx * geom.dy * geom.dz))
        self.state.a = kernels.DCT_II(T) * scale

    def finalize(self) -> None:
        """
        Clean up resources.
        """
        pass
