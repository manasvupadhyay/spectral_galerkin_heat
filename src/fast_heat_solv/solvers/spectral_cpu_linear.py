"""
Spectral CPU Linear Solver Implementation.

Author: Théo Andrieux (@TheoADX)
Copyright: (c) 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique. All rights reserved.
"""

__author__ = "Théo Andrieux"
__copyright__ = "Copyright 2026, LMS, École Polytechnique"

import numpy as np
import fast_heat_solv.physics.spectral_cpu_kernels as kernels
from fast_heat_solv.core.parameters import SimulationContext
from fast_heat_solv.solvers.base import HeatSolver
from fast_heat_solv.core.laser import LaserState, LaserPath
from typing import Optional

class SpectralSolverCPULinear(HeatSolver):
    """
    SpectralSolverCPULinear implements a strictly linear spectral method 
    for solving the heat equation on the CPU.
    
    It disables latent heat and evaporation, applying only the laser source.
    Useful for testing against exact linear analytical solutions.
    """
    def __init__(self, context: Optional[SimulationContext] = None):
        """
        Initializes the SpectralSolverCPULinear.

        Optionally attach a SimulationContext at construction time.
        The context can also be provided (or overridden) later via
        ``initialize(context)``.

        Parameters
        ----------
        context : SimulationContext, optional
            A dataclass containing complete simulation parameters, by default None.
        """
        self.context: Optional[SimulationContext] = context
        self.state: Optional[kernels.SpectralSolverState] = None
        
    def initialize(self, context: Optional[SimulationContext] = None):
        """
        Set up the spectral solver state, allocate buffers, and set the initial condition.

        Parameters
        ----------
        context : SimulationContext, optional
            If provided, replaces the stored SimulationContext. By default None.

        Returns
        -------
        fast_heat_solv.physics.spectral_cpu_kernels.SpectralSolverState
            The initialized solver state object containing grid buffers and spectra.
        
        Raises
        ------
        RuntimeError
            If SimulationContext is neither provided here nor at construction.
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

        Parameters
        ----------
        t : float
            Current simulation time.
        dt : float
            Time step size.

        Returns
        -------
        tuple
            SsState : fast_heat_solv.physics.spectral_cpu_kernels.SpectralSolverState
                 The updated linear solver state.
            metrics : dict
                 Metrics from the iteration, such as max temperature and number of evaporation steps.
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

        # 2. Linear step (ETD1)
        np.multiply(SsState.a, SsState.K, out=SsState.a, casting='same_kind')

        S_n = grid.dct_scale * kernels.DCT_II(q_las)
        kernels.update_modes_etd1(SsState.a, SsState.KK, grid.Cp32_broadcast, S_n, buffers.a_temp)

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

        Parameters
        ----------
        temperature_field : np.ndarray
            A 3-D array (shape ``(N_z, N_y, N_x)``) containing temperatures.
        
        Raises
        ------
        RuntimeError
            If called before the solver is initialized.
        """
        if self.state is None or self.context is None:
            raise RuntimeError("Solver must be initialized before calling set_state().")
        geom = self.context.geom
        T = np.asarray(temperature_field, dtype=np.float32)
        scale = np.sqrt(np.float32(geom.dx * geom.dy * geom.dz))
        self.state.a = kernels.DCT_II(T) * scale

    def finalize(self) -> None:
        """
        Clean up CPU resources.
        """
        pass
