"""
Abstract interface for Heat Equation Solvers.

Author: Théo Andrieux (@TheoADX)
Copyright: (c) 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique. All rights reserved.
"""

__author__ = "Théo Andrieux"
__copyright__ = "Copyright 2026, LMS, École Polytechnique"

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional
import numpy as np
from fast_heat_solv.core.parameters import SimulationContext

class HeatSolver(ABC):
    """
    Abstract interface for a Heat Equation Solver.
    Encapsulates the physics engine and numerical methods.

    The solver manages its own internal state. Callers interact with
    it through ``initialize``, ``step``, ``set_state`` and ``finalize``.
    """

    @abstractmethod
    def initialize(self, context: SimulationContext) -> Any:
        """
        Initialize (or re-initialize) the solver with the given context.

        Stores the context internally, allocates fields and spectral
        coefficients, and returns the initial state object.

        Parameters
        ----------
        context : SimulationContext
            Full simulation parameters (geometry, material, etc.).

        Returns
        -------
        Any
            The initial state object (implementation-dependent).
        """
        pass

    @abstractmethod
    def step(self, t: float, dt: float) -> Tuple[Any, Dict[str, float]]:
        """
        Advance the simulation by one time step `dt`.

        Parameters
        ----------
        t : float
            Current simulation time.
        dt : float
            Time step size.

        Returns
        -------
        tuple
            new_state : Any
                The evolved state object.
            metrics : dict
                Dictionary of scalar diagnostics (e.g., {'P_laser': 50.0, 'T_max': 2000.0}).
        """
        pass



    @abstractmethod
    def finalize(self) -> None:
        """
        Clean up resources (GPU memory, thread pools) if necessary.
        """
        pass
