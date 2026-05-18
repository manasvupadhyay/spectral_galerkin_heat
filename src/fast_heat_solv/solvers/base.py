"""Abstract interface for Heat Equation Solvers."""

# Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique
#
# Author: Théo Andrieux
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


__author__ = "Théo Andrieux"
__copyright__ = "Copyright 2026, LMS, École Polytechnique"

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional
import numpy as np
from fast_heat_solv.core.parameters import SimulationContext

class HeatSolver(ABC):
    """
    Abstract interface for a Heat Equation Solver.

    Encapsulates the physics engine and numerical methods for solving the
    transient heat equation with phase change and evaporation.

    **Contract:**
    - Call ``initialize`` once before ``step``.
    - Call ``step`` repeatedly to advance time.
    - Call ``finalize`` to release resources (GPU memory, etc.).
    - Do not modify returned state objects; call ``step`` to evolve.
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
