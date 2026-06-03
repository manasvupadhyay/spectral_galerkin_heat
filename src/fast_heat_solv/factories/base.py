"""Abstract interface for Simulation Factories."""

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
from typing import Any
from fast_heat_solv.solvers.base import HeatSolver
from fast_heat_solv.io_utils.io_base import IOManager

class SimulationFactory(ABC):
    """
    Abstract Factory for creating simulation components.
    Allows switching between different implementations (CPU/GPU, Spectral/FEM)
    without changing the main workflow logic.
    """

    def __init__(self, context: Any):
        self.context = context

    @abstractmethod
    def create_heat_solver(self) -> HeatSolver:
        """Create and return a configured HeatSolver instance."""
        pass

    @abstractmethod
    def create_io_manager(self) -> IOManager:
        """Create and return a configured IOManager instance."""
        pass
