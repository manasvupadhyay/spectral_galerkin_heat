"""
Abstract interface for Simulation Factories.

Author: Théo Andrieux (@TheoADX)
Copyright: (c) 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique. All rights reserved.
"""

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
