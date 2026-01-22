from abc import ABC, abstractmethod
from .solver import HeatSolver
from .io import IOManager
from ..core.parameters import SimulationContext

class SimulationFactory(ABC):
    """Abstract Factory: Creates families of related objects (Solver, IO, etc.)"""

    def __init__(self, context: SimulationContext):
        self.context = context

    @abstractmethod
    def create_heat_solver(self) -> HeatSolver:
        """Create a solver instance compatible with the factory's strategy (CPU/GPU)."""
        pass

    @abstractmethod
    def create_io_manager(self) -> IOManager:
        """Create an IO manager instance."""
        pass
