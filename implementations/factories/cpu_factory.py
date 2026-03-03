from interfaces.factory import SimulationFactory
from interfaces.solver import HeatSolver
from core.io import IOManager # Import interface from core, not stdlib io

# Concrete implementations
# Note: 'file_io' folder name is used to avoid conflict with Python's built-in 'io' module
from implementations.file_io.spectral_fs_io import LocalFSIOManager
from ..solvers.spectral_cpu import SpectralSolverCPU

class CPUSimulationFactory(SimulationFactory):
    """
    Concrete Factory for CPU-based simulations.
    Creates:
      - SpectralSolverCPU: The numerical physics engine.
      - LocalFSIOManager: Filesystem-based output handler.
    """

    def create_heat_solver(self) -> HeatSolver:
        """
        Create the CPU spectral solver instance using the standard Numba/NumPy backend.
        """
        return SpectralSolverCPU(self.context)

    def create_io_manager(self) -> IOManager:
        """
        Create the local file system IO manager.
        """
        return LocalFSIOManager()
