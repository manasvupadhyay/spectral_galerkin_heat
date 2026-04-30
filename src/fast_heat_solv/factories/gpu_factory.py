from fast_heat_solv.factories.base import SimulationFactory
from fast_heat_solv.solvers.base import HeatSolver
from fast_heat_solv.io_utils.io_base import IOManager

# Concrete implementations
# Note: 'file_io' folder name is used to avoid conflict with Python's built-in 'io' module
from fast_heat_solv.io_utils.spectral_fs_io import LocalFSIOManager
from ..solvers.spectral_gpu import SpectralSolverGPU

class GPUSimulationFactory(SimulationFactory):
    """
    Concrete Factory for GPU-based simulations.
    
    Creates:
    
      - SpectralSolverGPU: The numerical physics engine.
      - LocalFSIOManager: Filesystem-based output handler.
    """

    def create_heat_solver(self) -> HeatSolver:
        """
        Create the GPU spectral solver instance using the standard Numba/NumPy backend.
        """
        return SpectralSolverGPU()

    def create_io_manager(self) -> IOManager:
        """
        Create the local file system IO manager.
        """
        return LocalFSIOManager()
