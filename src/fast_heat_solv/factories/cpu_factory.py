from fast_heat_solv.factories.base import SimulationFactory
from fast_heat_solv.solvers.base import HeatSolver
from fast_heat_solv.io_utils.io_base import IOManager

# Concrete implementations
# Note: 'file_io' folder name is used to avoid conflict with Python's built-in 'io' module
from fast_heat_solv.io_utils.spectral_fs_io import LocalFSIOManager
from fast_heat_solv.backends import NumpyBackend
from ..solvers.spectral import SpectralSolver

class CPUSimulationFactory(SimulationFactory):
    """
    Concrete Factory for CPU-based simulations.

    Creates:

    - SpectralSolver with a NumpyBackend: the numerical physics engine.
    - LocalFSIOManager: Filesystem-based output handler.
    """

    def create_heat_solver(self) -> HeatSolver:
        """
        Create the spectral solver using the CPU (NumPy/Numba) backend.
        """
        return SpectralSolver(backend=NumpyBackend())

    def create_io_manager(self) -> IOManager:
        """
        Create the local file system IO manager.
        """
        return LocalFSIOManager()
