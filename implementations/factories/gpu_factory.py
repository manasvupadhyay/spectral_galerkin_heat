from interfaces.factory import SimulationFactory
from interfaces.solver import HeatSolver
from core.io import IOManager # Import interface from core, not stdlib io
from interfaces.microstructure import MicrostructureSolver

# Concrete implementations
# Note: 'file_io' folder name is used to avoid conflict with Python's built-in 'io' module
from implementations.file_io.spectral_fs_io import LocalFSIOManager
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
        return SpectralSolverGPU(self.context)

    def create_io_manager(self) -> IOManager:
        """
        Create the local file system IO manager.
        """
        return LocalFSIOManager()

    def create_microstructure_solver(self) -> MicrostructureSolver:
        """
        Currently, Microstructure Simulation is CPU-based.
        We return the CPU implementation even if the Heat Solver is GPU.
        The implementation handles data transfer (GPU->CPU) internally if needed.
        """
        from implementations.solvers.microstructure import TreeMicroSolver
        return TreeMicroSolver(self.context)