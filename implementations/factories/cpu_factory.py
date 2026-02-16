from interfaces.factory import SimulationFactory
from interfaces.solver import HeatSolver
from core.io import IOManager # Import interface from core, not stdlib io
from interfaces.microstructure import MicrostructureSolver

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

    def create_microstructure_solver(self) -> MicrostructureSolver:
        # For now, return a placeholder or initial implementation
        # The concrete TreeMicroSolver will be implemented in Step 3
        # For now, we can raise NotImplementedError or return a dummy
        # But to allow compilation/running, let's create a minimal mock inside 
        # implementations/solvers/microstructure.py eventually.
        # Here we import it dynamically to avoid circular imports.
        # 
        # TODO: Replace with actual TreeMicroSolver once implemented.
        from implementations.solvers.microstructure import TreeMicroSolver
        return TreeMicroSolver(self.context)