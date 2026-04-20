from fast_heat_solv.factories.base import SimulationFactory
from fast_heat_solv.solvers.base import HeatSolver
from fast_heat_solv.io_utils.io_base import IOManager

from fast_heat_solv.io_utils.spectral_fs_io import LocalFSIOManager
from ..solvers.spectral_cpu_linear import SpectralSolverCPULinear

class CPULinearSimulationFactory(SimulationFactory):
    """
    Concrete Factory for CPU-based purely linear simulations.
    Creates:
      - SpectralSolverCPULinear: The linear CPU solver.
      - LocalFSIOManager: Filesystem-based output handler.
    """

    def create_heat_solver(self) -> HeatSolver:
        """
        Create the linear CPU spectral solver instance.
        """
        return SpectralSolverCPULinear()

    def create_io_manager(self) -> IOManager:
        """
        Create the local file system IO manager.
        """
        return LocalFSIOManager()
