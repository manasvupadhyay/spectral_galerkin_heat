from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

class MicrostructureSolver(ABC):
    """
    Abstract Interface for Microstructure Evolution Solvers.
    
    This component is responsible for simulating the grain growth, phase transformation,
    or other microstructure-level phenomena during the process. It receives the
    thermal field from the HeatSolver and evolves its internal state.
    """

    @abstractmethod
    def initialize(self, output_dir: Optional[str] = None) -> Any:
        """
        Initialize the microstructure solver state.
        
        Args:
            output_dir (str, optional): Path to the output directory where artifacts (seeds.txt) should be saved.

        This may involve:
        - Loading initial grain seeds from file (Neper/MicrostructPy)
        - Generating synthetic microstructure (Voronoi)
        - Building spatial index structures (AABB Tree, Octree)
        
        Returns:
            Any: The initial state object or None.
        """
        pass

    @abstractmethod
    def update(self, t: float, dt: float, temperature_field: Any) -> Dict[str, Any]:
        """
        Evolve the microstructure state for one time step.

        Args:
            t (float): Current simulation time.
            dt (float): Time step size.
            temperature_field (Any): The current temperature field from the HeatSolver.
                                     Could be a numpy array, cupy array, or specific object.

        Returns:
            Dict[str, Any]: A dictionary of metrics or status updates (e.g., number of active grains, max growth).
        """
        pass

    @abstractmethod
    def finalize(self):
        """
        Perform any cleanup or final data persistence at the end of the simulation.
        """
        pass
