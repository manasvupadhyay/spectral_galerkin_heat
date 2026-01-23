from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional

class HeatSolver(ABC):
    """
    Abstract interface for a Heat Equation Solver.
    Encapsulates the physics engine and numerical methods.
    """

    @abstractmethod
    def initialize(self) -> Any:
        """
        Initialize the solver state (fields, spectral coefficients, etc.).

        Returns:
            The initial state object (implementation dependent, usually an array).
        """
        pass

    @abstractmethod
    def step(self, t: float, dt: float, state: Any) -> Tuple[Any, Dict[str, float]]:
        """
        Advance the simulation by one time step dt.

        Args:
            t: Current time.
            dt: Time step size.
            state: The current state of the system (from previous step).

        Returns:
            Tuple containing:
            - new_state: The evolved state.
            - metrics: Dictionary of scalar diagnostics (e.g., {'P_laser': 50.0, 'T_max': 2000.0}).
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """
        Clean up resources (GPU memory, thread pools) if necessary.
        """
        pass
