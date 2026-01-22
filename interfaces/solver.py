from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class HeatSolver(ABC):
    """Abstract Product: Interface for the heat equation solver."""

    @abstractmethod
    def initialize(self):
        """Perform any pre-computation or resource allocation."""
        pass

    @abstractmethod
    def solve_step(self, time: float, dt: float, laser_state: any) -> np.ndarray:
        """
        Advance the simulation by one time step.
        Returns the current temperature field (or a reference to it).
        """
        pass
    
    @abstractmethod
    def get_temperature_field(self) -> np.ndarray:
        """Return the current temperature field."""
        pass
