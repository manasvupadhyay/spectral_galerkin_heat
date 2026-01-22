from abc import ABC, abstractmethod
import numpy as np
from ..core.parameters import SimulationContext

class IOManager(ABC):
    """Abstract Product: Interface for Input/Output operations."""

    @abstractmethod
    def initialize(self, context: SimulationContext):
        """Setup output files, directories, headers."""
        pass

    @abstractmethod
    def save_step(self, time: float, step_index: int, temperature_field: np.ndarray):
        """Save data for the current time step."""
        pass

    @abstractmethod
    def finalize(self):
        """Close files and clean up resources."""
        pass
