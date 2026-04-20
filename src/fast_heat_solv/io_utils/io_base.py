from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class IOManager(ABC):
    """
    Abstract interface for Input/Output management in the simulation workflow.
    Handles data persistence (saving results) and potentially retrieval (checkpointing).

    This enforces a standard contract regardless of whether we write to
    local filesystem, HDF5, cloud storage, or databases.
    """

    @abstractmethod
    def initialize(self, context: Any) -> None:
        """
        Setup the output environment based on the simulation context.
        Must determine the unique 'run_id' and create necessary directory structures.

        Args:
            context: The SimulationContext object containing configuration (context.io is now a flexible dictionary from YAML, not IOParams).
        """
        pass

    @abstractmethod
    def save_step(self, time: float, step: int, state: Any, **kwargs) -> None:
        """
        Save simulation state for the current time step.
        Implementation decides what to extract and persist from the simulation state object.

        Args:
            time: Current simulation time (s).
            step: Current time step index.
            state: The simulation state object (e.g., spectral coefficients, temperature field, etc).
            **kwargs: Additional data to save (e.g., laser_state, derived metrics).
        """
        pass

    @abstractmethod
    def get_output_path(self, filename: str, subdir: Optional[str] = None) -> str:
        """
        Helper to construct a full path for a given filename within the current run directory.

        Args:
            filename: Name of the file.
            subdir: Optional subdirectory (e.g., 'fields', 'profiles').

        Returns:
            Full absolute path str.
        """
        pass

    @abstractmethod
    def load_step(self, step: Union[int, str] = 'latest') -> Optional[Dict[str, Any]]:
        """
        Retrieve simulation state for a specific step. Used for checkpointing or post-processing.

        Args:
            step: Step index to load, or 'latest' to find the most recent.

        Returns:
            Dictionary containing loaded state (scalars, arrays, time), or None if not found.
        """
        pass

    @abstractmethod
    def process_step(self, t: float, step: int, state: Any, laser_path: Any) -> None:
        """
        Called every time step. Internally decides whether to write to disk
        based on the configured interval and output types.

        Args:
            t: Current simulation time (s).
            step: Current time step index.
            state: The simulation state object.
            laser_path: The laser path object (or None).
        """
        pass

    @abstractmethod
    def process_end(self, t: float, step: int, state: Any, laser_path: Any) -> None:
        """
        Called once at the end of the simulation to save 'at_end' outputs.

        Args:
            t: Final simulation time (s).
            step: Final time step index.
            state: The simulation state object.
            laser_path: The laser path object (or None).
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """
        Clean up resources, close open file handles, flush buffers, and write final logs.
        """
        pass
