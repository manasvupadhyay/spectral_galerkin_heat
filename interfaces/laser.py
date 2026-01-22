from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LaserState:
    x: float
    y: float
    power: float
    is_on: bool

class LaserPath(ABC):
    """Abstract interface for defining laser movement and power evolution."""
    
    @abstractmethod
    def get_state(self, time: float) -> LaserState:
        """Returns laser position and power at a given simulation time."""
        pass
