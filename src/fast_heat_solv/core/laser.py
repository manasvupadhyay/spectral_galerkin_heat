from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LaserState:
    x: float
    y: float
    power: float
    is_on: bool
    v: tuple = (0.0, 0.0)  # (vx, vy)

class LaserPath(ABC):
    """Abstract interface for defining laser movement and power evolution."""

    @abstractmethod
    def get_state(self, time: float, dt: float) -> LaserState:
        """Returns laser position, power, and velocity at a given simulation time."""
        pass
