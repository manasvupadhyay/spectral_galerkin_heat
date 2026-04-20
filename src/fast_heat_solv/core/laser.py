from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LaserState:
    """
    Data class representing the state of the laser at a given time point.

    Attributes
    ----------
    x : float
        X coordinate of the laser position in meters.
    y : float
        Y coordinate of the laser position in meters.
    power : float
        Laser power in watts.
    is_on : bool
        True if the laser is currently emitting, False otherwise.
    v : tuple of float, optional
        Velocity of the laser in the x and y directions in m/s, by default (0.0, 0.0).
    """

    x: float
    y: float
    power: float
    is_on: bool
    v: tuple = (0.0, 0.0)  # (vx, vy)

class LaserPath(ABC):
    """
    Abstract interface for defining laser movement and power evolution.
    
    This class specifies how a simulation can query the laser's physical 
    status at an arbitrary point in time.
    """

    @abstractmethod
    def get_state(self, time: float, dt: float) -> LaserState:
        """
        Returns laser position, power, and velocity at a given simulation time.

        Parameters
        ----------
        time : float
            Current simulation time in seconds.
        dt : float
            Current simulation time step length in seconds.

        Returns
        -------
        LaserState
            An object describing the laser's momentary physical attributes.
        """
        pass
