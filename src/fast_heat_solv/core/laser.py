"""
Laser path and state definitions.
"""
# Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique
#
# Author: Théo Andrieux
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


__author__ = "Théo Andrieux"
__copyright__ = "Copyright 2026, LMS, École Polytechnique"

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LaserState:
    """
    Data class representing the state of the laser at a given time point.

    This immutable snapshot captures all laser parameters needed by the solver
    to compute heat source terms.

    Attributes
    ----------
    x : float
        X coordinate of the laser position in meters. Must be within domain bounds
        [0, Lx], where Lx is the domain size.
    y : float
        Y coordinate of the laser position in meters. Must be within domain bounds
        [0, Ly], where Ly is the domain size.
    power : float
        Laser power in watts. Typically 100–500 W for additive manufacturing.
        Negative or NaN values will cause numerical failures; ensure validation
        in your :class:`LaserPath` subclass.
    is_on : bool
        True if the laser is currently emitting, False otherwise.
        When False, power should still be specified (often 0.0).
    v : tuple[float, float], optional
        Velocity of the laser in the x and y directions in m/s, by default (0.0, 0.0).
        Used for Doppler correction and trajectory tracking. Set to (0.0, 0.0) for
        quasi-static heating.

    Examples
    --------
    Create a stationary laser at (0.001, 0.0005) with 200 W power:

    >>> state = LaserState(x=0.001, y=0.0005, power=200.0, is_on=True, v=(0.0, 0.0))
    """

    x: float
    y: float
    power: float
    is_on: bool
    v: tuple[float, float] = (0.0, 0.0)

class LaserPath(ABC):
    """
    Abstract interface for defining laser movement and power evolution.

    **Contract and Assumptions**

    - :meth:`get_state` is called once per simulation time step, at every grid point
      when computing heat sources. Avoid heavy I/O or expensive computations.
    - Returned :class:`LaserState` must have consistent units: coordinates in meters,
      power in watts, velocity in m/s.
    - If the laser travels beyond the domain, set ``is_on=False`` or return power=0.0
      to prevent numerical instability.
    - Time is monotonically increasing; you may cache trajectory data for fast lookups.

    **Examples**

    Implement a simple linear laser path at constant power:

    >>> from fast_heat_solv.core.laser import LaserPath, LaserState
    >>> class ConstantVelocityLaser(LaserPath):
    ...     def __init__(self, x0: float, y0: float, vx: float, power: float):
    ...         self.x0 = x0
    ...         self.y0 = y0
    ...         self.vx = vx
    ...         self.power = power
    ...
    ...     def get_state(self, time: float, dt: float) -> LaserState:
    ...         x = self.x0 + self.vx * time
    ...         return LaserState(x=x, y=self.y0, power=self.power, is_on=True, v=(self.vx, 0.0))

    For pulsed or modulated laser:

    >>> import math
    >>> class PulsedLaser(LaserPath):
    ...     def __init__(self, x: float, y: float, base_power: float, frequency: float = 1.0):
    ...         self.x = x
    ...         self.y = y
    ...         self.base_power = base_power
    ...         self.frequency = frequency
    ...
    ...     def get_state(self, time: float, dt: float) -> LaserState:
    ...         cycle = math.sin(2.0 * math.pi * self.frequency * time)
    ...         power = max(0.0, self.base_power * (0.5 + 0.5 * cycle))
    ...         is_on = power > 10.0
    ...         return LaserState(x=self.x, y=self.y, power=power, is_on=is_on, v=(0.0, 0.0))
    """

    @abstractmethod
    def get_state(self, time: float, dt: float) -> LaserState:
        """
        Returns laser position, power, and velocity at a given simulation time.

        **Performance Note:**
        This method may be called millions of times across all grid points and time steps.
        Cache computations where possible. Avoid I/O, external API calls, or heavy
        array allocations.

        Parameters
        ----------
        time : float
            Current simulation time in seconds. Monotonically increasing across the
            simulation.
        dt : float
            Current simulation time step length in seconds. Useful for adaptive
            sampling but not required for the state calculation.

        Returns
        -------
        LaserState
            An object describing the laser's momentary physical attributes.
            All fields must be finite (no NaN or inf).

        Raises
        ------
        ValueError
            If time < 0 or if parameters are out of physically valid ranges.
        """
        pass
