"""
Laser path and state definitions.
"""
# Copyright 2026 Laboratoire de Mécanique des Solides (LMS),
# École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris,
# Route de Saclay, Palaiseau, 91128, France.
#
# Author: Théo Andrieux, Jules Dichamp, Manas V. Upadhyay
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


__author__ = "Théo Andrieux, Jules Dichamp, Manas V. Upadhyay"
__copyright__ = "Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris"

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Beam profiles. The absorbed surface flux ``q = I₀ · s`` splits into the total
# power (``∬ q dA = A·P`` fixes ``I₀``) and the peak-normalized shape ``s``.


def super_gaussian_area_factor(order: float) -> float:
    """Dimensionless ``f`` such that ``∬ s dA = f · r_x · r_y`` for the
    super-Gaussian ``s = exp(-2 ρⁿ)``: ``f(n) = π · 2^(1-2/n) · Γ(2/n) / n``
    (``f(2) = π/2`` Gaussian, ``f(n→∞) → π`` flat-top)."""
    n = float(order)
    return math.pi * 2.0 ** (1.0 - 2.0 / n) * math.gamma(2.0 / n) / n


def super_gaussian_flux(xp, x, y, x0, y0, r_x, r_y, order, peak_intensity):
    """Super-Gaussian flux ``I₀ · exp(-2 ρⁿ)``, ``ρ² = ((x-x0)/r_x)² + ((y-y0)/r_y)²``.

    Backend-agnostic via ``xp`` (numpy/cupy); ``order = 2`` is the Gaussian.
    """
    dx = (x[None, :] - x0) / r_x
    dy = (y[:, None] - y0) / r_y
    rho_sq = dx * dx + dy * dy
    n = float(order)
    # order 2 is the common path; skip the fractional power.
    shape = xp.exp(-2.0 * rho_sq) if n == 2.0 else xp.exp(-2.0 * rho_sq ** (n / 2.0))
    return (peak_intensity * shape).astype(x.dtype)


def _erf(xp, z):
    """Backend-agnostic error function (numpy → SciPy, cupy → cupyx.scipy)."""
    if xp.__name__ == "cupy":
        from cupyx.scipy.special import erf as _e
    else:
        from scipy.special import erf as _e
    return _e(z)


def gaussian_cell_integrated_flux(xp, x, y, x0, y0, r_x, r_y, hx, hy, peak_intensity):
    """Return the cell-mean Gaussian flux, integrated analytically over each cell.

    Each node ``(x_i, y_j)`` stores the *mean* of ``q = I₀·exp(-2 ρ²)`` over its
    cell ``[x_i ± hx/2] × [y_j ± hy/2]`` instead of the point sample. The integral
    separates into 1-D factors

        ∫_a^b exp(-2 u²/r²) du = r·√(π/8)·[erf(√2·b/r) − erf(√2·a/r)].

    Done so a beam narrower than the grid (``r ≪ h``) still deposits its full power.

    ``hx``/``hy`` are the cell sizes (``geom.d.x``/``geom.d.y``). Gaussian only;
    other-order super-Gaussians have no elementary integral.
    """
    s2 = math.sqrt(2.0)
    norm_x = r_x * math.sqrt(math.pi / 8.0)
    norm_y = r_y * math.sqrt(math.pi / 8.0)
    # 1-D cell-mean factors (per unit length): integral over the cell / cell width.
    ax = (x - x0 - hx / 2.0) * s2 / r_x
    bx = (x - x0 + hx / 2.0) * s2 / r_x
    ay = (y - y0 - hy / 2.0) * s2 / r_y
    by = (y - y0 + hy / 2.0) * s2 / r_y
    mean_x = norm_x * (_erf(xp, bx) - _erf(xp, ax)) / hx   # (NX,)
    mean_y = norm_y * (_erf(xp, by) - _erf(xp, ay)) / hy   # (NY,)
    flux = peak_intensity * (mean_y[:, None] * mean_x[None, :])  # (NY, NX)
    return flux.astype(x.dtype)


@dataclass(frozen=True)
class LaserProfile:
    """A beam profile: spatial shape and its energy-conserving normalization.

    Both the area factor ``f`` and the spatial shape derive from ``order``.
    """
    name: str
    order: float
    # When True, the Gaussian flux is integrated analytically over each cell
    # Gaussian (order 2) only.
    cell_integrated: bool = False

    @property
    def area_factor(self) -> float:
        """Dimensionless ``f`` such that ``∬ s dA = f · r_x · r_y``."""
        return super_gaussian_area_factor(self.order)

    def peak_intensity(self, power: float, absorptivity: float, ref_area: float) -> float:
        """Energy-conserving peak ``I₀ = A·P / (f · ref_area)`` (``ref_area = r_x·r_y``)."""
        return absorptivity * power / (self.area_factor * ref_area)

    def flux(self, xp, x, y, x0, y0, r_x, r_y, peak_intensity, hx=None, hy=None):
        """Spatial flux field ``I₀ · s(x, y)`` for this profile.

        With ``cell_integrated`` set (Gaussian only) and the cell sizes
        ``hx``/``hy`` supplied, returns the exact cell-averaged flux so the total
        deposited power equals ``A·P`` independent of beam position; otherwise
        point-samples the shape at the cell centres.
        """
        if self.cell_integrated:
            if self.order != 2.0:
                raise ValueError(
                    "cell_integrated flux is only defined for the Gaussian "
                    f"(order 2), got order {self.order}."
                )
            if hx is None or hy is None:
                raise ValueError("cell_integrated flux requires cell sizes hx, hy.")
            return gaussian_cell_integrated_flux(
                xp, x, y, x0, y0, r_x, r_y, hx, hy, peak_intensity
            )
        return super_gaussian_flux(
            xp, x, y, x0, y0, r_x, r_y, self.order, peak_intensity
        )


# Canonical order per named profile; ``None`` means take it from the config.
# flat-top is a high-order super-Gaussian (n≈12) rather than a discontinuous
# step, which would ring (Gibbs) in the spectral solver.
_PROFILE_ORDERS: dict[str, float | None] = {
    "gaussian": 2.0,
    "flat_top": 12.0,
    "super_gaussian": None,
}


def build_laser_profile(name: str, order: float = 2.0,
                        cell_integrated: bool = False) -> LaserProfile:
    """Resolve a profile name to a :class:`LaserProfile`. ``order`` is used only
    for ``"super_gaussian"``; raises ``ValueError`` on an unknown name.

    ``cell_integrated`` enables exact analytic cell-averaging of the Gaussian
    flux (see :func:`gaussian_cell_integrated_flux`)."""
    key = str(name).lower()
    try:
        canonical = _PROFILE_ORDERS[key]
    except KeyError:
        choices = ", ".join(sorted(_PROFILE_ORDERS))
        raise ValueError(
            f"Unknown laser profile: {name!r}. Choose one of: {choices}."
        ) from None
    resolved_order = float(order) if canonical is None else canonical
    return LaserProfile(name=key, order=resolved_order,
                        cell_integrated=bool(cell_integrated))


@dataclass
class LaserState:
    """Immutable snapshot of the laser at one time point.

    Attributes
    ----------
    x, y : float
        Laser position in metres (within the domain ``[0, L]``).
    power : float
        Laser power in watts.
    is_on : bool
        True while the laser is emitting.
    v : tuple[float, float], optional
        Velocity ``(vx, vy)`` in m/s, by default ``(0.0, 0.0)``.
    """

    x: float
    y: float
    power: float
    is_on: bool
    v: tuple[float, float] = (0.0, 0.0)

class LaserPath(ABC):
    """Abstract laser trajectory: position and power as a function of time.

    Implementations return a :class:`LaserState` per time step (consistent units:
    metres, watts, m/s). Set ``is_on=False`` (or power 0) once the laser leaves
    the domain.

    Example
    -------
    >>> class ConstantVelocityLaser(LaserPath):
    ...     def __init__(self, x0, y0, vx, power):
    ...         self.x0, self.y0, self.vx, self.power = x0, y0, vx, power
    ...     def get_state(self, time, dt):
    ...         return LaserState(self.x0 + self.vx * time, self.y0,
    ...                           self.power, is_on=True, v=(self.vx, 0.0))
    """

    @abstractmethod
    def get_state(self, time: float, dt: float) -> LaserState:
        """Return the :class:`LaserState` at *time* (called once per time step).

        Parameters
        ----------
        time : float
            Current simulation time in seconds (monotonically increasing).
        dt : float
            Current time-step length in seconds.
        """
