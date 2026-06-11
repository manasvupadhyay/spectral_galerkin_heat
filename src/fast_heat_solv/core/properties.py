"""Temperature-dependent material properties (polynomial branches + f_l blend).

Each property ``k(T)``, ``rho(T)``, ``c(T)`` is described by a solid-branch and a
liquid-branch polynomial (ascending powers of ``T``), blended pointwise by the
liquid fraction ``f_l(T)`` between the solidus and liquidus temperatures::

    p(T) = (1 - f_l) * p_solid(T) + f_l * p_liquid(T)

This matches the Chadwick 316L form used by the finite-element reference solver.
The blended property is non-polynomial with slope kinks at ``T_S``/``T_L``; this
never enters the spectral scheme, because the weak-form property correction
samples ``k`` and ``a`` but never differentiates them in ``T``.

A plain scalar is also accepted (both branches collapse to the constant), so
existing constant-property configs keep working unchanged.
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

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

__all__ = [
    "liquid_fraction",
    "TempProperty",
    "MaterialModel",
]


def _xp(T):
    """Return the array module backing *T* (CuPy if it is a CuPy array, else NumPy).

    Lets the evaluator run unchanged on host (parse-time scalars / NumPy fields)
    and device (CuPy fields in the GPU solver) without importing CuPy eagerly.
    """
    if type(T).__module__.startswith("cupy"):
        import cupy
        return cupy
    return np


def _polyval(coeffs, T):
    """Evaluate ``sum_i coeffs[i] * T**i`` (ascending powers) by Horner's rule.

    Works for a Python scalar or a NumPy/CuPy array ``T``. ``coeffs`` is a 1-D
    sequence in ascending power order, e.g. ``[c0, c1, c2]`` → ``c0 + c1 T + c2 T²``.
    """
    result = T * 0.0 + float(coeffs[-1])
    for c in reversed(coeffs[:-1]):
        result = result * T + float(c)
    return result


def liquid_fraction(T, T_solidus, T_liquidus):
    """Liquid fraction ``f_l(T)`` clamped to ``[0, 1]``.

    ``0`` below the solidus, ``1`` above the liquidus, linear in between. When
    ``T_liquidus <= T_solidus`` (no mushy band configured) the ramp degenerates
    to a step at ``T_solidus``.
    """
    if T_liquidus <= T_solidus:
        # No mushy band: step from solid to liquid at the solidus.
        xp = _xp(T)
        return xp.where(xp.asarray(T) >= T_solidus, 1.0, 0.0).astype(np.float32) \
            if hasattr(T, "shape") else (1.0 if T >= T_solidus else 0.0)
    f = (T - T_solidus) / (T_liquidus - T_solidus)
    if hasattr(f, "clip"):
        return f.clip(0.0, 1.0)
    return min(max(f, 0.0), 1.0)


@dataclass
class TempProperty:
    """A single temperature-dependent property with solid/liquid polynomial branches.

    Attributes
    ----------
    solid, liquid : np.ndarray
        Polynomial coefficients (ascending powers of ``T``) for each phase.
    T_solidus, T_liquidus : float
        Mushy-band limits used to blend the branches via :func:`liquid_fraction`.
    is_constant : bool
        True when both branches are the same single constant — lets the solver
        take the constant-coefficient fast path.
    """

    solid: np.ndarray
    liquid: np.ndarray
    T_solidus: float
    T_liquidus: float
    is_constant: bool = False

    def __call__(self, T):
        """Evaluate the blended property at temperature(s) *T* (scalar or array)."""
        if self.is_constant:
            return T * 0.0 + float(self.solid[0])
        ps = _polyval(self.solid, T)
        pl = _polyval(self.liquid, T)
        fl = liquid_fraction(T, self.T_solidus, self.T_liquidus)
        return (1.0 - fl) * ps + fl * pl

    @classmethod
    def from_config(cls, spec: Any, T_solidus: float, T_liquidus: float) -> "TempProperty":
        """Build a :class:`TempProperty` from a config value.

        Accepts:
        - a plain number, or a ``{value, unit}`` mapping → constant property;
        - a ``{solid: [...], liquid: [...]}`` mapping → polynomial branches
          (a missing branch falls back to the other).
        """
        # {value, unit} or {solid, liquid} mapping.
        if isinstance(spec, dict) and ("solid" in spec or "liquid" in spec):
            solid = spec.get("solid", spec.get("liquid"))
            liquid = spec.get("liquid", spec.get("solid"))
            solid = np.asarray(solid, dtype=np.float64).ravel()
            liquid = np.asarray(liquid, dtype=np.float64).ravel()
            if solid.size == 0 or liquid.size == 0:
                raise ValueError("TempProperty branch coefficient list must be non-empty.")
            if not (np.all(np.isfinite(solid)) and np.all(np.isfinite(liquid))):
                raise ValueError("TempProperty coefficients must be finite.")
            # Degenerate case: both branches are the same single constant — flag
            # it so the solver can still take the constant-coefficient fast path.
            is_const = (solid.size == 1 and liquid.size == 1
                        and solid[0] == liquid[0])
            return cls(solid=solid, liquid=liquid,
                       T_solidus=float(T_solidus), T_liquidus=float(T_liquidus),
                       is_constant=is_const)

        # Scalar (plain number or {value, unit}).
        value = spec["value"] if isinstance(spec, dict) else spec
        coeff = np.asarray([float(value)], dtype=np.float64)
        return cls(solid=coeff, liquid=coeff.copy(),
                   T_solidus=float(T_solidus), T_liquidus=float(T_liquidus),
                   is_constant=True)


@dataclass
class MaterialModel:
    """Aggregate of the temperature-dependent thermophysical properties.

    Bundles ``k(T)``, ``rho(T)``, ``c(T)`` and the derived volumetric sensible
    heat capacity ``a(T) = rho(T) * c(T)``. Used by the property-correction path
    of the spectral solver; ``None`` on a material with only scalar properties.
    """

    k: TempProperty
    rho: TempProperty
    c: TempProperty

    def a(self, T):
        """Volumetric sensible heat capacity ``a(T) = rho(T) * c(T)``."""
        return self.rho(T) * self.c(T)

    @property
    def is_constant(self) -> bool:
        """True when every property is a single constant (null-correction path)."""
        return self.k.is_constant and self.rho.is_constant and self.c.is_constant

    def reference_constants(self, T0: float):
        """Reference scalars evaluated at the initial temperature *T0*.

        Returns ``(k_bar, a_bar, rho_bar, c_bar)``. These are the constants that
        **must** be baked into the ETD1 propagators ``K, KK`` so the exact
        forcing identity holds (see ``property_correction.tex`` §6.1).
        """
        k_bar = float(self.k(T0))
        rho_bar = float(self.rho(T0))
        c_bar = float(self.c(T0))
        return k_bar, rho_bar * c_bar, rho_bar, c_bar

    @classmethod
    def from_config(cls, mat_cfg: dict, T_solidus: float, T_liquidus: float
                    ) -> Optional["MaterialModel"]:
        """Build a :class:`MaterialModel` from the ``material`` config section.

        Returns ``None`` when none of ``k``, ``rho``, ``Cp`` is given as
        polynomial branches (i.e. the material is fully constant) — the solver
        then keeps its existing constant-coefficient path.
        """
        def _has_branches(spec):
            return isinstance(spec, dict) and ("solid" in spec or "liquid" in spec)

        specs = {name: mat_cfg.get(key)
                 for name, key in (("k", "k"), ("rho", "rho"), ("c", "Cp"))}
        if not any(_has_branches(s) for s in specs.values()):
            return None

        return cls(
            k=TempProperty.from_config(specs["k"], T_solidus, T_liquidus),
            rho=TempProperty.from_config(specs["rho"], T_solidus, T_liquidus),
            c=TempProperty.from_config(specs["c"], T_solidus, T_liquidus),
        )
