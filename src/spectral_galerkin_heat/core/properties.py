"""Temperature-dependent material properties (expression branches + f_l blend).

Each property ``k(T)``, ``rho(T)``, ``c(T)`` is described by a solid-branch and a
liquid-branch expression of ``T`` blended pointwise by the liquid fraction ``f_l(T)`` 
between the solidus and liquidus temperatures::

    p(T) = (1 - f_l) * p_solid(T) + f_l * p_liquid(T)

This matches the reference 316L form used by the finite-element reference solver.
The blended property is non-polynomial with slope kinks at ``T_S``/``T_L``; this
never enters the spectral scheme, because the weak-form property correction
samples ``k`` and ``a`` but never differentiates them in ``T``.

A plain scalar is also accepted (both branches collapse to the constant), so
existing constant-property configs keep working unchanged.
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

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from spectral_galerkin_heat.core.property_expr import (
    _C_SUFFIX,
    _C_TYPE,
    _PY_CAST,
    PropertyExpr,
    _precision,
    _xp,
)

__all__ = [
    "MaterialModel",
    "TempProperty",
    "liquid_fraction",
]


def liquid_fraction(T, T_solidus, T_liquidus):
    """Liquid fraction ``f_l(T)`` clamped to ``[0, 1]``.

    ``0`` below the solidus, ``1`` above the liquidus, linear in between. When
    ``T_liquidus <= T_solidus`` (no mushy band configured) the ramp degenerates
    to a step at ``T_solidus``.
    """
    if T_liquidus <= T_solidus:
        # No mushy band: step from solid to liquid at the solidus.
        xp = _xp(T)
        return xp.where(xp.asarray(T) >= T_solidus, 1.0, 0.0).astype(T.dtype) \
            if hasattr(T, "shape") else (1.0 if T >= T_solidus else 0.0)
    f = (T - T_solidus) / (T_liquidus - T_solidus)
    if hasattr(f, "clip"):
        return f.clip(0.0, 1.0)
    return min(max(f, 0.0), 1.0)


@dataclass
class TempProperty:
    """A single temperature-dependent property with solid/liquid branch expressions.

    Attributes
    ----------
    solid, liquid : PropertyExpr
        Math expression of ``T`` for each phase (see
        :class:`spectral_galerkin_heat.core.property_expr.PropertyExpr`).
    T_solidus, T_liquidus : float
        Mushy-band limits used to blend parameters via :func:`liquid_fraction`.
    is_constant : bool
        True when both branches are the same single constant, which lets the
        solver take the constant-coefficient fast path.
    """

    solid: PropertyExpr
    liquid: PropertyExpr
    T_solidus: float
    T_liquidus: float
    is_constant: bool = False

    def __call__(self, T):
        """Evaluate the blended property at temperature(s) *T* (scalar or array)."""
        if self.is_constant:
            return self.solid(T)
        ps = self.solid(T)
        pl = self.liquid(T)
        fl = liquid_fraction(T, self.T_solidus, self.T_liquidus)
        return (1.0 - fl) * ps + fl * pl

    @classmethod
    def from_config(cls, spec: Any, T_solidus: float, T_liquidus: float) -> "TempProperty":
        """Build a :class:`TempProperty` from a config value.

        Accepts:
        - a plain number, or a ``{value, unit}`` mapping → constant property;
        - a ``{solid: "<expr of T>", liquid: "<expr of T>"}`` mapping → branch
          expressions (a missing branch falls back to the other).

        A branch mapping may also carry ``reference:`` (the constant k̄/ρ̄/C̄p,
        read by the material parser) and ``unit:``. Both are ignored here.
        """
        # {value, unit} or {solid, liquid} mapping.
        if isinstance(spec, dict) and ("solid" in spec or "liquid" in spec):
            solid = PropertyExpr(spec.get("solid", spec.get("liquid")))
            liquid = PropertyExpr(spec.get("liquid", spec.get("solid")))
            # Degenerate case: both branches are the same single constant. Flag
            # it so the solver can still take the constant-coefficient fast path.
            is_const = (solid.is_constant and liquid.is_constant
                        and solid.constant_value == liquid.constant_value)
            return cls(solid=solid, liquid=liquid,
                       T_solidus=float(T_solidus), T_liquidus=float(T_liquidus),
                       is_constant=is_const)

        # Scalar (plain number or {value, unit}).
        value = spec["value"] if isinstance(spec, dict) else spec
        const = PropertyExpr(value)
        return cls(solid=const, liquid=const,
                   T_solidus=float(T_solidus), T_liquidus=float(T_liquidus),
                   is_constant=True)


# ---------------------------------------------------------------------------
# Numba-accelerated CPU fast path for the property fluctuations.
#
# We evaluate ``k' = k(T)-k̄`` and ``a' = rho·c-ā`` in one
# fused, ``prange``-parallel pass at the configured precision.
# Each property is a free-form expression.
# ---------------------------------------------------------------------------
from numba import njit as _njit  # noqa: E402  (deliberately after the note above)
from numba import prange as _prange  # noqa: E402


def _build_kp_ap_kernel(k_solid, k_liquid, rho_solid, rho_liquid, cp_solid,
                        cp_liquid, dtype=np.float32):
    """Compile a fused njit ``k'(T), a'(T)`` kernel from six branch expressions.

    Parameters
    ----------
    k_solid, k_liquid, rho_solid, rho_liquid, cp_solid, cp_liquid : PropertyExpr
        The six branch expressions written into the kernel source.
    dtype : data-type, optional
        Precision of the emitted literals, ``numpy.float32`` (default) or
        ``numpy.float64``. Must match the dtype of the arrays passed at call
        time, or numba will compile a second, promoted specialisation.
    """
    cast = _precision(dtype, _PY_CAST)
    src = f"""
def _kp_ap_kernel(Tfield, t_sol, inv_band, degenerate, k_bar, a_bar, kp_out, ap_out):
    nz, ny, nx = Tfield.shape
    one = {cast}(1.0)
    zero = {cast}(0.0)
    for kk in prange(nz):
        for j in range(ny):
            for i in range(nx):
                T = Tfield[kk, j, i]
                if degenerate:
                    fl = one if T >= t_sol else zero
                else:
                    fl = (T - t_sol) * inv_band
                    if fl < zero:
                        fl = zero
                    elif fl > one:
                        fl = one
                om = one - fl
                kv = om * ({k_solid.to_py_source(dtype)}) + fl * ({k_liquid.to_py_source(dtype)})
                rv = om * ({rho_solid.to_py_source(dtype)}) + fl * ({rho_liquid.to_py_source(dtype)})
                cv = om * ({cp_solid.to_py_source(dtype)}) + fl * ({cp_liquid.to_py_source(dtype)})
                kp_out[kk, j, i] = kv - k_bar
                ap_out[kk, j, i] = rv * cv - a_bar
"""
    namespace = {"np": np, "prange": _prange}
    exec(compile(src, "<property_kp_ap_kernel>", "exec"), namespace)
    return _njit(parallel=True, fastmath=True)(namespace["_kp_ap_kernel"])


def _build_latent_kernel(rho_solid, rho_liquid, dtype=np.float32):
    """Compile a fused njit latent-heat-source kernel from the rho branches.

    Parameters
    ----------
    rho_solid, rho_liquid : PropertyExpr
        The density branches written into the kernel source.
    dtype : data-type, optional
        Precision of the emitted literals, ``numpy.float32`` (default) or
        ``numpy.float64``.
    """
    cast = _precision(dtype, _PY_CAST)
    src = f"""
def _latent_kernel(Tfield, Tprev, t_sol, inv_band, degenerate, Lf, inv_dt, out):
    nz, ny, nx = Tfield.shape
    one = {cast}(1.0)
    zero = {cast}(0.0)
    for kk in prange(nz):
        for j in range(ny):
            for i in range(nx):
                T = Tfield[kk, j, i]
                Tp = Tprev[kk, j, i]
                if degenerate:
                    flc = one if T >= t_sol else zero
                    flp = one if Tp >= t_sol else zero
                else:
                    flc = (T - t_sol) * inv_band
                    if flc < zero:
                        flc = zero
                    elif flc > one:
                        flc = one
                    flp = (Tp - t_sol) * inv_band
                    if flp < zero:
                        flp = zero
                    elif flp > one:
                        flp = one
                om = one - flc
                rho = om * ({rho_solid.to_py_source(dtype)}) + flc * ({rho_liquid.to_py_source(dtype)})
                out[kk, j, i] = -rho * Lf * (flc - flp) * inv_dt
"""
    namespace = {"np": np, "prange": _prange}
    exec(compile(src, "<property_latent_kernel>", "exec"), namespace)
    return _njit(parallel=True, fastmath=True)(namespace["_latent_kernel"])


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

    def _cpu_band_params(self, dtype=np.float32):
        """Cached mushy-band parameters (t_sol, inv_band, degenerate) for the CPU kernels.

        The fused kernels blend all three properties with one liquid fraction, so
        they must share a single mushy band (guaranteed by ``from_config``, which
        builds them from the same solidus/liquidus).
        """
        cache = getattr(self, "_cpu_band_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_cpu_band_cache", cache)
        hit = cache.get(np.dtype(dtype))
        if hit is not None:
            return hit
        bands = {(p.T_solidus, p.T_liquidus) for p in (self.k, self.rho, self.c)}
        if len(bands) != 1:
            raise ValueError(
                "k, rho and c must share one mushy band for the fused CPU "
                f"property kernel; got bands {bands}.")
        t_sol, t_liq = self.k.T_solidus, self.k.T_liquidus
        degenerate = t_liq <= t_sol
        real_t = np.dtype(dtype).type
        inv_band = real_t(0.0 if degenerate else 1.0 / (t_liq - t_sol))
        hit = (real_t(t_sol), inv_band, bool(degenerate))
        cache[np.dtype(dtype)] = hit
        return hit

    def _cpu_kp_ap_kernel(self, dtype=np.float32):
        """Build (and cache) the fused njit ``k',a'`` kernel for this model.

        One kernel is compiled per precision: a float32 and a float64
        run need different kernels.
        """
        cache = getattr(self, "_kp_ap_cpu_kernel", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_kp_ap_cpu_kernel", cache)
        key = np.dtype(dtype)
        kern = cache.get(key)
        if kern is None:
            kern = _build_kp_ap_kernel(self.k.solid, self.k.liquid,
                                       self.rho.solid, self.rho.liquid,
                                       self.c.solid, self.c.liquid, dtype=key)
            cache[key] = kern
        return kern

    def _cpu_latent_kernel(self, dtype=np.float32):
        """Build (and cache) the fused njit latent-heat-source kernel per precision."""
        cache = getattr(self, "_latent_cpu_kernel", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_latent_cpu_kernel", cache)
        key = np.dtype(dtype)
        kern = cache.get(key)
        if kern is None:
            kern = _build_latent_kernel(self.rho.solid, self.rho.liquid, dtype=key)
            cache[key] = kern
        return kern

    def k_prime_a_prime(self, T, k_bar, a_bar):
        """Fluctuations ``(k(T) - k_bar, a(T) - a_bar)`` in a single fused pass.
        """
        xp = _xp(T)
        real_t = np.dtype(getattr(T, "dtype", np.float32))
        scalar = real_t.type
        if xp is np:
            t_sol, inv_band, degenerate = self._cpu_band_params(real_t)
            kern = self._cpu_kp_ap_kernel(real_t)
            Tf = np.ascontiguousarray(T, dtype=real_t)
            kp = np.empty(Tf.shape, dtype=real_t)
            ap = np.empty(Tf.shape, dtype=real_t)
            kern(Tf, t_sol, inv_band, degenerate,
                 scalar(k_bar), scalar(a_bar), kp, ap)
            return kp, ap
        kern = self._fused_gpu_kernel(real_t)
        kp = xp.empty(T.shape, dtype=real_t)
        ap = xp.empty(T.shape, dtype=real_t)
        kern(T, scalar(k_bar), scalar(a_bar), kp, ap)
        return kp, ap

    def _fused_gpu_kernel(self, dtype=np.float32):
        """Build (and cache) the fused CuPy kernel for ``k', a'`` per precision.
        """
        cache = getattr(self, "_kp_ap_kernel", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_kp_ap_kernel", cache)
        key = np.dtype(dtype)
        kern = cache.get(key)
        if kern is not None:
            return kern
        import cupy

        ct = _precision(key, _C_TYPE)
        sfx = _precision(key, _C_SUFFIX)
        Ts, Tl = float(self.k.T_solidus), float(self.k.T_liquidus)
        if Tl > Ts:
            fl = (f"{ct} fl = (T-{Ts:.9e}{sfx})*{1.0/(Tl-Ts):.9e}{sfx}; "
                  f"fl = fl<0.0{sfx}?0.0{sfx}:(fl>1.0{sfx}?1.0{sfx}:fl);")
        else:  # degenerate mushy band -> step at the solidus
            fl = f"{ct} fl = (T>={Ts:.9e}{sfx})?1.0{sfx}:0.0{sfx};"
        body = f"""
        {fl}
        {ct} omfl = 1.0{sfx} - fl;
        {ct} kk  = omfl*({self.k.solid.to_c_source(key)})   + fl*({self.k.liquid.to_c_source(key)});
        {ct} rho = omfl*({self.rho.solid.to_c_source(key)}) + fl*({self.rho.liquid.to_c_source(key)});
        {ct} cc  = omfl*({self.c.solid.to_c_source(key)})   + fl*({self.c.liquid.to_c_source(key)});
        kp = kk - kbar;
        ap = rho*cc - abar;
        """
        kern = cupy.ElementwiseKernel(
            f"{key.name} T, {key.name} kbar, {key.name} abar",
            f"{key.name} kp, {key.name} ap", body, f"prop_kp_ap_{key.name}")
        cache[key] = kern
        return kern

    def latent_heat_source(self, T, T_prev, T_S, T_L, L_f, dt, out):
        """Latent-heat sink ``Q = -rho(T) L_f (f_l(T)-f_l(T_prev))/dt`` into *out*.
        """
        xp = _xp(T)
        if T_L > T_S:
            inv_band = 1.0 / (T_L - T_S)
        else:  # degenerate mushy band -> step at the solidus
            inv_band = -1.0
        real_t = np.dtype(getattr(out, "dtype", getattr(T, "dtype", np.float32)))
        scalar = real_t.type
        if xp is np:
            degenerate = inv_band < 0.0
            kern = self._cpu_latent_kernel(real_t)
            kern(np.ascontiguousarray(T, dtype=real_t),
                 np.ascontiguousarray(T_prev, dtype=real_t),
                 scalar(T_S), scalar(0.0 if degenerate else inv_band),
                 bool(degenerate), scalar(L_f), scalar(1.0 / dt), out)
            return
        kern = self._latent_source_kernel(real_t)
        kern(T, T_prev, scalar(T_S), scalar(inv_band),
             scalar(L_f), scalar(dt), out)

    def _latent_source_kernel(self, dtype=np.float32):
        """Build (and cache) the fused CuPy latent-heat-source kernel per precision."""
        cache = getattr(self, "_lh_kernel", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_lh_kernel", cache)
        key = np.dtype(dtype)
        kern = cache.get(key)
        if kern is not None:
            return kern
        import cupy

        ct = _precision(key, _C_TYPE)
        sfx = _precision(key, _C_SUFFIX)
        # Mushy band (Ts, inv) passed at call time; inv < 0 flags the degenerate
        # no-band case (step at the solidus). Only the rho branches are inlined.
        clc = (f"flc = inv<0.0{sfx} ? (T>=Ts?1.0{sfx}:0.0{sfx}) "
               f": fmin{sfx}(fmax{sfx}((T-Ts)*inv,0.0{sfx}),1.0{sfx});")
        clp = (f"flp = inv<0.0{sfx} ? (Tp>=Ts?1.0{sfx}:0.0{sfx}) "
               f": fmin{sfx}(fmax{sfx}((Tp-Ts)*inv,0.0{sfx}),1.0{sfx});")
        body = f"""
        {ct} flc, flp;
        {clc}
        {clp}
        {ct} omfl = 1.0{sfx} - flc;
        {ct} rho = omfl*({self.rho.solid.to_c_source(key)}) + flc*({self.rho.liquid.to_c_source(key)});
        out = -rho * Lf * (flc - flp) / dt;
        """
        kern = cupy.ElementwiseKernel(
            f"{key.name} T, {key.name} Tp, {key.name} Ts, {key.name} inv, "
            f"{key.name} Lf, {key.name} dt",
            f"{key.name} out", body, f"latent_heat_source_{key.name}")
        cache[key] = kern
        return kern

    @property
    def is_constant(self) -> bool:
        """True when every property is a single constant (null-correction path)."""
        return self.k.is_constant and self.rho.is_constant and self.c.is_constant

    def recommended_references(self, T_lo: float, T_hi: float,
                               n_samples: int = 512):
        """Recommended reference scalars over the working band ``[T_lo, T_hi]``.

        The user supplies each T-dependent property's reference constant
        explicitly, as a ``reference:`` key inside its config block. This helper
        only computes a recommendation for each: the midpoint of the property's
        own extrema sampled across the range, ``½(min p + max p)``. Centering every
        reference in the middle of its working range reduces the peak fluctuations
        and helps convergence.

        Returns the recommended ``(k, rho, Cp)`` references.
        """
        T = np.linspace(float(T_lo), float(T_hi), int(n_samples))

        def _mid(prop):
            vals = np.asarray(prop(T), dtype=np.float64)
            return 0.5 * (float(vals.min()) + float(vals.max()))

        return _mid(self.k), _mid(self.rho), _mid(self.c)

    @classmethod
    def from_config(cls, mat_cfg: dict, T_solidus: float, T_liquidus: float
                    ) -> Optional["MaterialModel"]:
        """Build a :class:`MaterialModel` from the ``material`` config section.

        Returns ``None`` when none of ``k``, ``rho``, ``Cp`` is given as
        solid/liquid branch expressions, i.e. the material is fully constant.
        The solver then keeps its existing constant-coefficient path.
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
