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

from fast_heat_solv.core.property_expr import PropertyExpr, _xp

__all__ = [
    "liquid_fraction",
    "TempProperty",
    "MaterialModel",
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
        return xp.where(xp.asarray(T) >= T_solidus, 1.0, 0.0).astype(np.float32) \
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
        :class:`fast_heat_solv.core.property_expr.PropertyExpr`).
    T_solidus, T_liquidus : float
        Mushy-band limits used to blend the branches via :func:`liquid_fraction`.
    is_constant : bool
        True when both branches are the same single constant — lets the solver
        take the constant-coefficient fast path.
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

        A branch mapping may also carry ``reference:`` (the baked reference
        constant k̄/ρ̄/C̄p, consumed by the material parser, see
        ``SimulationContext.from_dict``) and ``unit:`` (documentary). Both are
        metadata for the evaluator and are ignored here.
        """
        # {value, unit} or {solid, liquid} mapping.
        if isinstance(spec, dict) and ("solid" in spec or "liquid" in spec):
            solid = PropertyExpr(spec.get("solid", spec.get("liquid")))
            liquid = PropertyExpr(spec.get("liquid", spec.get("solid")))
            # Degenerate case: both branches are the same single constant — flag
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
# The plain NumPy evaluator allocates ~25 full-volume temporaries and runs
# single-threaded; on the CPU backend (used when a case is too large for GPU
# memory) that single op dominates the per-Picard-iteration cost (~2 s per call
# on a 34M-cell volume). We evaluate ``k' = k(T)-k̄`` and ``a' = rho·c-ā`` in one
# fused, ``prange``-parallel pass — in float32, matching the weak-scalar-promotion
# arithmetic of the GPU ``ElementwiseKernel``. ~100× faster on a many-core CPU.
#
# Each property is a free-form expression, not a fixed-shape polynomial, so the
# kernel can't be a single static ``@njit`` function: it is generated (its
# source embeds the six branch expressions verbatim, via ``PropertyExpr.to_py_source``)
# and JIT-compiled once per ``MaterialModel``, then cached on the instance —
# exactly mirroring how ``_fused_gpu_kernel`` below builds and caches a bespoke
# CUDA kernel per model. numba is a hard dependency, so this is the only CPU path.
# ---------------------------------------------------------------------------
from numba import njit as _njit, prange as _prange


def _build_kp_ap_kernel(k_solid, k_liquid, rho_solid, rho_liquid, cp_solid, cp_liquid):
    """Compile a fused njit ``k'(T), a'(T)`` kernel from six branch expressions."""
    src = f"""
def _kp_ap_kernel(Tfield, t_sol, inv_band, degenerate, k_bar, a_bar, kp_out, ap_out):
    nz, ny, nx = Tfield.shape
    one = np.float32(1.0)
    zero = np.float32(0.0)
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
                kv = om * ({k_solid.to_py_source()}) + fl * ({k_liquid.to_py_source()})
                rv = om * ({rho_solid.to_py_source()}) + fl * ({rho_liquid.to_py_source()})
                cv = om * ({cp_solid.to_py_source()}) + fl * ({cp_liquid.to_py_source()})
                kp_out[kk, j, i] = kv - k_bar
                ap_out[kk, j, i] = rv * cv - a_bar
"""
    namespace = {"np": np, "prange": _prange}
    exec(compile(src, "<property_kp_ap_kernel>", "exec"), namespace)
    return _njit(parallel=True, fastmath=True)(namespace["_kp_ap_kernel"])


def _build_latent_kernel(rho_solid, rho_liquid):
    """Compile a fused njit latent-heat-source kernel from the rho branches."""
    src = f"""
def _latent_kernel(Tfield, Tprev, t_sol, inv_band, degenerate, Lf, inv_dt, out):
    nz, ny, nx = Tfield.shape
    one = np.float32(1.0)
    zero = np.float32(0.0)
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
                rho = om * ({rho_solid.to_py_source()}) + flc * ({rho_liquid.to_py_source()})
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

    def _cpu_band_params(self):
        """Cached mushy-band parameters (t_sol, inv_band, degenerate) for the CPU kernels.

        The fused kernels blend all three properties with one liquid fraction, so
        they must share a single mushy band (guaranteed by ``from_config``, which
        builds them from the same solidus/liquidus).
        """
        cache = getattr(self, "_cpu_band_cache", None)
        if cache is not None:
            return cache
        bands = {(p.T_solidus, p.T_liquidus) for p in (self.k, self.rho, self.c)}
        if len(bands) != 1:
            raise ValueError(
                "k, rho and c must share one mushy band for the fused CPU "
                f"property kernel; got bands {bands}.")
        t_sol, t_liq = self.k.T_solidus, self.k.T_liquidus
        degenerate = t_liq <= t_sol
        inv_band = np.float32(0.0 if degenerate else 1.0 / (t_liq - t_sol))
        cache = (np.float32(t_sol), inv_band, bool(degenerate))
        object.__setattr__(self, "_cpu_band_cache", cache)
        return cache

    def _cpu_kp_ap_kernel(self):
        """Build (and cache) the fused njit ``k',a'`` kernel for this model."""
        kern = getattr(self, "_kp_ap_cpu_kernel", None)
        if kern is not None:
            return kern
        kern = _build_kp_ap_kernel(self.k.solid, self.k.liquid,
                                    self.rho.solid, self.rho.liquid,
                                    self.c.solid, self.c.liquid)
        object.__setattr__(self, "_kp_ap_cpu_kernel", kern)
        return kern

    def _cpu_latent_kernel(self):
        """Build (and cache) the fused njit latent-heat-source kernel for this model."""
        kern = getattr(self, "_latent_cpu_kernel", None)
        if kern is not None:
            return kern
        kern = _build_latent_kernel(self.rho.solid, self.rho.liquid)
        object.__setattr__(self, "_latent_cpu_kernel", kern)
        return kern

    def k_prime_a_prime(self, T, k_bar, a_bar):
        """Fluctuations ``(k(T) - k_bar, a(T) - a_bar)`` in a single fused pass.

        The straightforward ``k(T)``/``a(T)`` evaluation allocates ~25 temporary
        full-volume arrays (Horner + liquid-fraction blend, twice for ``a=rho*c``);
        on GPU that is ~140 ms per Picard iteration. This fuses the whole thing
        into one kernel (built once from the model's expressions), cutting it to a
        few ms: a numba ``prange`` pass on CPU, an ``ElementwiseKernel`` on GPU.
        """
        xp = _xp(T)
        if xp is np:
            t_sol, inv_band, degenerate = self._cpu_band_params()
            kern = self._cpu_kp_ap_kernel()
            Tf = np.ascontiguousarray(T, dtype=np.float32)
            kp = np.empty(Tf.shape, dtype=np.float32)
            ap = np.empty(Tf.shape, dtype=np.float32)
            kern(Tf, t_sol, inv_band, degenerate,
                 np.float32(k_bar), np.float32(a_bar), kp, ap)
            return kp, ap
        import cupy
        kern = self._fused_gpu_kernel()
        kp = cupy.empty(T.shape, dtype=cupy.float32)
        ap = cupy.empty(T.shape, dtype=cupy.float32)
        kern(T.astype(cupy.float32, copy=False),
             cupy.float32(k_bar), cupy.float32(a_bar), kp, ap)
        return kp, ap

    def _fused_gpu_kernel(self):
        """Build (and cache) the fused CuPy kernel for ``k', a'`` from the expressions."""
        kern = getattr(self, "_kp_ap_kernel", None)
        if kern is not None:
            return kern
        import cupy

        Ts, Tl = float(self.k.T_solidus), float(self.k.T_liquidus)
        if Tl > Ts:
            fl = (f"float fl = (T-{Ts:.9e}f)*{1.0/(Tl-Ts):.9e}f; "
                  f"fl = fl<0.0f?0.0f:(fl>1.0f?1.0f:fl);")
        else:  # degenerate mushy band -> step at the solidus
            fl = f"float fl = (T>={Ts:.9e}f)?1.0f:0.0f;"
        body = f"""
        {fl}
        float omfl = 1.0f - fl;
        float kk  = omfl*({self.k.solid.to_c_source()})   + fl*({self.k.liquid.to_c_source()});
        float rho = omfl*({self.rho.solid.to_c_source()}) + fl*({self.rho.liquid.to_c_source()});
        float cc  = omfl*({self.c.solid.to_c_source()})   + fl*({self.c.liquid.to_c_source()});
        kp = kk - kbar;
        ap = rho*cc - abar;
        """
        kern = cupy.ElementwiseKernel(
            "float32 T, float32 kbar, float32 abar",
            "float32 kp, float32 ap", body, "prop_kp_ap")
        self._kp_ap_kernel = kern
        return kern

    def latent_heat_source(self, T, T_prev, T_S, T_L, L_f, dt, out):
        """Latent-heat sink ``Q = -rho(T) L_f (f_l(T)-f_l(T_prev))/dt`` into *out*.

        Backend-agnostic like :meth:`k_prime_a_prime`: a fused ``ElementwiseKernel``
        on GPU (avoiding ~5 box-sized temporaries)
        """
        xp = _xp(T)
        if T_L > T_S:
            inv_band = 1.0 / (T_L - T_S)
        else:  # degenerate mushy band -> step at the solidus
            inv_band = -1.0
        if xp is np:
            f32 = np.float32
            degenerate = inv_band < 0.0
            kern = self._cpu_latent_kernel()
            kern(np.ascontiguousarray(T, dtype=f32),
                 np.ascontiguousarray(T_prev, dtype=f32),
                 f32(T_S), f32(0.0 if degenerate else inv_band),
                 bool(degenerate), f32(L_f), f32(1.0 / dt), out)
            return
        import cupy
        kern = self._latent_source_kernel()
        kern(T.astype(cupy.float32, copy=False),
             T_prev.astype(cupy.float32, copy=False),
             cupy.float32(T_S), cupy.float32(inv_band),
             cupy.float32(L_f), cupy.float32(dt), out)

    def _latent_source_kernel(self):
        """Build (and cache) the fused CuPy latent-heat-source kernel."""
        kern = getattr(self, "_lh_kernel", None)
        if kern is not None:
            return kern
        import cupy

        # Mushy band (Ts, inv) passed at call time; inv < 0 flags the degenerate
        # no-band case (step at the solidus). Only the rho branches are baked.
        clc = ("flc = inv<0.0f ? (T>=Ts?1.0f:0.0f) "
               ": fminf(fmaxf((T-Ts)*inv,0.0f),1.0f);")
        clp = ("flp = inv<0.0f ? (Tp>=Ts?1.0f:0.0f) "
               ": fminf(fmaxf((Tp-Ts)*inv,0.0f),1.0f);")
        body = f"""
        float flc, flp;
        {clc}
        {clp}
        float omfl = 1.0f - flc;
        float rho = omfl*({self.rho.solid.to_c_source()}) + flc*({self.rho.liquid.to_c_source()});
        out = -rho * Lf * (flc - flp) / dt;
        """
        kern = cupy.ElementwiseKernel(
            "float32 T, float32 Tp, float32 Ts, float32 inv, float32 Lf, float32 dt",
            "float32 out", body, "latent_heat_source")
        self._lh_kernel = kern
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
        only computes a *recommendation* for each: the midpoint of the property's
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
        solid/liquid branch expressions (i.e. the material is fully constant) —
        the solver then keeps its existing constant-coefficient path.
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
