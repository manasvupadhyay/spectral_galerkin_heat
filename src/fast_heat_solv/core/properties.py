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


# ---------------------------------------------------------------------------
# Numba-accelerated CPU fast path for the property fluctuations.
#
# The plain NumPy evaluator allocates ~25 full-volume temporaries and runs
# single-threaded; on the CPU backend (used when a case is too large for GPU
# memory) that single op dominates the per-Picard-iteration cost (~2 s per call
# on a 34M-cell volume). We evaluate ``k' = k(T)-k̄`` and ``a' = rho·c-ā`` in one
# fused, ``prange``-parallel pass — in float32, matching the weak-scalar-promotion
# arithmetic of the GPU ``ElementwiseKernel``. ~100× faster on a many-core node.
# numba is a hard dependency, so this is the only CPU path.
# ---------------------------------------------------------------------------
from numba import njit as _njit, prange as _prange


@_njit(fastmath=True, cache=True)
def _horner32(coeffs, x):
    """Horner evaluation of an ascending-power float32 polynomial at ``x``."""
    acc = coeffs[coeffs.shape[0] - 1]
    for idx in range(coeffs.shape[0] - 2, -1, -1):
        acc = acc * x + coeffs[idx]
    return acc


@_njit(parallel=True, fastmath=True, cache=True)
def _kp_ap_cpu(T, ks, kl, rs, rl, cs, cl,
               t_sol, inv_band, degenerate, k_bar, a_bar, kp_out, ap_out):
    """Fused ``k'(T), a'(T)`` over the volume (float32, prange-parallel)."""
    nz, ny, nx = T.shape
    one = np.float32(1.0)
    zero = np.float32(0.0)
    for kk in _prange(nz):
        for j in range(ny):
            for i in range(nx):
                x = T[kk, j, i]
                if degenerate:
                    fl = one if x >= t_sol else zero
                else:
                    fl = (x - t_sol) * inv_band
                    if fl < zero:
                        fl = zero
                    elif fl > one:
                        fl = one
                om = one - fl
                kv = om * _horner32(ks, x) + fl * _horner32(kl, x)
                rv = om * _horner32(rs, x) + fl * _horner32(rl, x)
                cv = om * _horner32(cs, x) + fl * _horner32(cl, x)
                kp_out[kk, j, i] = kv - k_bar
                ap_out[kk, j, i] = rv * cv - a_bar


@_njit(parallel=True, fastmath=True, cache=True)
def _latent_src_cpu(T, Tprev, rs, rl, t_sol, inv_band, degenerate,
                    Lf, inv_dt, out):
    """Latent-heat sink ``Q = -rho(T) L_f (f_l(T)-f_l(T_prev))/dt`` (float32)."""
    nz, ny, nx = T.shape
    one = np.float32(1.0)
    zero = np.float32(0.0)
    for kk in _prange(nz):
        for j in range(ny):
            for i in range(nx):
                x = T[kk, j, i]
                xp_prev = Tprev[kk, j, i]
                if degenerate:
                    flc = one if x >= t_sol else zero
                    flp = one if xp_prev >= t_sol else zero
                else:
                    flc = (x - t_sol) * inv_band
                    if flc < zero:
                        flc = zero
                    elif flc > one:
                        flc = one
                    flp = (xp_prev - t_sol) * inv_band
                    if flp < zero:
                        flp = zero
                    elif flp > one:
                        flp = one
                om = one - flc
                rho = om * _horner32(rs, x) + flc * _horner32(rl, x)
                out[kk, j, i] = -rho * Lf * (flc - flp) * inv_dt


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

    def _cpu_kp_ap_coeffs(self):
        """Cached float32 coefficients for the numba CPU ``k',a'`` kernel.

        The fused kernel blends all three properties with one liquid fraction, so
        they must share a single mushy band (guaranteed by ``from_config``, which
        builds them from the same solidus/liquidus).
        """
        cache = getattr(self, "_cpu_coeffs_cache", None)
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
        f32 = lambda arr: np.ascontiguousarray(arr, dtype=np.float32)
        cache = (f32(self.k.solid), f32(self.k.liquid),
                 f32(self.rho.solid), f32(self.rho.liquid),
                 f32(self.c.solid), f32(self.c.liquid),
                 np.float32(t_sol), inv_band, bool(degenerate))
        object.__setattr__(self, "_cpu_coeffs_cache", cache)
        return cache

    def _cpu_rho_coeffs(self):
        """Cached float32 rho branch coefficients for the numba latent kernel."""
        cache = getattr(self, "_cpu_rho_cache", None)
        if cache is None:
            cache = (np.ascontiguousarray(self.rho.solid, dtype=np.float32),
                     np.ascontiguousarray(self.rho.liquid, dtype=np.float32))
            object.__setattr__(self, "_cpu_rho_cache", cache)
        return cache

    def k_prime_a_prime(self, T, k_bar, a_bar):
        """Fluctuations ``(k(T) - k_bar, a(T) - a_bar)`` in a single fused pass.

        The straightforward ``k(T)``/``a(T)`` evaluation allocates ~25 temporary
        full-volume arrays (Horner + liquid-fraction blend, twice for ``a=rho*c``);
        on GPU that is ~140 ms per Picard iteration. This fuses the whole thing
        into one kernel (built once from the model's coefficients), cutting it to a
        few ms: a numba ``prange`` pass on CPU, an ``ElementwiseKernel`` on GPU.
        """
        xp = _xp(T)
        if xp is np:
            ks, kl, rs, rl, cs, cl, t_sol, inv_band, degenerate = self._cpu_kp_ap_coeffs()
            Tf = np.ascontiguousarray(T, dtype=np.float32)
            kp = np.empty(Tf.shape, dtype=np.float32)
            ap = np.empty(Tf.shape, dtype=np.float32)
            _kp_ap_cpu(Tf, ks, kl, rs, rl, cs, cl, t_sol, inv_band,
                       degenerate, np.float32(k_bar), np.float32(a_bar), kp, ap)
            return kp, ap
        import cupy
        kern = self._fused_gpu_kernel()
        kp = cupy.empty(T.shape, dtype=cupy.float32)
        ap = cupy.empty(T.shape, dtype=cupy.float32)
        kern(T.astype(cupy.float32, copy=False),
             cupy.float32(k_bar), cupy.float32(a_bar), kp, ap)
        return kp, ap

    def _fused_gpu_kernel(self):
        """Build (and cache) the fused CuPy kernel for ``k', a'`` from the coeffs."""
        kern = getattr(self, "_kp_ap_kernel", None)
        if kern is not None:
            return kern
        import cupy

        def horner(coeffs):
            # ascending coeffs -> nested Horner C expression in variable T
            c = [float(v) for v in coeffs]
            expr = f"{c[-1]:.9e}f"
            for v in reversed(c[:-1]):
                expr = f"({v:.9e}f + T*{expr})"
            return expr

        Ts, Tl = float(self.k.T_solidus), float(self.k.T_liquidus)
        if Tl > Ts:
            fl = (f"float fl = (T-{Ts:.9e}f)*{1.0/(Tl-Ts):.9e}f; "
                  f"fl = fl<0.0f?0.0f:(fl>1.0f?1.0f:fl);")
        else:  # degenerate mushy band -> step at the solidus
            fl = f"float fl = (T>={Ts:.9e}f)?1.0f:0.0f;"
        body = f"""
        {fl}
        float omfl = 1.0f - fl;
        float kk  = omfl*{horner(self.k.solid)}   + fl*{horner(self.k.liquid)};
        float rho = omfl*{horner(self.rho.solid)} + fl*{horner(self.rho.liquid)};
        float cc  = omfl*{horner(self.c.solid)}   + fl*{horner(self.c.liquid)};
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
            rs, rl = self._cpu_rho_coeffs()
            _latent_src_cpu(np.ascontiguousarray(T, dtype=f32),
                            np.ascontiguousarray(T_prev, dtype=f32),
                            rs, rl, f32(T_S),
                            f32(0.0 if degenerate else inv_band),
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

        def horner(coeffs):
            c = [float(v) for v in coeffs]
            expr = f"{c[-1]:.9e}f"
            for v in reversed(c[:-1]):
                expr = f"({v:.9e}f + T*{expr})"
            return expr

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
        float rho = omfl*{horner(self.rho.solid)} + flc*{horner(self.rho.liquid)};
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
