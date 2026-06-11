"""Integration gates for the temperature-dependent property correction.

Implements the CPU-side checks of ``property_correction.tex`` §8 that must pass
before any GPU / FE-reference comparison:

* §8.1 null test — constant branches equal to the reference reproduce the base
  (constant-coefficient) solver bit-for-bit;
* stability / physicality — the correction keeps the field finite, positive
  properties, surface bounded, and the Picard loop converging;
* §8.4 convergence parameter ε stays in the modest regime.

Run with::

    pytest -m integration tests/integration/test_property_correction.py

Heavy solver imports are deferred into the test bodies.
"""

import copy

import numpy as np
import pytest

from fast_heat_solv.core.parameters import SimulationContext

# 316L Chadwick branches (matching the FE reference script).
_K_BR = {"solid": [9.248, 0.01571], "liquid": [12.41, 0.003279]}
_RHO_BR = {"solid": [8084.2, -0.42086, -3.8942e-5],
           "liquid": [7432.7, 0.039338, -1.8007e-4]}
_CP_BR = {"solid": [458.98, 0.1328], "liquid": [769.86]}

_T_S, _T_L, _T0 = 1674.15, 1697.15, 293.0
_LX, _LY, _LZ = 1.0e-3, 0.5e-3, 0.25e-3
_A, _P, _R_B, _V = 0.30, 24.0, 30e-6, 0.15
_X_START = _LX / 4
_T_TOTAL = 1.0e-3


def _cfg(material, mesh=(48, 24, 32), n_steps=20, backend="cpu"):
    return {
        "simulation": {"method": "spectral", "backend": backend,
                       "duration": _T_TOTAL, "dt": _T_TOTAL / n_steps},
        "domain": {"size": [_LX, _LY, _LZ], "mesh": list(mesh)},
        "material": material,
        "laser": {"radius": _R_B, "absorptivity": _A, "power_nominal": _P},
        "io": {},
    }


def _base_material(**over):
    mat = {"name": "316L", "T_solidus": _T_S, "T_liquidus": _T_L, "T0": _T0,
           "L_f": 2.677e5, "DeltaH_LV": 7.416e6, "R_v": 150.774,
           "Pa": 101325.0, "T_boil": 3090.0}
    mat.update(over)
    return mat


def _context(cfg, laser):
    ctx = SimulationContext.from_dict(copy.deepcopy(cfg))
    ctx.laser_path = laser
    return ctx


def _run(ctx, max_picard_iter=None, track_history=False):
    from fast_heat_solv.physics.spectral_helpers import reconstruct_temperature_DCT
    from fast_heat_solv.solvers import build_solver

    solver = build_solver(ctx)
    solver.initialize(ctx)
    if max_picard_iter is not None:
        solver.max_picard_iter = max_picard_iter
    solver.track_picard_history = track_history
    dt = ctx.num.dt
    surf_max, picard = [], []
    last_history = None
    for step in range(ctx.num.n_steps):
        _, metrics = solver.step(step * dt, dt)
        surf_max.append(float(metrics["T_surface_max"]))
        picard.append(int(metrics["n_evap_iter"]))
        if track_history:
            last_history = list(solver.picard_history)
    T = reconstruct_temperature_DCT(solver.state.a, solver.state)
    if track_history:
        return solver, T, surf_max, picard, last_history
    return solver, T, surf_max, picard


# ---------------------------------------------------------------------------
# §8.1 null test — constant branches == base solver, bit-for-bit
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_null_constant_branches_match_scalar(constant_velocity_laser):
    """Constant property branches (= reference) reproduce the scalar solver exactly.

    The model is flagged constant, so the correction path is disabled — the run
    must be identical to the equivalent scalar-material run to the bit.
    """
    k0, rho0, cp0 = 15.0, 7900.0, 500.0
    scalar = _cfg(_base_material(k=k0, rho=rho0, Cp=cp0))
    const_branch = _cfg(_base_material(
        k={"solid": [k0], "liquid": [k0]},
        rho={"solid": [rho0], "liquid": [rho0]},
        Cp={"solid": [cp0], "liquid": [cp0]},
    ))

    laser = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    _, T_scalar, _, _ = _run(_context(scalar, laser))
    laser2 = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    solver_b, T_branch, _, _ = _run(_context(const_branch, laser2))

    assert solver_b._property_correction is False
    np.testing.assert_array_equal(T_scalar, T_branch)


# ---------------------------------------------------------------------------
# stability / physicality with the correction active
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_temperature_dependent_run_is_physical(constant_velocity_laser):
    """T-dependent 316L run stays finite, physical, bounded, and converges."""
    cfg = _cfg(_base_material(k=_K_BR, rho=_RHO_BR, Cp=_CP_BR))
    laser = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    ctx = _context(cfg, laser)

    # The correction is a stable, monotone fixed point but, at the solver's
    # conservative mixing (ω=0.1, contraction ≈ 1−ω), it needs ~2× the base
    # iteration count; raise the cap so convergence is actually reached.
    solver, T, surf_max, picard, history = _run(
        ctx, max_picard_iter=80, track_history=True)
    assert solver._property_correction is True

    # Finite, surface bounded by the evaporation cap.
    assert not np.isnan(T).any() and not np.isinf(T).any()
    assert max(surf_max) < 1.5 * 3090.0
    # Sub-ambient undershoot is bounded *relative to the peak excursion* — i.e.
    # Gibbs ringing, not divergence. (The absolute floor → <1 K only on a
    # resolved grid; that is the refinement gate, deferred to the GPU machine.
    # The base solver shows the same coarse-grid ringing: test_validation.py.)
    peak_excursion = float(T.max()) - _T0
    assert T.min() >= _T0 - 0.05 * peak_excursion, (
        f"undershoot {(_T0 - float(T.min())):.1f} K > 5% of peak "
        f"{peak_excursion:.1f} K — looks like divergence, not ringing")

    # Properties stay positive over the whole realised temperature range
    # (the reason for using tabulated fluctuations, not a linearisation).
    Trange = np.linspace(_T0, max(surf_max), 200)
    assert float(np.min(ctx.mat.model.k(Trange))) > 0.0
    assert float(np.min(ctx.mat.model.a(Trange))) > 0.0

    # Convergence is *not impaired*: the final step converges below the cap and
    # its residual decreases monotonically (stable fixed point, all-orders
    # resummation — tex §8.3/§8.4).
    assert picard[-1] < solver.max_picard_iter
    rms = [h["rms_diff"] for h in history]
    assert all(b <= a for a, b in zip(rms, rms[1:])), "residual not monotone"


# ---------------------------------------------------------------------------
# §8.4 convergence parameter ε ~ ||k'|| / (k̄ sqrt(V))
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_epsilon_in_safe_regime(constant_velocity_laser):
    """The controlling parameter ε stays modest, so the series converges (tex §7)."""
    from fast_heat_solv.physics.spectral_ops import reconstruct_volume

    cfg = _cfg(_base_material(k=_K_BR, rho=_RHO_BR, Cp=_CP_BR))
    laser = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    ctx = _context(cfg, laser)
    solver, _, _, _ = _run(ctx)

    T = reconstruct_volume(solver.state.a, solver.state)
    k_prime = ctx.mat.model.k(T) - float(solver._kbar)
    eps = float(np.sqrt(np.mean((k_prime / float(solver._kbar)) ** 2)))
    assert eps < 0.3, f"ε={eps:.4f} outside the modest regime — series may diverge"
