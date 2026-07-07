"""Physics validation: linear analytical (§2.1) + non-linear sanity (§2.2).

The linear tests anchor *correctness* against the closed-form Eagar-Tsai moving
heat-source solution (``tests/eagar_tsai.py``, method-of-images corrected for the
solver's six insulated walls); the non-linear test guards against regressions
with loose physical bounds (no analytical reference once latent heat /
evaporation are active).

All tests are ``@pytest.mark.integration`` (run the full numba pipeline) and are
excluded from the default suite::

    pytest -m integration tests/integration/test_validation.py

Heavy solver imports are deferred into the test bodies.
"""

import copy

import numpy as np
import pytest

from fast_heat_solv.core.parameters import SimulationContext
from fast_heat_solv.io_utils.compute_L2_error import compute_L2

# ---------------------------------------------------------------------------
# Physical case — the convergence-study setup the Eagar-Tsai solution
# The laser travels in the middle of the domain (x_start = Lx/4)
# ---------------------------------------------------------------------------
_RHO, _K, _CP, _T0 = 7958.0, 13.85, 498.0, 293.0
_LX, _LY, _LZ = 1.0e-3, 0.5e-3, 0.25e-3
_A, _P, _R_B, _V = 0.30, 24.0, 30e-6, 0.15
_X_START = _LX / 4
_T_TOTAL = 1.0e-3
_X_END = _X_START + _V * _T_TOTAL

# Discretization for the analytical match. z resolution dominates the error
# (the near-surface gradient is steepest), so nz is the largest count. This
# config gives L2_rel < 0.2 % (verified); the run is deterministic.
_NX, _NY, _NZ, _N_STEPS = 128, 64, 160, 100

_MAT = {
    "name": "316L", "rho": _RHO, "k": _K, "Cp": _CP,
    "L_f": 2.677e5, "T_solidus": 1674.15, "T_liquidus": 1697.15, "T0": _T0,
    "DeltaH_LV": 7.416e6, "R_v": 150.774, "Pa": 101325.0, "T_boil": 3090.0,
}


def _config(power=_P, mesh=(_NX, _NY, _NZ), n_steps=_N_STEPS, backend="cpu_linear"):
    return {
        "simulation": {"backend": backend,
                       "duration": _T_TOTAL, "dt": _T_TOTAL / n_steps},
        "domain": {"size": [_LX, _LY, _LZ], "mesh": list(mesh)},
        "material": copy.deepcopy(_MAT),
        "laser": {"radius": _R_B, "absorptivity": _A, "power_nominal": power},
        "io": {},
    }


def _context(cfg, laser):
    ctx = SimulationContext.from_dict(copy.deepcopy(cfg))
    ctx.laser_path = laser
    return ctx


def _run_linear_field(ctx):
    """Step the linear solver to t_end; return the node-centred field [z, y, x]."""
    from fast_heat_solv.physics.spectral_helpers import reconstruct_temperature_DCT
    from fast_heat_solv.solvers import build_solver

    solver = build_solver(ctx)
    solver.initialize(ctx)
    dt = ctx.num.dt
    for step in range(ctx.num.n_steps):
        solver.step(step * dt, dt)
    # reconstruct_temperature_DCT returns [x, y, z]; transpose to [z, y, x] to
    # match the Eagar-Tsai reference and compute_L2's weight ordering.
    T_xyz = reconstruct_temperature_DCT(solver.state.a, solver.state)
    return np.ascontiguousarray(T_xyz.transpose(2, 1, 0))


# ---------------------------------------------------------------------------
# linear — closed-form Eagar-Tsai (the correctness anchor)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_linear_matches_eagar_tsai(constant_velocity_laser):
    """Linear solver vs the method-of-images Eagar-Tsai field."""
    from tests.eagar_tsai import eagar_tsai_field

    x = np.linspace(0.0, _LX, _NX + 1)
    y = np.linspace(0.0, _LY, _NY + 1)
    z = np.linspace(0.0, _LZ, _NZ + 1)

    laser = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    T = _run_linear_field(_context(_config(), laser))

    # Time-centering: the solver holds the source piecewise-constant over each
    # ETD1 step (sampled at the step start), so the field at t_total corresponds
    # to the source at the *midpoint* of the last interval, x_end − ½·v·dt — not
    # at x_end. 
    # Verified: the L2 error is parabolic in the shift with a sharp minimum 
    # at exactly ½·v·dt.
    # The ETD1 field equals the field of a source whose entire trajectory 
    # is shifted back by dt/2 if we change to ETD2 or RK4, reconsider this 
    # time-centering correction then.
    dt = _T_TOTAL / _N_STEPS
    T_ref = eagar_tsai_field(
        rho=_RHO, k=_K, Cp=_CP, T0=_T0, A=_A, P=_P, r_b=_R_B,
        Lx=_LX, Ly=_LY, Lz=_LZ, nx=_NX, ny=_NY, nz=_NZ,
        x_end=_X_END - 0.5 * _V * dt, y_laser=_LY / 2, v=_V, t_total=_T_TOTAL,
        images=True,
    )

    assert not np.isnan(T).any() and not np.isinf(T).any()
    assert T.max() > _T0
    assert T.min() >= _T0 - 1.0  # linear problem cannot cool below ambient

    res = compute_L2(T_ref, T, x, y, z)
    peak = float(T.max() - _T0)
    assert res["L2_rel"] < 0.002, f"L2_rel={res['L2_rel']:.5f}"
    assert res["Linf"] / peak < 0.10, f"Linf/peak={res['Linf'] / peak:.4f}"

    pk_solver = np.unravel_index(int(np.argmax(T)), T.shape)
    pk_ref = np.unravel_index(int(np.argmax(T_ref)), T_ref.shape)
    assert max(abs(a - b) for a, b in zip(pk_solver, pk_ref)) <= 2


@pytest.mark.integration
def test_linear_zero_power_stays_ambient(constant_velocity_laser):
    """Power = 0 ⇒ field stays at T0 (only the mean mode is populated)."""
    laser = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, 0.0)
    ctx = _context(_config(power=0.0, mesh=(48, 24, 32), n_steps=20), laser)
    T = _run_linear_field(ctx)
    np.testing.assert_allclose(T, _T0, rtol=1e-4)


@pytest.mark.integration
@pytest.mark.slow
def test_linear_superposition(constant_velocity_laser):
    """The linear solver is linear: 2P gives exactly 2×(T−T0) of P (float32).

    Pure algebraic property — exact regardless of mesh, so a coarse grid keeps
    it cheap.
    """
    mesh, ns = (64, 32, 64), 40
    laser_p = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    laser_2p = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, 2 * _P)
    T_p = _run_linear_field(_context(_config(power=_P, mesh=mesh, n_steps=ns), laser_p))
    T_2p = _run_linear_field(_context(_config(power=2 * _P, mesh=mesh, n_steps=ns), laser_2p))
    np.testing.assert_allclose(T_2p - _T0, 2.0 * (T_p - _T0), rtol=1e-4, atol=1e-2)


# ---------------------------------------------------------------------------
# non-linear — sanity bounds (no analytical reference)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_nonlinear_sanity_bounds(constant_velocity_laser, assert_final_step_converged):
    """Full non-linear run stays physically / numerically well-behaved.

    Latent heat + evaporation are active; the evaporation flux is the
    regularizer that caps the surface temperature. No closed form — loose
    bounds only. The linear peak for this case is above T_liquidus = 1697 K), 
    so evaporation engages.
    """
    from fast_heat_solv.physics.spectral_helpers import reconstruct_temperature_DCT
    from fast_heat_solv.solvers import build_solver

    # Resolved enough that the source no longer rings below ambient: at this
    # mesh the surface undershoot is ~0.4 K (vs ~10 K on a coarse 64³ grid).
    cfg = _config(mesh=(112, 56, 112), n_steps=40, backend="cpu")
    laser = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    ctx = _context(cfg, laser)

    solver = build_solver(ctx)
    solver.initialize(ctx)
    dt = ctx.num.dt
    surf_max, picard = [], []
    for step in range(ctx.num.n_steps):
        _, metrics = solver.step(step * dt, dt)
        surf_max.append(float(metrics["T_surface_max"]))
        picard.append(int(metrics["n_evap_iter"]))

    T = reconstruct_temperature_DCT(solver.state.a, solver.state)
    assert not np.isnan(T).any() and not np.isinf(T).any()
    # No sub-ambient ringing on the resolved grid (observed undershoot ≈ 0.4 K).
    assert T.min() >= _T0 - 1.0
    # Surface capped well below a blow-up — evaporation must regularize it.
    # 1.5×T_boil is the generous physical cap from the plan (§2.2).
    assert max(surf_max) < 1.5 * _MAT["T_boil"]
    # Picard loop respects the cap, and the final step converges below it.
    assert all(n <= solver.max_picard_iter for n in picard)
    assert_final_step_converged(picard, solver.max_picard_iter)
