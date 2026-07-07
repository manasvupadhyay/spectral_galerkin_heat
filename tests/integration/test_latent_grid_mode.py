"""Grid-mode latent heat (no fine sub-box) equivalence and behaviour.

When ``fine_mesh`` is absent the latent-heat source is evaluated on the coarse
main grid (lower memory, faster for small runs) instead of the laser-following
fine box. This must be numerically equivalent to the box path at the same
resolution — i.e. a box with ``refinement: 1`` covering the whole domain — since
they compute the same source and differ only in the projection (full-volume DCT
vs the box contraction).

Run with::

    pytest -m integration tests/integration/test_latent_grid_mode.py
"""

import copy

import numpy as np
import pytest

from fast_heat_solv.core.parameters import SimulationContext

_T_S, _T_L, _T0 = 1674.15, 1697.15, 293.0
_LX, _LY, _LZ = 0.6e-3, 0.4e-3, 0.15e-3
_P, _V = 120.0, 0.4


def _cfg(fine=None):
    cfg = {
        "simulation": {"backend": "cpu", "duration": 6.0e-5, "dt": 5.0e-6},
        "domain": {"size": [_LX, _LY, _LZ], "mesh": [40, 24, 60]},
        "material": {"name": "316L", "k": 13.851, "rho": 7957.5, "Cp": 497.89,
                     "T_solidus": _T_S, "T_liquidus": _T_L, "T0": _T0,
                     "L_f": 2.677e5, "h_conv": 3000.0, "DeltaH_LV": 7.416e6,
                     "R_v": 150.774, "Pa": 101325.0, "T_boil": 3090.0},
        "laser": {"radius": 30e-6, "absorptivity": 0.30, "power_nominal": _P},
        "io": {},
    }
    if fine is not None:
        cfg["fine_mesh"] = fine
    return cfg


def _run(cfg, laser, n=12):
    from fast_heat_solv.physics.spectral_helpers import reconstruct_temperature_DCT
    from fast_heat_solv.solvers import build_solver
    ctx = SimulationContext.from_dict(copy.deepcopy(cfg))
    ctx.laser_path = laser
    solver = build_solver(ctx)
    solver.initialize(ctx)
    dt = ctx.num.dt
    for step in range(n):
        solver.step(step * dt, dt)
    return solver, reconstruct_temperature_DCT(solver.state.a, solver.state)


@pytest.mark.integration
def test_grid_latent_matches_box_refinement1(constant_velocity_laser):
    """Grid mode == box mode at refinement 1 / full domain (same resolution).

    Uses a **stationary** laser so the box does not translate its history: the
    two paths then compute the identical latent source and differ only in the
    projection (full-volume DCT vs the box einsum), i.e. float32 noise. (Under a
    *moving* laser the box additionally interpolates its shifted history, an
    error grid mode does not incur — so exact equivalence only holds at v = 0.)
    """
    laser1 = constant_velocity_laser(_LX / 2, _LY / 2, 0.0, 0.0, _P)
    solver_g, T_grid = _run(_cfg(fine=None), laser1)

    laser2 = constant_velocity_laser(_LX / 2, _LY / 2, 0.0, 0.0, _P)
    solver_b, T_box = _run(
        _cfg(fine={"refinement": 1, "box_size": [_LX, _LY, _LZ]}), laser2)

    # Mode wiring is what we expect.
    assert solver_g._grid_latent is True and solver_g.state.fine_mesh is None
    assert solver_b._grid_latent is False and solver_b.state.fine_mesh is not None

    # Both actually melted (latent heat is exercised).
    assert T_grid.max() > _T_L
    # Equivalent up to float32 DCT-vs-einsum ordering.
    np.testing.assert_allclose(T_grid, T_box, rtol=1e-3, atol=0.2)


@pytest.mark.integration
def test_grid_mode_is_default_without_fine_mesh(constant_velocity_laser):
    """Omitting ``fine_mesh`` selects grid mode and allocates a full-volume buffer."""
    laser = constant_velocity_laser(_LX / 4, _LY / 2, _V, 0.0, _P)
    ctx = SimulationContext.from_dict(_cfg(fine=None))
    ctx.laser_path = laser
    assert ctx.fine is None
    from fast_heat_solv.solvers import build_solver
    solver = build_solver(ctx)
    solver.initialize(ctx)
    assert solver.state.fine_mesh is None
    assert solver.state.buffers.Q_latent_buffer.shape == (
        ctx.num.nz, ctx.num.ny, ctx.num.nx)
