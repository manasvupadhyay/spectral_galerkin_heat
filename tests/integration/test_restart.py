"""Restart / state-injection round-trip (§2.6).

Run a few steps, snapshot the reconstructed field, build a fresh solver,
``set_state`` the snapshot back, and assert the reconstruction is unchanged
within float32.  This pins the DCT/IDCT scaling constants that ``set_state``
(forward ``DCT_II``·√(dx·dy·dz)) and ``reconstruct_temperature_DCT`` share — a
mismatch there would silently corrupt every restart.
"""

import copy

import numpy as np
import pytest

from fast_heat_solv.core.parameters import SimulationContext

_CONFIG = {
    "simulation": {"method": "spectral", "backend": "cpu", "duration": 3e-5, "dt": 6e-6},
    "domain": {"size": [4e-4, 4e-4, 1e-4], "mesh": [24, 24, 12]},
    "material": {
        "name": "316L", "rho": 7850.0, "k": 15.0, "Cp": 500.0, "L_f": 267700.0,
        "T_solidus": 1700.0, "T_liquidus": 1800.0, "T0": 293.0,
        "DeltaH_LV": 7.41e6, "R_v": 150.774, "Pa": 101325.0, "T_boil": 3090.0,
    },
    "laser": {"radius": 60.0e-6, "absorptivity": 0.3, "power_nominal": 200.0},
    "io": {},
}


@pytest.mark.integration
@pytest.mark.slow
def test_set_state_roundtrip(fixed_laser):
    """modes → cell field → set_state → modes must reconstruct the same field."""
    from fast_heat_solv.physics.spectral_helpers import reconstruct_temperature_DCT
    from fast_heat_solv.solvers import build_solver

    ctx = SimulationContext.from_dict(copy.deepcopy(_CONFIG))
    ctx.laser_path = fixed_laser(ctx.geom.size.x / 2, ctx.geom.size.y / 2, 200.0)

    # Run a few steps to get a non-trivial (non-uniform) modal state.
    solver = build_solver(ctx)
    solver.initialize(ctx)
    dt = ctx.num.dt
    for step in range(ctx.num.n_steps):
        solver.step(step * dt, dt)

    snapshot = reconstruct_temperature_DCT(solver.state.a, solver.state)

    # Recover the cell-centred field set_state consumes (exact inverse of its
    # forward DCT_II·scale), then inject it into a fresh solver.
    import math
    k = solver.backend.kernels
    geom = ctx.geom
    scale = math.sqrt(float(geom.d.x * geom.d.y * geom.d.z))
    cell_field = k.IDCT_II(solver.state.a / np.float32(scale))

    fresh = build_solver(ctx)
    fresh.initialize(ctx)
    fresh.set_state(cell_field)
    restored = reconstruct_temperature_DCT(fresh.state.a, fresh.state)

    # float32 DCT/IDCT round-trip — rtol covers reduction-order round-off only.
    np.testing.assert_allclose(restored, snapshot, rtol=1e-4, atol=1e-2)
