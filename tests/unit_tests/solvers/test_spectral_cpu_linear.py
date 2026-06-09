"""Tests for SpectralSolverCPULinear (solvers/spectral_cpu_linear.py)."""

import numpy as np
import pytest

from fast_heat_solv.solvers.spectral_cpu_linear import SpectralSolverCPULinear


def test_initialize_requires_context():
    # Same context guard as SpectralSolver, but a separate code path.
    with pytest.raises(RuntimeError):
        SpectralSolverCPULinear().initialize()


@pytest.mark.slow
def test_step_reports_no_evaporation(tiny_context):
    # The linear solver applies only the laser source — no Picard / evaporation,
    # so n_evap_iter is always 0 (its distinguishing contract vs SpectralSolver).
    s = SpectralSolverCPULinear(tiny_context)
    s.initialize()
    _, metrics = s.step(0.0, tiny_context.num.dt)
    assert metrics["n_evap_iter"] == 0
    assert np.isfinite(float(metrics["T_surface_max"]))


@pytest.mark.slow
def test_zero_power_stays_ambient(make_tiny_context):
    # Zero laser power -> no heating -> stays at T0.
    from fast_heat_solv.physics import spectral_cpu_kernels as k

    ctx = make_tiny_context(power=0.0)
    s = SpectralSolverCPULinear(ctx)
    s.initialize()
    state, metrics = s.step(0.0, ctx.num.dt)
    assert float(metrics["P_laser"]) == 0.0
    surface = k.reconstruct_surface_temperature(state.a, state)
    np.testing.assert_allclose(surface, ctx.mat.T0, rtol=1e-4)
