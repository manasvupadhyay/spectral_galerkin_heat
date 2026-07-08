"""Tests for SpectralSolver lifecycle (solvers/spectral.py)."""

import math

import numpy as np
import pytest

from fast_heat_solv.backends import NumpyBackend
from fast_heat_solv.solvers.spectral import SpectralSolver


def test_defaults():
    # Constructor defaults for the Picard fixed-point loop; these hold until
    # initialize() applies any overrides from the simulation config.
    s = SpectralSolver(NumpyBackend())
    assert float(s.mixing_omega) == pytest.approx(0.1)
    assert float(s.convergence_tol) == pytest.approx(1e-4)
    assert s.max_picard_iter == 30
    assert s.track_picard_history is False
    assert s.state is None


def test_initialize_applies_picard_config_overrides(tiny_config):
    # initialize() overrides the Picard defaults from the simulation block
    # (max_picard_iter / picard_tol / picard_omega); absent keys keep the defaults.
    from fast_heat_solv.core.parameters import SimulationContext

    tiny_config["simulation"].update(
        max_picard_iter=55, picard_tol=1e-6, picard_omega=0.25)
    ctx = SimulationContext.from_dict(tiny_config)
    s = SpectralSolver(NumpyBackend(), ctx)
    s.initialize()
    assert s.max_picard_iter == 55
    assert float(s.convergence_tol) == pytest.approx(1e-6)
    assert float(s.mixing_omega) == pytest.approx(0.25)


def test_initialize_keeps_picard_defaults_when_unset(tiny_config):
    # No Picard keys in the config -> the constructor defaults survive initialize().
    from fast_heat_solv.core.parameters import SimulationContext

    ctx = SimulationContext.from_dict(tiny_config)
    s = SpectralSolver(NumpyBackend(), ctx)
    s.initialize()
    assert s.max_picard_iter == 30
    assert float(s.convergence_tol) == pytest.approx(1e-4)
    assert float(s.mixing_omega) == pytest.approx(0.1)


def test_initialize_requires_context():
    # Stepping needs a context; initialize() without one fails fast, not later.
    with pytest.raises(RuntimeError):
        SpectralSolver(NumpyBackend()).initialize()


def test_set_state_before_init_raises():
    # set_state needs the grid/buffers built by initialize() first.
    with pytest.raises(RuntimeError):
        SpectralSolver(NumpyBackend()).set_state(np.zeros((4, 4, 4), dtype=np.float32))


def test_initialize_sets_uniform_ambient_field(tiny_context):
    # IC lives entirely in mode (0,0,0); the reconstructed surface must be T0
    # everywhere. Guards the sqrt(V) scaling of the mean mode.
    from fast_heat_solv.physics import spectral_cpu_kernels as k

    s = SpectralSolver(NumpyBackend(), tiny_context)
    state = s.initialize()
    geom, T0 = tiny_context.geom, tiny_context.mat.T0
    expected = T0 * math.sqrt(geom.size.x * geom.size.y * geom.size.z)
    assert float(state.a[0, 0, 0]) == pytest.approx(expected, rel=1e-5)
    assert np.count_nonzero(state.a) == 1  # only the mean mode is set
    surface = k.reconstruct_surface_temperature(state.a, state)
    np.testing.assert_allclose(surface, T0, rtol=1e-4)


@pytest.mark.slow
def test_set_state_roundtrip(tiny_context):
    # set_state(uniform T) -> reconstructed surface == T (DCT scaling round-trip).
    from fast_heat_solv.physics import spectral_cpu_kernels as k

    s = SpectralSolver(NumpyBackend(), tiny_context)
    s.initialize()
    num = tiny_context.num
    field = np.full((num.nz, num.ny, num.nx), 500.0, dtype=np.float32)
    s.set_state(field)
    surface = k.reconstruct_surface_temperature(s.state.a, s.state)
    np.testing.assert_allclose(surface, 500.0, rtol=1e-4)


@pytest.mark.slow
def test_step_returns_state_and_metrics(tiny_context):
    # One step returns (state, metrics) with the documented metric keys, all finite.
    s = SpectralSolver(NumpyBackend(), tiny_context)
    s.initialize()
    state, metrics = s.step(0.0, tiny_context.num.dt)
    assert state is s.state
    assert set(metrics) == {"T_surface_max", "P_laser", "n_evap_iter"}
    assert all(np.isfinite(float(v)) for v in metrics.values())
    # P_laser is the discrete integral of the absorbed Gaussian ≈ A·P.
    a, p = tiny_context.laser.absorptivity, 200.0
    assert float(metrics["P_laser"]) == pytest.approx(a * p, rel=0.1)
    assert metrics["n_evap_iter"] >= 1


@pytest.mark.slow
def test_step_laser_off_stays_ambient(make_tiny_context):
    # Laser off -> zero source -> the field stays at ambient T0.
    from fast_heat_solv.physics import spectral_cpu_kernels as k

    ctx = make_tiny_context(is_on=False)
    s = SpectralSolver(NumpyBackend(), ctx)
    s.initialize()
    state, metrics = s.step(0.0, ctx.num.dt)
    assert float(metrics["P_laser"]) == 0.0
    surface = k.reconstruct_surface_temperature(state.a, state)
    np.testing.assert_allclose(surface, ctx.mat.T0, rtol=1e-4)


@pytest.mark.slow
def test_track_picard_history(tiny_context):
    # Opt-in history records one dict per Picard iteration; length == n_evap_iter.
    s = SpectralSolver(NumpyBackend(), tiny_context)
    s.track_picard_history = True
    s.initialize()
    _, metrics = s.step(0.0, tiny_context.num.dt)
    assert len(s.picard_history) == metrics["n_evap_iter"]
    assert s.picard_history  # non-empty
    assert set(s.picard_history[0]) == {"iter", "rms_diff", "rho", "true_rel_err"}
