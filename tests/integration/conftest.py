"""Shared fixtures and helpers for the integration suite.

Laser implementations and the Picard-convergence helpers are promoted here so
the pipeline, validation, and restart tests share one copy.  Everything is
exposed as a fixture (the pytest-idiomatic way to share across files) rather
than imported from ``conftest`` directly.
"""

import contextlib
import logging

import numpy as np
import pytest

from fast_heat_solv.core.laser import LaserPath, LaserState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Laser implementations
# ---------------------------------------------------------------------------

class FixedLaser(LaserPath):
    """Stationary laser at a fixed (x, y) with constant power."""

    def __init__(self, x: float, y: float, power: float, is_on: bool = True) -> None:
        self._x, self._y, self._power, self._on = x, y, power, is_on

    def get_state(self, time: float, dt: float) -> LaserState:
        return LaserState(x=self._x, y=self._y, power=self._power, is_on=self._on)


class ConstantVelocityLaser(LaserPath):
    """Laser moving in a straight line at constant velocity from (x0, y0).

    Position at time ``t`` is ``(x0 + vx·t, y0 + vy·t)``; velocity is reported
    on the state so the solver's Doppler / shift bookkeeping sees it.  Used by
    the Eagar-Tsai analytical check, which assumes exactly this motion.
    """

    def __init__(self, x0, y0, vx, vy, power, is_on=True):
        self._x0, self._y0 = x0, y0
        self._vx, self._vy = vx, vy
        self._power, self._on = power, is_on

    def get_state(self, time: float, dt: float) -> LaserState:
        return LaserState(
            x=self._x0 + self._vx * time,
            y=self._y0 + self._vy * time,
            power=self._power,
            is_on=self._on,
            v=(self._vx, self._vy),
        )


@pytest.fixture
def fixed_laser():
    """The stationary-laser class (instantiate with ``(x, y, power)``)."""
    return FixedLaser


@pytest.fixture
def constant_velocity_laser():
    """The constant-velocity-laser class (``(x0, y0, vx, vy, power)``)."""
    return ConstantVelocityLaser


# ---------------------------------------------------------------------------
# Physics-sanity / convergence helpers
# ---------------------------------------------------------------------------

def _assert_field_sane(T: np.ndarray, T0: float) -> None:
    assert not np.isnan(T).any(), "NaN values detected in temperature field"
    assert not np.isinf(T).any(), "Inf values detected in temperature field"
    assert T.max() > T0, f"No measurable heating: T_max={T.max():.1f} ≤ T0={T0}"


def _assert_final_step_converged(picard_counts: list[int], max_iter: int) -> None:
    """The final step must converge strictly below the Picard cap.

    A step that ran the full ``max_picard_iter`` iterations bailed at the cap
    without meeting ``convergence_tol``.  Early steps may legitimately hit it
    while the field changes fastest, but by the end the solution should settle.
    """
    assert picard_counts, "No Picard iteration counts were recorded"
    final = picard_counts[-1]
    assert final < max_iter, (
        f"Solver did not converge at the end: final step ran the full "
        f"{max_iter} Picard iterations (cap). Per-step counts: {picard_counts}"
    )


@contextlib.contextmanager
def _log_picard_iterations(runner):
    """Collect per-step ``n_evap_iter`` by wrapping the solver's ``step``."""
    counts: list[int] = []
    solver = runner.heat_solver
    real_step = solver.step

    def step_with_count(t, dt):
        state, metrics = real_step(t, dt)
        n = metrics.get("n_evap_iter")
        if n is not None:
            counts.append(int(n))
        return state, metrics

    solver.step = step_with_count
    try:
        yield counts
    finally:
        solver.step = real_step
        if counts:
            logger.info(
                "Picard iterations per step (%d steps): %s | min=%d max=%d mean=%.1f",
                len(counts), counts, min(counts), max(counts),
                sum(counts) / len(counts),
            )


@pytest.fixture
def assert_field_sane():
    return _assert_field_sane


@pytest.fixture
def assert_final_step_converged():
    return _assert_final_step_converged


@pytest.fixture
def log_picard_iterations():
    return _log_picard_iterations
