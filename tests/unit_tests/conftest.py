"""Shared fixtures for the unit suite (small contexts + a stationary laser)."""

import copy

import pytest

from spectral_galerkin_heat.core.laser import LaserPath, LaserState
from spectral_galerkin_heat.core.parameters import SimulationContext


class FixedLaser(LaserPath):
    """Stationary laser at a fixed (x, y); power/on-state are constant."""

    def __init__(self, x, y, power, is_on=True):
        self._x, self._y, self._power, self._on = x, y, power, is_on

    def get_state(self, time, dt):
        return LaserState(x=self._x, y=self._y, power=self._power, is_on=self._on)


# 400 µm domain on a 32×32×16 mesh resolves the 60 µm spot (~25 cells across the
# diameter), so the Gaussian's discrete integral matches A·P closely. Full 316L
# values keep R_v / T_boil non-zero (the evaporation kernel divides by them).
_TINY_CONFIG = {
    "simulation": {"backend": "cpu", "duration": 6e-6, "dt": 6e-6},
    "domain": {"size": [4e-4, 4e-4, 1e-4], "mesh": [32, 32, 16]},
    "material": {
        "name": "316L", "rho": 7850.0, "k": 15.0, "Cp": 500.0, "L_f": 267700.0,
        "T_solidus": 1700.0, "T_liquidus": 1800.0, "T0": 293.0,
        "DeltaH_LV": 7.41e6, "R_v": 150.774, "Pa": 101325.0, "T_boil": 3090.0,
    },
    "laser": {"radius": 60.0e-6, "absorptivity": 0.3, "power_nominal": 200.0},
    "io": {"at_end": ["full_volume"]},
}


@pytest.fixture
def tiny_config():
    return copy.deepcopy(_TINY_CONFIG)


@pytest.fixture
def make_tiny_context(tiny_config):
    """Factory: build a context with a centred FixedLaser (power / on-state tunable)."""
    def _make(power=None, is_on=True):
        ctx = SimulationContext.from_dict(copy.deepcopy(tiny_config))
        p = tiny_config["laser"]["power_nominal"] if power is None else power
        ctx.laser_path = FixedLaser(ctx.geom.size.x / 2, ctx.geom.size.y / 2, p, is_on)
        return ctx
    return _make


@pytest.fixture
def tiny_context(make_tiny_context):
    return make_tiny_context()
