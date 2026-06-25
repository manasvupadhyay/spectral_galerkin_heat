"""Unit tests for the configurable solver precision (``simulation.dtype``).

Fast tests (no numba/solver imports): they exercise the config parsing in
:meth:`SimulationContext.from_dict` and the ``_resolve_dtype`` helper.
"""

import copy

import numpy as np
import pytest

from fast_heat_solv.core.parameters import SimulationContext, _resolve_dtype


_BASE_CFG = {
    "simulation": {"method": "spectral", "backend": "cpu",
                   "duration": 6e-5, "dt": 6e-6},
    "domain": {"size": [4e-4, 4e-4, 1e-4], "mesh": [8, 8, 4]},
    "material": {"name": "316L", "rho": 7850.0, "k": 15.0, "Cp": 500.0, "T0": 293.0},
    "laser": {"radius": 60e-6, "absorptivity": 0.3, "power_nominal": 200.0},
    "io": {},
}


# ---------------------------------------------------------------------------
# _resolve_dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("float32", np.float32),
    ("float64", np.float64),
    ("FLOAT64", np.float64),   # case-insensitive
])
def test_resolve_dtype_valid(name, expected):
    assert _resolve_dtype(name) is expected


def test_resolve_dtype_invalid_raises():
    with pytest.raises(ValueError, match="Unknown dtype"):
        _resolve_dtype("float16")


# ---------------------------------------------------------------------------
# SimulationContext.from_dict
# ---------------------------------------------------------------------------

def test_dtype_defaults_to_float32_when_absent():
    ctx = SimulationContext.from_dict(copy.deepcopy(_BASE_CFG))
    assert ctx.num.dtype is np.float32


@pytest.mark.parametrize("name, expected", [
    ("float32", np.float32),
    ("float64", np.float64),
])
def test_dtype_parsed_from_config(name, expected):
    cfg = copy.deepcopy(_BASE_CFG)
    cfg["simulation"]["dtype"] = name
    ctx = SimulationContext.from_dict(cfg)
    assert ctx.num.dtype is expected
    # Material params are cast at the configured precision.
    assert ctx.mat.T0.dtype == np.dtype(expected)


def test_unknown_dtype_in_config_raises():
    cfg = copy.deepcopy(_BASE_CFG)
    cfg["simulation"]["dtype"] = "float128"
    with pytest.raises(ValueError, match="Unknown dtype"):
        SimulationContext.from_dict(cfg)
