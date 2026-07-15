"""Tests for config parsing / derived params (core/parameters.py)."""

import numpy as np
import pytest

from fast_heat_solv.core.parameters import (
    GeomParams,
    MaterialParams,
    NumParams,
    SimulationContext,
    _get_value,
)
from fast_heat_solv.core.vector import Vec3

# `cfg` here is a parsed-YAML config dict (the input to SimulationContext.from_dict):
# See parameters.py for the full key list. `_cfg()` builds a minimal valid one.


def _cfg(**sections):
    # Minimal valid config; sections override the defaults below.
    base = {
        "simulation": {"dt": 1e-5, "duration": 1e-3},
        "domain": {"size": [5e-3, 2.5e-3, 1.25e-3], "mesh": [64, 32, 16]},
        "material": {"rho": 7900.0, "k": 15.0, "Cp": 500.0},
        "laser": {"radius": 50e-6, "absorptivity": 0.4, "power_nominal": 200.0},
    }
    for key, val in sections.items():
        base[key] = {**base.get(key, {}), **val} if val is not None else val
    return base


def test_from_dict_minimal_defaults():
    ctx = SimulationContext.from_dict(_cfg())
    assert isinstance(ctx.num, NumParams)
    assert isinstance(ctx.geom, GeomParams)
    assert ctx.backend == "cpu"      # default
    assert ctx.mat.name == "Material"  # default applied
    assert ctx.laser_path is None      # no gcode block


def test_backend_lowercased():
    ctx = SimulationContext.from_dict(_cfg(simulation={"backend": "GPU"}))
    assert ctx.backend == "gpu"


def test_dt_correction_makes_steps_tile_t_end():
    # dt_nominal need not divide t_end; the solver rounds n_steps and rescales dt
    # so n_steps * dt == t_end. Use a non-dividing nominal to exercise the rescale.
    ctx = SimulationContext.from_dict(_cfg(simulation={"dt": 3e-5, "duration": 1e-3}))
    num = ctx.num
    assert num.n_steps == round(num.t_end / num.dt_nominal)
    assert num.dt != num.dt_nominal               # actually rescaled
    assert num.n_steps * num.dt == pytest.approx(num.t_end)
    assert num.dt_nominal == pytest.approx(np.float32(3e-5))  # nominal preserved


def test_get_value_unwraps_unit_dict_and_passes_scalars():
    # _get_value lets config fields be either a number or
    # a {value, unit}. Unit-tested directly so the two branches work
    assert _get_value({"value": 200.0, "unit": "W"}) == 200.0  
    assert _get_value(7.0) == 7.0                              


def test_get_value_converts_units():
    # Test conversion from a non-SI unit to the target SI unit.
    val_in_mm = {"value": 500.0, "unit": "mm"}
    assert _get_value(val_in_mm, target_unit="m") == pytest.approx(0.5)

    # Test when units are already correct (no-op conversion).
    val_in_m = {"value": 0.5, "unit": "m"}
    assert _get_value(val_in_m, target_unit="m") == pytest.approx(0.5)

    # Test incompatible units should raise.
    # pint.errors.DimensionalityError is the specific exception.
    try:
        import pint
        ErrorType = pint.errors.DimensionalityError
    except (ImportError, AttributeError):
        ErrorType = Exception  # Fallback for environments without pint
    with pytest.raises(ErrorType):
        _get_value(val_in_mm, target_unit="kg")

@pytest.mark.parametrize("bad_input", [{}, {"unit": "W"}, None, "a string"])
def test_get_value_invalid_input_raises(bad_input):
    # _get_value should raise if the input is not a number or a valid value dict.
    with pytest.raises((KeyError, TypeError)):
        _get_value(bad_input)


def test_unit_dict_values_unwrapped_in_parse():
    # Same {value, unit}, but test end-to-end through from_dict: a
    # laser power written as {value: 250, unit: "W"} must reach LaserParams.power
    # as the plain number 250
    ctx = SimulationContext.from_dict(_cfg(laser={"power_nominal": {"value": 250.0, "unit": "W"}}))
    assert ctx.laser.power == pytest.approx(250.0)


def test_unit_conversion_in_parse():
    # This test demonstrates the goal of unit-aware parsing.
    # It provides density in g/cm^3 and expects the parser to convert it to kg/m^3.
    # 1 g/cm^3 = 1000 kg/m^3.
    cfg = _cfg(material={"rho": {"value": 7.9575, "unit": "g/cm**3"}})
    ctx = SimulationContext.from_dict(cfg)
    assert ctx.mat.rho == pytest.approx(7957.5)


def test_material_diff_is_k_over_rho_cp():
    # `diff` is a derived @property (thermal diffusivity k/(rho*Cp))
    m = MaterialParams(rho=2.0, k=10.0, Cp=5.0)
    assert m.diff == pytest.approx(10.0 / (2.0 * 5.0))


def test_geom_d_is_size_over_n():
    # GeomParams.__post_init__ derives the grid spacing d = size / n. Inputs are
    # asymmetric (different per axis) so a swapped/transposed axis can't pass by
    # coincidence.
    g = GeomParams(size=Vec3(5.0, 2.0, 1.0), n=Vec3(10, 8, 4))
    assert g.d == Vec3(0.5, 0.25, 0.25)


def test_from_dict_preserves_geom_axis_order():
    # Guards against an x/y/z mix-up while parsing domain.size / domain.mesh into
    # Vec3s. _cfg() uses distinct mesh counts [64,32,16] and distinct extents, so
    # any reordering during parsing would change these and fail the assertions.
    ctx = SimulationContext.from_dict(_cfg())
    assert ctx.geom.n == Vec3(64, 32, 16)
    assert ctx.geom.size == Vec3(5e-3, 2.5e-3, 1.25e-3)


# ---------------------------------------------------------------------------
# Temperature-dependent properties (material.model)
# ---------------------------------------------------------------------------

# Typical 316L branch expressions, matching the FE reference.
_K_BR = {"solid": "9.248 + 0.01571 * T", "liquid": "12.41 + 0.003279 * T"}
_RHO_BR = {"solid": "8084.2 - 0.42086 * T - 3.8942e-5 * T**2",
           "liquid": "7432.7 + 0.039338 * T - 1.8007e-4 * T**2"}
_CP_BR = {"solid": "458.98 + 0.1328 * T", "liquid": "769.86"}


def test_scalar_material_has_no_model():
    # Regression: a fully scalar material leaves model=None and scalars intact.
    ctx = SimulationContext.from_dict(_cfg())
    assert ctx.mat.model is None
    assert float(ctx.mat.k) == pytest.approx(15.0)
    assert float(ctx.mat.rho) == pytest.approx(7900.0)
    assert float(ctx.mat.Cp) == pytest.approx(500.0)


def test_branch_material_uses_in_block_reference_scalars():
    # A T-dependent material bakes each property's in-block `reference:` verbatim
    # into the scalar reference field (no auto-evaluation at any temperature).
    ctx = SimulationContext.from_dict(_cfg(material={
        "k": {**_K_BR, "reference": 22.5, "unit": "W/(m.K)"},
        "rho": {**_RHO_BR, "reference": 7600.0, "unit": "kg/m^3"},
        "Cp": {**_CP_BR, "reference": 620.0, "unit": "J/(kg.K)"},
        "T_solidus": 1674.15, "T_liquidus": 1697.15, "T0": 293.0, "T_boil": 3090.0,
    }))
    assert ctx.mat.model is not None
    assert float(ctx.mat.k) == pytest.approx(22.5)
    assert float(ctx.mat.rho) == pytest.approx(7600.0)
    assert float(ctx.mat.Cp) == pytest.approx(620.0)


def test_branch_material_missing_reference_raises_with_recommendation():
    # Omitting a property's `reference:` is a hard error whose message names the
    # missing properties, recommends the mid-range values, and flags convergence.
    with pytest.raises(ValueError) as excinfo:
        SimulationContext.from_dict(_cfg(material={
            "k": _K_BR, "rho": _RHO_BR, "Cp": _CP_BR,
            "T_solidus": 1674.15, "T_liquidus": 1697.15, "T0": 293.0,
            "T_boil": 3090.0,
        }))
    msg = str(excinfo.value)
    assert "reference" in msg
    assert "k.reference" in msg and "rho.reference" in msg and "Cp.reference" in msg
    assert "convergence" in msg
    assert "Recommended" in msg


def test_mixed_branch_and_scalar_material():
    # k T-dependent (needs an in-block reference), rho/Cp scalar (each is its own
    # reference) → still a model; the scalars pass straight through.
    ctx = SimulationContext.from_dict(_cfg(material={
        "k": {**_K_BR, "reference": 22.5}, "rho": 7900.0, "Cp": 500.0,
        "T_solidus": 1674.15, "T_liquidus": 1697.15, "T0": 293.0, "T_boil": 3090.0,
    }))
    assert ctx.mat.model is not None
    assert float(ctx.mat.k) == pytest.approx(22.5)
    assert float(ctx.mat.rho) == pytest.approx(7900.0)
    assert float(ctx.mat.Cp) == pytest.approx(500.0)


@pytest.mark.parametrize(
    "drop",
    [("domain", "size"), ("domain", "mesh"), ("material", "rho"),
     ("laser", "radius"), ("laser", "power_nominal"), ("simulation", "dt")],
)
def test_missing_required_key_raises(drop):
    # Documents the hard-required config keys.
    #Each builds a minimal valid cfg, deletes exactly one 
    # required key, and asserts from_dict rejects it with KeyError
    section, key = drop
    cfg = _cfg()
    del cfg[section][key]
    with pytest.raises(KeyError):
        SimulationContext.from_dict(cfg)
