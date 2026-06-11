"""Tests for temperature-dependent properties (core/properties.py).

The polynomial-branch values are checked against the Chadwick 316L expressions
hard-coded in the finite-element reference solver
(``andreas_heat_solv/.../Theo_cuboid_FE_temp_dep.py``); keeping these in sync is
what lets the spectral solver be compared to the FE field.
"""

import numpy as np
import pytest

from fast_heat_solv.core.properties import (
    MaterialModel,
    TempProperty,
    liquid_fraction,
)

# Chadwick 316L (ascending powers of T), copied from the FE reference script.
_T_S, _T_L = 1674.15, 1697.15
_RHO_S = [8084.2, -0.42086, -3.8942e-5]
_RHO_L = [7432.7, 0.039338, -1.8007e-4]
_CP_S = [458.98, 0.1328]
_CP_L = [769.86]
_K_S = [9.248, 0.01571]
_K_L = [12.41, 0.003279]


def _fe_rho(T):
    lf = 0.0 if T < _T_S else 1.0 if T > _T_L else (T - _T_S) / (_T_L - _T_S)
    rs = 8084.2 - 0.42086 * T - 3.8942e-5 * T**2
    rl = 7432.7 + 0.039338 * T - 1.8007e-4 * T**2
    return (1 - lf) * rs + lf * rl


def _fe_k(T):
    lf = 0.0 if T < _T_S else 1.0 if T > _T_L else (T - _T_S) / (_T_L - _T_S)
    return (1 - lf) * (9.248 + 0.01571 * T) + lf * (12.41 + 0.003279 * T)


# ---------------------------------------------------------------------------
# liquid_fraction
# ---------------------------------------------------------------------------

def test_liquid_fraction_clamps_and_ramps():
    assert liquid_fraction(_T_S - 100.0, _T_S, _T_L) == 0.0
    assert liquid_fraction(_T_L + 100.0, _T_S, _T_L) == 1.0
    mid = 0.5 * (_T_S + _T_L)
    assert liquid_fraction(mid, _T_S, _T_L) == pytest.approx(0.5)


def test_liquid_fraction_array_clips():
    T = np.array([0.0, _T_S, 0.5 * (_T_S + _T_L), _T_L, 5000.0], dtype=np.float64)
    fl = liquid_fraction(T, _T_S, _T_L)
    np.testing.assert_allclose(fl, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_liquid_fraction_degenerate_band_is_step():
    # T_liquidus <= T_solidus → step at the solidus, no div-by-zero.
    assert liquid_fraction(1000.0, _T_S, _T_S) == 0.0
    assert liquid_fraction(_T_S + 1.0, _T_S, _T_S) == 1.0


# ---------------------------------------------------------------------------
# TempProperty — scalar
# ---------------------------------------------------------------------------

def test_scalar_property_is_constant_everywhere():
    p = TempProperty.from_config(15.0, _T_S, _T_L)
    assert p.is_constant
    for T in (300.0, 1500.0, 3000.0):
        assert p(T) == pytest.approx(15.0)


def test_value_unit_mapping_is_scalar():
    p = TempProperty.from_config({"value": 13.85, "unit": "W/(m.K)"}, _T_S, _T_L)
    assert p.is_constant
    assert p(2000.0) == pytest.approx(13.85)


def test_scalar_property_array_input():
    p = TempProperty.from_config(7.0, _T_S, _T_L)
    out = p(np.array([300.0, 2000.0, 4000.0]))
    np.testing.assert_allclose(out, 7.0)


# ---------------------------------------------------------------------------
# TempProperty — polynomial branches vs the FE reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [300.0, 1000.0, _T_S, 0.5 * (_T_S + _T_L), _T_L, 3000.0])
def test_k_branches_match_fe(T):
    k = TempProperty.from_config({"solid": _K_S, "liquid": _K_L}, _T_S, _T_L)
    assert not k.is_constant
    assert float(k(T)) == pytest.approx(_fe_k(T), rel=1e-6)


@pytest.mark.parametrize("T", [300.0, 1000.0, _T_S, 0.5 * (_T_S + _T_L), _T_L, 3000.0])
def test_rho_branches_match_fe(T):
    rho = TempProperty.from_config({"solid": _RHO_S, "liquid": _RHO_L}, _T_S, _T_L)
    assert float(rho(T)) == pytest.approx(_fe_rho(T), rel=1e-6)


def test_branches_array_matches_scalar_loop():
    k = TempProperty.from_config({"solid": _K_S, "liquid": _K_L}, _T_S, _T_L)
    T = np.linspace(300.0, 3000.0, 50)
    np.testing.assert_allclose(k(T), [float(k(float(t))) for t in T], rtol=1e-6)


def test_missing_branch_falls_back():
    # Only a solid branch given → liquid mirrors it.
    k = TempProperty.from_config({"solid": _K_S}, _T_S, _T_L)
    assert float(k(3000.0)) == pytest.approx(9.248 + 0.01571 * 3000.0, rel=1e-6)


def test_empty_branch_rejected():
    with pytest.raises(ValueError):
        TempProperty.from_config({"solid": [], "liquid": _K_L}, _T_S, _T_L)


def test_nonfinite_coeff_rejected():
    with pytest.raises(ValueError):
        TempProperty.from_config({"solid": [1.0, np.nan], "liquid": _K_L}, _T_S, _T_L)


# ---------------------------------------------------------------------------
# MaterialModel
# ---------------------------------------------------------------------------

def _full_model():
    return MaterialModel.from_config(
        {"k": {"solid": _K_S, "liquid": _K_L},
         "rho": {"solid": _RHO_S, "liquid": _RHO_L},
         "Cp": {"solid": _CP_S, "liquid": _CP_L}},
        _T_S, _T_L,
    )


def test_model_a_is_rho_times_c():
    m = _full_model()
    T = 1200.0
    assert float(m.a(T)) == pytest.approx(float(m.rho(T)) * float(m.c(T)), rel=1e-9)


def test_model_reference_constants_at_T0():
    m = _full_model()
    T0 = 293.0
    k_bar, a_bar, rho_bar, c_bar = m.reference_constants(T0)
    assert k_bar == pytest.approx(_fe_k(T0), rel=1e-6)
    assert rho_bar == pytest.approx(_fe_rho(T0), rel=1e-6)
    assert a_bar == pytest.approx(rho_bar * c_bar, rel=1e-9)


def test_model_none_when_all_scalar():
    m = MaterialModel.from_config({"k": 15.0, "rho": 7900.0, "Cp": 500.0}, _T_S, _T_L)
    assert m is None


def test_model_mixed_scalar_and_branches():
    # k temperature-dependent, rho/Cp scalar → still a model, with constant flags.
    m = MaterialModel.from_config(
        {"k": {"solid": _K_S, "liquid": _K_L}, "rho": 7900.0, "Cp": 500.0},
        _T_S, _T_L,
    )
    assert m is not None
    assert not m.is_constant
    assert m.rho.is_constant and m.c.is_constant and not m.k.is_constant
    assert float(m.rho(3000.0)) == pytest.approx(7900.0)


def test_model_is_constant_when_branches_all_constant():
    # Branches present but numerically constant on both phases.
    m = MaterialModel.from_config(
        {"k": {"solid": [15.0], "liquid": [15.0]}, "rho": 7900.0, "Cp": 500.0},
        _T_S, _T_L,
    )
    assert m is not None
    assert m.is_constant
