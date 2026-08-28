"""Precision of the generated property-expression sources.
"""

import numpy as np
import pytest

from fast_heat_solv.core.property_expr import PropertyExpr


@pytest.mark.parametrize("dtype, cast", [
    (np.float32, "np.float32"),
    (np.float64, "np.float64"),
])
def test_py_source_literals_match_dtype(dtype, cast):
    src = PropertyExpr("9.248 + 0.01571 * T").to_py_source(dtype)
    assert src.count(cast) == 2, src
    other = "np.float64" if dtype is np.float32 else "np.float32"
    assert other not in src


@pytest.mark.parametrize("dtype, cast", [
    (np.float32, "np.float32"),
    (np.float64, "np.float64"),
])
def test_py_source_constant_matches_dtype(dtype, cast):
    assert PropertyExpr("7900.0").to_py_source(dtype) == f"{cast}(7900.0)"


def test_c_source_single_precision_suffixes_and_intrinsics():
    src = PropertyExpr("sqrt(T) + 2.5").to_c_source(np.float32)
    assert "sqrtf(" in src and "2.500000000e+00f" in src


def test_c_source_double_precision_drops_suffixes():
    src = PropertyExpr("sqrt(T) + 2.5").to_c_source(np.float64)
    assert "sqrtf(" not in src and "sqrt(" in src
    assert "2.500000000e+00" in src and "e+00f" not in src


def test_c_source_pow_follows_precision():
    assert "powf(" in PropertyExpr("T ** 2").to_c_source(np.float32)
    single = PropertyExpr("T ** 2").to_c_source(np.float64)
    assert "powf(" not in single and "pow(" in single


def test_default_precision_is_float32():
    """The default must stay single precision: it is the documented default
    for ``simulation.dtype`` and every existing caller relies on it."""
    e = PropertyExpr("9.248 + 0.01571 * T")
    assert e.to_py_source() == e.to_py_source(np.float32)
    assert e.to_c_source() == e.to_c_source(np.float32)


def test_unsupported_precision_raises():
    with pytest.raises(ValueError, match="Unsupported property precision"):
        PropertyExpr("1.0").to_py_source(np.float16)
    with pytest.raises(ValueError, match="Unsupported property precision"):
        PropertyExpr("1.0").to_c_source(np.float16)
