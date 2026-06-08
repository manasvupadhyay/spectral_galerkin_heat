"""Tests for the MathBackend container and get_backend dispatch (backends/)."""

import numpy as np
import pytest

import fast_heat_solv.physics.spectral_cpu_kernels as cpu_kernels
from fast_heat_solv.backends import MathBackend, NumpyBackend, get_backend
from fast_heat_solv.backends.base import to_host


def test_backend_container_stores_fields():
    # MathBackend is a thin container — it just holds name/xp/kernels.
    b = MathBackend(name="x", xp=np, kernels=cpu_kernels)
    assert (b.name, b.xp, b.kernels) == ("x", np, cpu_kernels)


def test_repr_contains_name():
    assert "name='numpy'" in repr(NumpyBackend())


def test_numpy_backend_wires_cpu_kernels():
    b = NumpyBackend()
    assert b.name == "numpy"
    assert b.xp is np
    assert b.kernels is cpu_kernels


def test_get_backend_numpy():
    b = get_backend("numpy")
    assert isinstance(b, NumpyBackend)
    assert b.name == "numpy"


def test_get_backend_defaults_to_numpy():
    assert isinstance(get_backend(), NumpyBackend)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("foo")


def test_to_numpy_host_passthrough():
    # On a host array to_numpy is a no-op asarray — returns equal host data.
    a = np.arange(6.0).reshape(2, 3)
    out = NumpyBackend().to_numpy(a)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, a)


def test_to_host_on_numpy_array():
    a = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(to_host(a), a)
