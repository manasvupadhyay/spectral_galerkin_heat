"""Tests for the MathBackend container and get_backend dispatch (backends/)."""

import numpy as np
import pytest

import spectral_galerkin_heat.physics.spectral_cpu_kernels as cpu_kernels
from spectral_galerkin_heat.backends import MathBackend, NumpyBackend, get_backend
from spectral_galerkin_heat.backends.base import to_host


def test_backend_container_stores_fields():
    # MathBackend is a thin container — it just holds name/xp/kernels.
    b = MathBackend(name="x", xp=np, kernels=cpu_kernels)
    assert (b.name, b.xp, b.kernels) == ("x", np, cpu_kernels)


def test_repr_contains_name():
    # backend name.
    assert "name='numpy'" in repr(NumpyBackend())


def test_numpy_backend_wires_cpu_kernels():
    # numpy + the CPU kernel module the solver calls.
    b = NumpyBackend()
    assert b.name == "numpy"
    assert b.xp is np
    assert b.kernels is cpu_kernels


def test_get_backend_numpy():
    # get_backend
    b = get_backend("numpy")
    assert isinstance(b, NumpyBackend)
    assert b.name == "numpy"


def test_get_backend_defaults_to_numpy():
    # No name given -> CPU (numpy) is the default.
    assert isinstance(get_backend(), NumpyBackend)


def test_get_backend_unknown_raises():
    # An unknown backend name is a config error.
    with pytest.raises(ValueError):
        get_backend("foo")


def test_to_numpy_host_passthrough():
    a = np.arange(6.0).reshape(2, 3)
    out = NumpyBackend().to_numpy(a)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, a)


def test_to_host_on_numpy_array():
    a = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(to_host(a), a)
