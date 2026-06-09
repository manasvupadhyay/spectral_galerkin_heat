"""Tests for solver dispatch (solvers/__init__.py build_solver)."""

import pytest

from fast_heat_solv.backends import NumpyBackend
from fast_heat_solv.solvers import SpectralSolver, build_solver
from fast_heat_solv.solvers.spectral_cpu_linear import SpectralSolverCPULinear


class _Ctx:
    # build_solver only reads method/backend, so a tiny stub is enough.
    def __init__(self, method="spectral", backend="cpu"):
        self.method = method
        self.backend = backend


def test_build_solver_cpu():
    # "cpu" -> the full SpectralSolver on the NumPy backend.
    s = build_solver(_Ctx(backend="cpu"))
    assert isinstance(s, SpectralSolver)
    assert isinstance(s.backend, NumpyBackend)


def test_build_solver_cpu_linear():
    # "cpu_linear" selects the linear-only solver variant.
    assert isinstance(build_solver(_Ctx(backend="cpu_linear")), SpectralSolverCPULinear)


def test_build_solver_unknown_method_raises():
    # Only "spectral" is wired (FEM is a stub) — other methods are a clear error.
    with pytest.raises(ValueError):
        build_solver(_Ctx(method="fem"))


def test_build_solver_unknown_backend_raises():
    # Unrecognised backend is rejected, not silently defaulted.
    with pytest.raises(ValueError):
        build_solver(_Ctx(backend="quantum"))
