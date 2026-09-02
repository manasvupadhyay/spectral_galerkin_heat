"""Tests for grid + ETD propagators (physics/spectral_state.py).

Pure NumPy (no numba/FFT), so these stay fast and run by default.
"""

import numpy as np
import pytest

from spectral_galerkin_heat.core.parameters import GeomParams, MaterialParams, NumParams
from spectral_galerkin_heat.core.vector import Vec3
from spectral_galerkin_heat.physics.spectral_state import SpectralGrid, precompute_K_KK


def _geom(nx=8, ny=6, nz=4, Lx=4e-3, Ly=2e-3, Lz=1e-3):
    # Asymmetric by so any axis mix-up is caught.
    return GeomParams(size=Vec3(Lx, Ly, Lz), n=Vec3(nx, ny, nz))


def _mat(rho=7900.0, k=15.0, Cp=500.0):
    return MaterialParams(rho=rho, k=k, Cp=Cp)


def _num(dt=1e-6, nx=8, ny=6, nz=4):
    return NumParams(dt=dt, nx=nx, ny=ny, nz=nz)


def test_grid_coords_are_cell_centered():
    # Global grid samples cell centres ((i+0.5)*d), not nodes — pin that convention.
    g = _geom()
    grid = SpectralGrid(g, np)
    np.testing.assert_allclose(grid.x, (np.arange(8) + 0.5) * g.d.x, rtol=1e-6)
    np.testing.assert_allclose(grid.z, (np.arange(4) + 0.5) * g.d.z, rtol=1e-6)


def test_K_KK_zero_mode():
    # At k=0 the propagator is identity and KK -> dt/(rho*Cp) (the phi_1 limit).
    mat, num, g = _mat(), _num(), _geom()
    K, KK = precompute_K_KK(mat, num, g, np)
    assert K[0, 0, 0] == pytest.approx(1.0)
    assert KK[0, 0, 0] == pytest.approx(num.dt / (mat.rho * mat.Cp), rel=1e-6)


def test_K_shape_is_zyx():
    # Propagators are indexed [z, y, x]; the asymmetric mesh catches a transpose.
    K, _ = precompute_K_KK(_mat(), _num(), _geom(), np)
    assert K.shape == (4, 6, 8)  # (nz, ny, nx)


def test_K_decay_matches_exp_formula():
    # K = exp(-alpha * k^2 * dt) for the first mode along each axis.
    mat, num, g = _mat(), _num(), _geom()
    alpha = mat.k / (mat.rho * mat.Cp)
    K, _ = precompute_K_KK(mat, num, g, np)
    kz = np.pi / g.size.z
    kx = np.pi / g.size.x
    assert K[1, 0, 0] == pytest.approx(np.exp(-alpha * kz**2 * num.dt), rel=1e-5)
    assert K[0, 0, 1] == pytest.approx(np.exp(-alpha * kx**2 * num.dt), rel=1e-5)


def test_K_axis_ordering_uses_zyx():
    # The .zyx() reversal in precompute_K_KK: the first z-mode must use Lz and the
    # first x-mode Lx. With Lx != Lz they differ — a reversed (x,y,z) order fails.
    g = _geom(Lx=4e-3, Lz=1e-3)
    K, _ = precompute_K_KK(_mat(), _num(), g, np)
    assert K[1, 0, 0] != K[0, 0, 1]
