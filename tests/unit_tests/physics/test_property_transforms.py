"""Tests for the property-correction global transforms (spectral_ops).

These pin the two things easiest to get wrong (property_correction.tex §6, §10):
the DCT-II/IDCT-II volume normalisation and the mixed sine/cosine transform
(including the DST-II mode-index shift), validated against an explicit
brute-force modal sum.
"""

import numpy as np
import pytest

from fast_heat_solv.core.parameters import GeomParams, NumParams
from fast_heat_solv.core.vector import Vec3


def _state(nx, ny, nz, Lx=1.0e-3, Ly=0.7e-3, Lz=0.4e-3):
    from fast_heat_solv.physics.spectral_cpu_kernels import SpectralSolverState

    class _Phys:  # minimal — only k/rho/Cp used by precompute_K_KK
        k, rho, Cp = 15.0, 7900.0, 500.0
        T_solidus = T_liquidus = 0.0
        L_f = 0.0

    geom = GeomParams(size=Vec3(Lx, Ly, Lz), n=Vec3(nx, ny, nz))
    num = NumParams(dt=1e-6, nx=nx, ny=ny, nz=nz)
    from fast_heat_solv.core.parameters import FineMeshParams
    return SpectralSolverState(_Phys(), geom, num, FineMeshParams())


def _grid_coords(SsState):
    g = SsState.grid
    return np.asarray(g.x), np.asarray(g.y), np.asarray(g.z)


# ---------------------------------------------------------------------------
# Volume DCT-II / IDCT-II round trip
# ---------------------------------------------------------------------------

def test_volume_roundtrip_identity():
    from fast_heat_solv.physics.spectral_ops import project_volume, reconstruct_volume

    st = _state(12, 10, 8)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 10, 12)).astype(np.float32)
    a_rt = project_volume(reconstruct_volume(a, st), st)
    np.testing.assert_allclose(a_rt, a, rtol=2e-4, atol=2e-4)


def test_reconstruct_volume_matches_modal_sum():
    """reconstruct_volume(a) == sum_mnp a_mnp C_m C_n C_p cos(...) on the grid."""
    from fast_heat_solv.physics.spectral_ops import reconstruct_volume

    st = _state(6, 5, 4)
    nz, ny, nx = 4, 5, 6
    rng = np.random.default_rng(1)
    a = rng.standard_normal((nz, ny, nx)).astype(np.float32)
    T = reconstruct_volume(a, st)

    x, y, z = _grid_coords(st)
    Cx, Cy, Cz = (np.asarray(c) for c in st.grid.C)
    Lx, Ly, Lz = 1.0e-3, 0.7e-3, 0.4e-3
    # reconstruct_volume(a) == sum_mnp a_mnp C_m C_n C_p cos cos cos: the C-norm
    # cosine series equals IDCT_II(a)/sqrt(dV) exactly (the /sqrt_dV folds in here).
    cos_x = Cx[:, None] * np.cos(np.pi * np.arange(nx)[:, None] * x[None, :] / Lx)
    cos_y = Cy[:, None] * np.cos(np.pi * np.arange(ny)[:, None] * y[None, :] / Ly)
    cos_z = Cz[:, None] * np.cos(np.pi * np.arange(nz)[:, None] * z[None, :] / Lz)
    T_ref = np.einsum('pnm,pk,nj,mi->kji', a, cos_z, cos_y, cos_x)
    np.testing.assert_allclose(T, T_ref, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Mixed sine/cosine transform vs the brute-force -∫ g ∂_d Φ dV
# ---------------------------------------------------------------------------

def _brute_force_Ck_x(g, st):
    """Explicit  C^k_x[m,n,p] = (mπ/Lx) ∫ g · C_m sin(mπx/Lx) C_n cos C_p cos dV."""
    nz, ny, nx = g.shape
    x, y, z = _grid_coords(st)
    Cx, Cy, Cz = (np.asarray(c) for c in st.grid.C)
    Lx, Ly, Lz = 1.0e-3, 0.7e-3, 0.4e-3
    dV = float(st.grid.sqrt_dV) ** 2
    sin_x = Cx[:, None] * np.sin(np.pi * np.arange(nx)[:, None] * x[None, :] / Lx)
    cos_y = Cy[:, None] * np.cos(np.pi * np.arange(ny)[:, None] * y[None, :] / Ly)
    cos_z = Cz[:, None] * np.cos(np.pi * np.arange(nz)[:, None] * z[None, :] / Lz)
    # S_x{g}[p,n,m] = sum_kji g[k,j,i] sin_x[m,i] cos_y[n,j] cos_z[p,k] dV
    Sx = np.einsum('kji,mi,nj,pk->pnm', g, sin_x, cos_y, cos_z) * dV
    kx = (np.pi * np.arange(nx) / Lx)
    return kx[None, None, :] * Sx


def test_mixed_transform_x_matches_brute_force():
    from fast_heat_solv.physics.spectral_ops import _mixed_sine_transform

    st = _state(6, 5, 4)
    rng = np.random.default_rng(2)
    g = rng.standard_normal((4, 5, 6)).astype(np.float32)

    Sx = _mixed_sine_transform(g, 2, st)
    kx = (np.pi * np.arange(6) / 1.0e-3)
    Ck_x = kx[None, None, :] * Sx
    np.testing.assert_allclose(Ck_x, _brute_force_Ck_x(g, st), rtol=1e-3, atol=1e-2)


def test_mixed_transform_zeroes_constant_mode():
    """The differentiated axis annihilates its m=0 (constant) mode."""
    from fast_heat_solv.physics.spectral_ops import _mixed_sine_transform

    st = _state(6, 5, 4)
    g = np.random.default_rng(3).standard_normal((4, 5, 6)).astype(np.float32)
    Sx = _mixed_sine_transform(g, 2, st)
    np.testing.assert_allclose(Sx[:, :, 0], 0.0, atol=1e-6)
    Sz = _mixed_sine_transform(g, 0, st)
    np.testing.assert_allclose(Sz[0, :, :], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# assemble_property_correction — null behaviour (tex §8.1)
# ---------------------------------------------------------------------------

def _const_model(k=15.0, rho=7900.0, cp=500.0):
    from fast_heat_solv.core.properties import MaterialModel
    # Branch form (so is_constant flagging is exercised) with equal branches.
    return MaterialModel.from_config(
        {"k": {"solid": [k], "liquid": [k]},
         "rho": {"solid": [rho], "liquid": [rho]},
         "Cp": {"solid": [cp], "liquid": [cp]}},
        1674.15, 1697.15,
    )


def test_correction_zero_when_fluctuations_vanish():
    """k'=a'=0 (reference equals the constant property) ⇒ C ≈ 0 (tex null test)."""
    from fast_heat_solv.physics.spectral_ops import (
        assemble_property_correction,
        reconstruct_volume,
    )

    st = _state(10, 8, 6)
    model = _const_model()
    k_bar = 15.0
    a_bar = 7900.0 * 500.0

    rng = np.random.default_rng(7)
    a_trial = rng.standard_normal((6, 8, 10)).astype(np.float32) * 50.0
    a_trial[0, 0, 0] += 1.0e4  # a realistic warm mean mode
    T_prev = reconstruct_volume(a_trial, st)  # ∂_t T = 0 too

    C = assemble_property_correction(st, a_trial, T_prev, 1e-6, model, k_bar, a_bar)
    # k'=0 ⇒ C^k=0; a'=0 ⇒ C^a=0. Residual is pure fp round-off.
    assert float(np.max(np.abs(C))) < 1e-2 * float(np.max(np.abs(a_trial)))


def test_capacity_correction_zero_for_steady_field():
    """Uniform-in-time field ⇒ ∂_t T = 0 ⇒ no capacity correction contribution."""
    from fast_heat_solv.physics.spectral_ops import (
        assemble_property_correction,
        conductivity_correction_modes,
        reconstruct_volume,
    )
    from fast_heat_solv.core.properties import MaterialModel

    st = _state(10, 8, 6)
    model = MaterialModel.from_config(
        {"k": {"solid": [9.248, 0.01571], "liquid": [12.41, 0.003279]},
         "rho": 7900.0, "Cp": 500.0},
        1674.15, 1697.15,
    )
    k_bar, a_bar = 13.0, 7900.0 * 500.0

    rng = np.random.default_rng(8)
    a_trial = rng.standard_normal((6, 8, 10)).astype(np.float32)
    a_trial[0, 0, 0] += 1.0e4
    T = reconstruct_volume(a_trial, st)

    C = assemble_property_correction(st, a_trial, T, 1e-6, model, k_bar, a_bar)
    # With ∂_t T = 0 the whole correction is the conductivity term.
    T_grad = np.gradient(T, st.grid.dz, st.grid.dy, st.grid.dx)
    kp = (model.k(T) - k_bar).astype(np.float32)
    Ck = conductivity_correction_modes(kp * T_grad[2], kp * T_grad[1], kp * T_grad[0], st)
    np.testing.assert_allclose(C, Ck, rtol=1e-4, atol=1e-4)


def test_conductivity_correction_sums_three_axes():
    from fast_heat_solv.physics.spectral_ops import (
        _mixed_sine_transform,
        conductivity_correction_modes,
    )

    st = _state(6, 5, 4)
    rng = np.random.default_rng(4)
    gx = rng.standard_normal((4, 5, 6)).astype(np.float32)
    gy = rng.standard_normal((4, 5, 6)).astype(np.float32)
    gz = rng.standard_normal((4, 5, 6)).astype(np.float32)

    Ck = conductivity_correction_modes(gx, gy, gz, st)
    grid = st.grid
    expect = (np.asarray(grid.kx)[None, None, :] * _mixed_sine_transform(gx, 2, st)
              + np.asarray(grid.ky)[None, :, None] * _mixed_sine_transform(gy, 1, st)
              + np.asarray(grid.kz)[:, None, None] * _mixed_sine_transform(gz, 0, st))
    np.testing.assert_allclose(Ck, expect, rtol=1e-5, atol=1e-5)
