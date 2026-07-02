"""Tests for the property-correction global transforms (spectral_ops).

These pin the DCT-II/IDCT-II volume normalisation (property_correction.tex §6,
§10), validated against an explicit brute-force modal sum, and the assembly of
the property correction ``C = P{∇·(k'∇T) - a'∂_tT}``.
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
        project_volume,
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
    # With ∂_t T = 0 the whole correction is the conductivity term ∇·(k'∇T).
    gz, gy, gx = (((model.k(T) - k_bar).astype(np.float32)) * d
                  for d in np.gradient(T, st.grid.dz, st.grid.dy, st.grid.dx))
    div_g = (np.gradient(gx, st.grid.dx, axis=2)
             + np.gradient(gy, st.grid.dy, axis=1)
             + np.gradient(gz, st.grid.dz, axis=0))
    C_ref = project_volume(div_g.astype(np.float32), st)
    np.testing.assert_allclose(C, C_ref, rtol=1e-4,
                               atol=1e-5 * float(np.abs(C_ref).max()))


# ---------------------------------------------------------------------------
# Property correction (Green's first identity, property_correction.tex)
# ---------------------------------------------------------------------------

def _kdep_model():
    from fast_heat_solv.core.properties import MaterialModel
    return MaterialModel.from_config(
        {"k": {"solid": [9.248, 0.01571], "liquid": [12.41, 0.003279]},
         "rho": 7900.0, "Cp": 500.0},
        1674.15, 1697.15,
    )


def test_correction_null_is_exact():
    """k'=a'=0 ⇒ the correction (volume + faces) is exactly zero too."""
    from fast_heat_solv.physics.spectral_ops import (
        assemble_property_correction, reconstruct_volume)

    st = _state(10, 8, 6)
    model = _const_model()
    rng = np.random.default_rng(11)
    a = rng.standard_normal((6, 8, 10)).astype(np.float32) * 50.0
    a[0, 0, 0] += 1.0e4
    T_prev = reconstruct_volume(a, st)
    C = assemble_property_correction(st, a, T_prev, 1e-6, model,
                                     15.0, 7900.0 * 500.0)
    assert float(np.max(np.abs(C))) == 0.0


def test_correction_assembles_volume_only():
    """The correction is the volume projection of ``s_a + ∇·(k'∇T)``.

    The conductivity boundary term ``-∮k'∂_nT Φ dS`` is not assembled inside the
    correction: using the Neumann BC it is folded into the prescribed surface flux
    (rescaled by ``k̄/k``) by the solver. So ``assemble_property_correction`` must
    return exactly the merged volume DCT, with no face contribution.
    """
    from fast_heat_solv.physics.spectral_ops import (
        assemble_property_correction, project_volume, reconstruct_volume)

    st = _state(40, 32, 24)
    model = _kdep_model()
    k_bar, a_bar = 13.0, 7900.0 * 500.0
    rng = np.random.default_rng(12)
    nz, ny, nx = 24, 32, 40
    decay = (np.exp(-(np.arange(nx) / 6.0) ** 2)[None, None, :]
             * np.exp(-(np.arange(ny) / 6.0) ** 2)[None, :, None]
             * np.exp(-(np.arange(nz) / 6.0) ** 2)[:, None, None])
    a = (rng.standard_normal((nz, ny, nx)).astype(np.float32) * 200.0 * decay)
    a[0, 0, 0] += 1.2e4
    T_prev = reconstruct_volume(a, st)
    dt = 1e-6

    C = assemble_property_correction(st, a, T_prev, dt, model, k_bar, a_bar)

    # Reference: volume projection of f = -a' ∂_t T + ∇·(k' ∇T), no face terms.
    T = reconstruct_volume(a, st)
    kp = (model.k(T) - k_bar).astype(np.float32)
    ap = (model.a(T) - a_bar).astype(np.float32)
    gz, gy, gx = (kp * d for d in np.gradient(T, st.grid.dz, st.grid.dy, st.grid.dx))
    div_g = (np.gradient(gx, st.grid.dx, axis=2)
             + np.gradient(gy, st.grid.dy, axis=1)
             + np.gradient(gz, st.grid.dz, axis=0))
    s_a = -ap * (T - T_prev) / np.float32(dt)
    C_ref = project_volume((s_a + div_g).astype(np.float32), st)

    np.testing.assert_allclose(C, C_ref, rtol=1e-4,
                               atol=1e-5 * float(np.abs(C_ref).max()))
