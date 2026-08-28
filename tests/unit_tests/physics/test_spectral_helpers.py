"""Tests for spectral_helpers.py (normalization, subgrid centering, reconstruction).

Note: an earlier `_cp` regression test targeted `_get_array_module` /
`_get_kernels`, which were removed when the runtime backend dispatch was dropped
from this module — there is no longer any such code path to guard.
"""

import numpy as np
import pytest

from fast_heat_solv.core.parameters import GeomParams
from fast_heat_solv.core.vector import Vec3
from fast_heat_solv.physics import spectral_cpu_kernels as k
from fast_heat_solv.physics import spectral_helpers as hp


def _geom():
    return GeomParams(size=Vec3(4e-3, 2e-3, 1e-3), n=Vec3(8, 6, 4))


def test_C_coef_formula():
    # DCT normalisation: sqrt(2/L) for every mode except the k=0 mode (sqrt(1/L)).
    C = hp._C_coef(4, 2.0)
    assert C[0] == pytest.approx(np.sqrt(1.0 / 2.0))
    np.testing.assert_allclose(C[1:], np.sqrt(2.0 / 2.0))

# Each time step, as the laser moves, the fine mesh box has to re-center on 
# the laser's new (x, y) — while staying inside the domain.

# _calculate_subgrid_indices centres a box of `n_box` cells on physical position
# `pos` inside a fine grid of `n_total_fine` cells (spacing dx), clamping the box
# to stay within the grid. It returns (start, end, offset): the box's [start, end)
# cell range and pos's index *offset within that box*. Here dx=1 so pos maps
# straight to a cell index, n_box=4, and the fine grid has 20 cells.
@pytest.mark.parametrize(
    "pos,expected",
    [
        # pos -> cell 5, mid-grid: box of 4 centred -> [3, 7), pos at offset 2. No clamp.
        (5.0, (3, 7, 2)),
        # pos -> cell 0 (low edge): box can't extend below 0 -> pinned to [0, 4), offset 0.
        (0.0, (0, 4, 0)),
        # pos past the grid -> clamped to the last cell 19; box pinned to the
        # rightmost [16, 20) so it still fits, pos at offset 3 (the box's last cell).
        (100.0, (16, 20, 3)),
    ],
)
def test_calculate_subgrid_indices(pos, expected):
    assert hp._calculate_subgrid_indices(pos, dx=1.0, n_total_fine=20, n_box=4) == expected


def test_calculate_subgrid_indices_box_bigger_than_grid():
    start, end, _ = hp._calculate_subgrid_indices(5.0, dx=1.0, n_total_fine=20, n_box=30)
    assert (start, end) == (0, 30)  # start pinned to 0 when the box can't fit


def test_cosine_basis_shape_and_values():
    coords = np.array([0.0, 1.0, 2.0])
    B = hp._cosine_basis_along_axis(3, 2.0, coords)
    assert B.shape == (3, len(coords))
    np.testing.assert_allclose(B[0], 1.0)  # k=0 mode is constant 1
    np.testing.assert_allclose(B, np.cos(np.pi * np.arange(3)[:, None] * coords[None, :] / 2.0))


def test_at_points_rejects_bad_coords_shape():
    # Shape is validated before any state is touched, so dummies are fine.
    with pytest.raises(ValueError):
        hp.reconstruct_temperature_volume_at_points(None, None, None, None, np.zeros((5, 2)))


@pytest.fixture
def state_with_modes(tiny_context):
    c = tiny_context
    st = k.SpectralSolverState(c.mat, c.geom, c.num, c.fine)
    st.a = np.zeros((c.num.nz, c.num.ny, c.num.nx), dtype=np.float32)
    st.a[0, 0, 0] = 1000.0   # mean mode
    st.a[1, 1, 1] = 50.0     # a low-frequency wiggle
    return st


def test_reconstruct_volume_requires_prepared_bases(state_with_modes):
    # The tensor-product reconstruction needs prepare_full_reconstruction() first.
    with pytest.raises(RuntimeError):
        hp.reconstruct_temperature_volume(state_with_modes.a, state_with_modes)


def test_dct_reconstruction_matches_tensor_product(tiny_context, state_with_modes):
    # The fast DCT path must equal the explicit tensor-product reconstruction
    # Important to check that the DCT's implicit normalization matches the C_coef normalization
    # (easy source of error)
    state_with_modes.grid.prepare_full_reconstruction(tiny_context.geom)
    T_dct = hp.reconstruct_temperature_DCT(state_with_modes.a, state_with_modes)
    T_ref = hp.reconstruct_temperature_volume(state_with_modes.a, state_with_modes)
    assert T_dct.shape == T_ref.shape
    np.testing.assert_allclose(T_dct, T_ref, rtol=1e-3, atol=1.0)


def test_save_temp_profiles_laser_center_requires_position():
    # center="laser" needs a laser position to sample around.
    with pytest.raises(ValueError):
        hp.save_temp_profiles(None, None, _geom(), None, laser_position=None, center="laser")


def test_save_temp_profiles_unknown_center_raises():
    # An unrecognised center mode is rejected.
    with pytest.raises(ValueError):
        hp.save_temp_profiles(None, None, _geom(), None, laser_position=(0.0, 0.0), center="bogus")


def test_save_temp_profiles_returns_xyz(tiny_context, state_with_modes):
    # returns x/y/z line profiles, each num_points long and finite.
    geom = tiny_context.geom
    pos = (geom.size.x / 2, geom.size.y / 2)
    profiles = hp.save_temp_profiles(
        state_with_modes.a, tiny_context.num, geom, state_with_modes,
        laser_position=pos, center="laser", num_points=50,
    )
    assert set(profiles) == {"x", "y", "z"}
    for coords, temps in profiles.values():
        assert len(coords) == 50 and len(temps) == 50
        assert np.all(np.isfinite(temps))
