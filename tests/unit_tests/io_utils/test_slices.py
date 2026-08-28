"""Tests for the slice interpolation data path (io_utils/slices.py).

"""

import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")  # no display needed


def _ramp_grid():
    # Structured grid with a linear field T = 2x + 3y + 5z.
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)
    z = np.linspace(0.0, 0.5, 6)
    T = 2 * x[None, None, :] + 3 * y[None, :, None] + 5 * z[:, None, None]  # (z, y, x)
    return {"x": x, "y": y, "z": z, "T": T, "type": "structured"}


def test_get_slice_recovers_linear_field_z_normal():
    from spectral_galerkin_heat.io_utils.slices import _get_slice

    data = _ramp_grid()
    cz = 0.25
    U, V, S, *_ = _get_slice(data, "z", center=(0.5, 0.5, cz),
                             width=0.4, height=0.4, resolution=20)
    # z-normal -> plane is X(U)-Y(V); the linear field must be reproduced exactly.
    np.testing.assert_allclose(S, 2 * U + 3 * V + 5 * cz, atol=1e-6)


def test_get_slice_bad_normal_raises():
    # Only x/y/z planes exist; any other normal is a clear error.
    from spectral_galerkin_heat.io_utils.slices import _get_slice

    with pytest.raises(ValueError):
        _get_slice(_ramp_grid(), "w", center=(0.5, 0.5, 0.25), width=0.4, height=0.4)


def test_axis_labels_mapping_and_bad_normal():
    from spectral_galerkin_heat.io_utils.slices import _axis_labels

    assert _axis_labels("z") == ("X (m)", "Y (m)", "x", "y")
    with pytest.raises(ValueError):
        _axis_labels("w")


def test_get_slice_returns_labels_for_normal():
    from spectral_galerkin_heat.io_utils.slices import _get_slice

    *_, xlabel, ylabel = _get_slice(_ramp_grid(), "z", center=(0.5, 0.5, 0.25),
                                    width=0.4, height=0.4, resolution=10)
    assert (xlabel, ylabel) == ("X (m)", "Y (m)")


def test_get_slice_reverse_axis_flips_horizontally():
    from spectral_galerkin_heat.io_utils.slices import _get_slice

    data, kw = _ramp_grid(), dict(center=(0.5, 0.5, 0.25), width=0.4, height=0.4, resolution=20)
    _, _, base, *_ = _get_slice(data, "z", **kw)
    _, _, rev, *_ = _get_slice(data, "z", reverse_axes=("x",), **kw)
    np.testing.assert_allclose(rev, np.fliplr(base), atol=1e-6)
