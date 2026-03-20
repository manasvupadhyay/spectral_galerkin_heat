"""
Correctness tests for the interpolation and L2 comparison pipeline
in ``compute_L2_error.py``.

All tests use **synthetic** data (no XDMF / HDF5 files) so they run
in a few seconds with no external dependencies beyond numpy + scipy.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Import data classes from io_utils
from fast_heat_solv.io_utils import StructuredField, UnstructuredField

# Add research/ to path so we can import compute_L2_error computation functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research'))

from compute_L2_error import (
    _evaluate_on_grid,
    compute_L2,
    compute_L2_unstructured,
    _compute_vertex_volumes,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _analytical(x, y, z, Lx=1.0, Ly=1.0, Lz=1.0):
    """Smooth analytical field: sin(π x/Lx) · sin(π y/Ly) · sin(π z/Lz)."""
    return np.sin(np.pi * x / Lx) * np.sin(np.pi * y / Ly) * np.sin(np.pi * z / Lz)


def _make_structured(nx, ny, nz, Lx=1.0, Ly=1.0, Lz=1.0):
    """Build a StructuredField from the analytical function."""
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)
    Zg, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")
    T = _analytical(Xg, Yg, Zg, Lx, Ly, Lz)
    return StructuredField(x=x, y=y, z=z, T=T)


def _make_unstructured(n_pts, Lx=1.0, Ly=1.0, Lz=1.0, seed=42):
    """Build an UnstructuredField from random scattered points."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform([0, 0, 0], [Lx, Ly, Lz], size=(n_pts, 3))
    T = _analytical(xyz[:, 0], xyz[:, 1], xyz[:, 2], Lx, Ly, Lz)
    return UnstructuredField(xyz=xyz, T=T)


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

class TestEvaluateOnGrid:
    """Tests for ``_evaluate_on_grid`` — the core interpolation dispatch."""

    def test_structured_exact_match(self):
        """When eval grid == source grid, result should be exact (copy path)."""
        sf = _make_structured(10, 12, 8)
        T = _evaluate_on_grid(sf, sf.x, sf.y, sf.z)
        np.testing.assert_array_equal(T, sf.T)

    def test_structured_interpolation(self):
        """Structured field evaluated on a different (coarser) grid."""
        sf = _make_structured(30, 30, 30)
        # Evaluate on a coarser grid strictly inside the domain
        x2 = np.linspace(0.05, 0.95, 15)
        y2 = np.linspace(0.05, 0.95, 15)
        z2 = np.linspace(0.05, 0.95, 10)
        T = _evaluate_on_grid(sf, x2, y2, z2)

        # Compute the analytical reference at the same grid
        Zg, Yg, Xg = np.meshgrid(z2, y2, x2, indexing="ij")
        T_ref = _analytical(Xg, Yg, Zg)

        # Trilinear interpolation of a smooth function on a grid with
        # dx ≈ 1/30 should give errors roughly O(dx²) ~ 1e-3.
        np.testing.assert_allclose(T, T_ref, atol=5e-3)

    def test_unstructured_interpolation(self):
        """Unstructured (scattered) field evaluated on a structured grid."""
        uf = _make_unstructured(5000)
        x = np.linspace(0.1, 0.9, 10)
        y = np.linspace(0.1, 0.9, 10)
        z = np.linspace(0.1, 0.9, 8)
        T = _evaluate_on_grid(uf, x, y, z)

        Zg, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")
        T_ref = _analytical(Xg, Yg, Zg)

        # LinearNDInterpolator on 5000 random points over [0,1]³:
        # piecewise-linear interpolation error depends on triangle size.
        # With 5000 points the typical simplex edge is ~0.1, so expect
        # errors on the order of 1e-1 to 1e-2.
        # We check NaN-free interior points only.
        valid = np.isfinite(T)
        assert valid.sum() > 0.5 * T.size, "Too many NaN — convex-hull issue?"
        np.testing.assert_allclose(T[valid], T_ref[valid], atol=0.15)

    def test_output_shape(self):
        """Output shape should always be (nz, ny, nx)."""
        sf = _make_structured(8, 10, 12)
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 6)
        z = np.linspace(0, 1, 7)
        T = _evaluate_on_grid(sf, x, y, z)
        assert T.shape == (7, 6, 5)


class TestComputeL2:
    """Tests for ``compute_L2`` — numerical integration of error norms."""

    def test_zero_error(self):
        """Identical fields should give L2_abs = 0."""
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        z = np.linspace(0, 1, 20)
        Zg, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")
        T = _analytical(Xg, Yg, Zg)

        result = compute_L2(T, T, x, y, z)
        assert result["L2_abs"] == pytest.approx(0.0, abs=1e-15)
        assert result["Linf"] == pytest.approx(0.0, abs=1e-15)

    def test_constant_offset(self):
        """A uniform offset δ should give L2_abs = δ · √(volume)."""
        x = np.linspace(0, 2, 50)
        y = np.linspace(0, 3, 60)
        z = np.linspace(0, 1, 40)
        Lx, Ly, Lz = 2.0, 3.0, 1.0
        vol = Lx * Ly * Lz

        T_b = np.ones((40, 60, 50))
        delta = 0.5
        T_a = T_b + delta

        result = compute_L2(T_a, T_b, x, y, z)
        expected_L2 = delta * np.sqrt(vol)
        assert result["L2_abs"] == pytest.approx(expected_L2, rel=1e-4)
        assert result["Linf"] == pytest.approx(delta, abs=1e-12)
        assert result["volume"] == pytest.approx(vol, rel=1e-4)

    def test_nan_masking(self):
        """NaN values should be masked out — not pollute the norms."""
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        z = np.linspace(0, 1, 10)
        T_a = np.ones((10, 10, 10))
        T_b = np.ones((10, 10, 10))
        # Inject NaN in a corner
        T_a[0, 0, 0] = np.nan

        result = compute_L2(T_a, T_b, x, y, z)
        assert result["L2_abs"] == pytest.approx(0.0, abs=1e-14)
        assert result["n_nan"] == 1


class TestStructuredVsUnstructured:
    """End-to-end test: structured and unstructured fields representing the
    same analytical solution should yield a small L2 error when compared
    on a common evaluation grid.

    This is the specific scenario the user asked to validate.
    """

    def test_same_function_small_error(self):
        """Structured vs unstructured with same analytical function."""
        Lx, Ly, Lz = 1.0, 1.0, 0.5

        sf = _make_structured(20, 20, 15, Lx, Ly, Lz)
        uf = _make_unstructured(8000, Lx, Ly, Lz)

        # Common evaluation grid (inside domain to avoid convex-hull NaN)
        x = np.linspace(0.05, Lx - 0.05, 18)
        y = np.linspace(0.05, Ly - 0.05, 18)
        z = np.linspace(0.05, Lz - 0.05, 12)

        T_s = _evaluate_on_grid(sf, x, y, z)
        T_u = _evaluate_on_grid(uf, x, y, z)

        result = compute_L2(T_s, T_u, x, y, z)
        print(f"Structured vs Unstructured L2_rel: {result['L2_rel']:.4f}, pct_valid: {result['pct_valid']:.1f}%")
        # Both represent the same smooth function, so the relative L2
        # error should be small (dominated by the unstructured mesh
        # interpolation accuracy).
        assert result["L2_rel"] < 0.15, (
            f"L2_rel = {result['L2_rel']:.4f} — too large for matching fields"
        )
        assert result["pct_valid"] > 90.0


class TestComputeL2Unstructured:
    """Tests for ``compute_L2_unstructured`` — vertex-weighted integration."""

    @staticmethod
    def _make_simple_tet_mesh():
        """Create a simple unit cube mesh with 5 tetrahedra."""
        # Unit cube corners
        xyz = np.array([
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
        ], dtype=np.float64)

        # 5 tetrahedra filling the unit cube
        connectivity = np.array([
            [0, 1, 3, 4],
            [1, 2, 3, 6],
            [1, 3, 4, 6],
            [3, 4, 6, 7],
            [1, 4, 5, 6],
        ], dtype=np.int64)

        return xyz, connectivity

    def test_vertex_volumes_sum_to_mesh_volume(self):
        """Vertex volumes should sum to the mesh volume."""
        xyz, conn = self._make_simple_tet_mesh()
        vertex_vols = _compute_vertex_volumes(xyz, conn)

        # Unit cube has volume 1.0
        assert vertex_vols.sum() == pytest.approx(1.0, rel=1e-10)

    def test_zero_error(self):
        """Identical fields should give L2_abs = 0."""
        xyz, conn = self._make_simple_tet_mesh()
        vertex_vols = _compute_vertex_volumes(xyz, conn)
        T = np.ones(len(xyz))

        result = compute_L2_unstructured(T, T, vertex_vols)

        assert result["L2_abs"] == pytest.approx(0.0, abs=1e-15)
        assert result["L2_rel"] == pytest.approx(0.0, abs=1e-15)
        assert not np.isnan(result["L2_abs"]), "L2_abs should never be NaN"
        assert not np.isnan(result["L2_rel"]), "L2_rel should never be NaN"

    def test_constant_offset(self):
        """A uniform offset should give known L2 error."""
        xyz, conn = self._make_simple_tet_mesh()
        vertex_vols = _compute_vertex_volumes(xyz, conn)
        vol = vertex_vols.sum()

        T_a = np.ones(len(xyz)) * 2.0
        T_b = np.ones(len(xyz)) * 1.5
        delta = 0.5

        result = compute_L2_unstructured(T_a, T_b, vertex_vols)
        expected_L2 = delta * np.sqrt(vol)

        assert result["L2_abs"] == pytest.approx(expected_L2, rel=1e-10)
        assert not np.isnan(result["L2_abs"]), "L2_abs should never be NaN"

    def test_nan_masking_no_nan_output(self):
        """NaN in input should NOT produce NaN in output L2 norms."""
        xyz, conn = self._make_simple_tet_mesh()
        vertex_vols = _compute_vertex_volumes(xyz, conn)

        T_a = np.ones(len(xyz))
        T_b = np.ones(len(xyz))

        # Inject NaN at some vertices (simulating out-of-domain interpolation)
        T_a[0] = np.nan
        T_a[1] = np.nan

        result = compute_L2_unstructured(T_a, T_b, vertex_vols)

        # Critical: L2 values must NOT be NaN
        assert not np.isnan(result["L2_abs"]), "L2_abs must not be NaN when input has NaN"
        assert not np.isnan(result["L2_rel"]), "L2_rel must not be NaN when input has NaN"

        # The remaining valid points have zero error
        assert result["L2_abs"] == pytest.approx(0.0, abs=1e-14)
        assert result["n_nan"] == 2
        assert result["pct_valid"] < 100.0

    def test_partial_nan_correct_error(self):
        """With some NaN values, error should be computed on valid points only."""
        xyz, conn = self._make_simple_tet_mesh()
        vertex_vols = _compute_vertex_volumes(xyz, conn)

        T_a = np.ones(len(xyz)) * 2.0
        T_b = np.ones(len(xyz)) * 1.0  # error = 1.0 at valid points

        # Inject NaN at vertex 0
        T_a[0] = np.nan

        result = compute_L2_unstructured(T_a, T_b, vertex_vols)

        # Valid volume excludes vertex 0
        valid_vol = vertex_vols[1:].sum()
        expected_L2 = 1.0 * np.sqrt(valid_vol)  # error = 1.0, integrated over valid vol

        assert not np.isnan(result["L2_abs"]), "L2_abs must not be NaN"
        assert result["L2_abs"] == pytest.approx(expected_L2, rel=1e-6)
        assert result["volume"] == pytest.approx(valid_vol, rel=1e-10)
