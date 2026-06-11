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

from fast_heat_solv.io_utils.compute_L2_error import (
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
    """Build an UnstructuredField from deterministic scattered points.
    Includes corners to ensure the convex hull covers the full domain.
    """
    if n_pts < 8:
        raise ValueError("n_pts must be at least 8 to cover the corners")
    
    # Deterministic sequence (fractional part of irrational steps)
    i = np.arange(n_pts - 8)
    x = (i * 0.6180339887) % Lx
    y = (i * 0.7320508075) % Ly
    z = (i * 0.5560265774) % Lz
    
    corners = np.array([
        [0, 0, 0], [Lx, 0, 0], [0, Ly, 0], [Lx, Ly, 0],
        [0, 0, Lz], [Lx, 0, Lz], [0, Ly, Lz], [Lx, Ly, Lz]
    ])
    
    xyz = np.vstack([np.column_stack([x, y, z]), corners])
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
        assert T == pytest.approx(sf.T, rel=1e-12, abs=1e-15)

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
        # We use relative error checks. Note that we provide an absolute
        # tolerance fallback (abs=1e-5) for values extremely close to zero.
        assert T == pytest.approx(T_ref, rel=5e-3, abs=1e-5)

    def test_unstructured_interpolation(self):
        """Unstructured (scattered) field evaluated on a structured grid."""
        uf = _make_unstructured(10000)
        x = np.linspace(0.1, 0.9, 10)
        y = np.linspace(0.1, 0.9, 10)
        z = np.linspace(0.1, 0.9, 8)
        T = _evaluate_on_grid(uf, x, y, z)

        Zg, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")
        T_ref = _analytical(Xg, Yg, Zg)
        
        # rel covers the interior;
        # abs is the fallback for near-zero values 
        assert T == pytest.approx(T_ref, rel=0.01, abs=0.01)

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

