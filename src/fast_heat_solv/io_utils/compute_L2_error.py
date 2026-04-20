"""
compute_L2_error.py
===================
Compute the L2 norm error between two XDMF solution files at their last
time step, and optionally output an ``error.xdmf`` / ``error.h5`` file
for ParaView visualisation.

Supported grid combinations
----------------------------
1. **Both structured** (``3DRectMesh`` / ``VXVYVZ``):
   If the grids match exactly (same dimensions AND same coordinate
   vectors up to tolerance), a **direct pointwise** comparison is done.
   Otherwise the finer grid is interpolated onto the coarser one using
   ``RegularGridInterpolator`` (trilinear).

2. **One structured + one unstructured** (``Tetrahedron`` / ``XYZ``):
   The **structured** field is interpolated at the **unstructured mesh
   vertices** using ``RegularGridInterpolator`` (fast trilinear).
   This avoids the slow Delaunay triangulation entirely.
   L2 integration uses vertex volumes computed from the tetrahedral
   connectivity (lumped mass approach).

3. **Both unstructured**: requires ``--resolution`` to create a common
   structured grid and uses ``LinearNDInterpolator`` for both fields.

Numerical integration
---------------------
- **Structured grids**: composite trapezoidal rule (second-order accurate)
- **Unstructured evaluation**: lumped mass integration using vertex
  volumes computed from the tetrahedral mesh connectivity

CLI
---
See ``scripts/compute_L2_error.py``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

# Import IO utilities from the main package
from fast_heat_solv.io_utils import (
    FieldData,
    StructuredField,
    UnstructuredField,
    load_xdmf,
    write_structured_fields,
    write_unstructured_fields,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Grid alignment check
# ---------------------------------------------------------------------------

def _grids_aligned(a: StructuredField, b: StructuredField, tol: float = 1e-10) -> bool:
    """Check whether two structured grids have identical coordinates."""
    if a.T.shape != b.T.shape:
        return False
    for ca, cb in [(a.x, b.x), (a.y, b.y), (a.z, b.z)]:
        if len(ca) != len(cb):
            return False
        if not np.allclose(ca, cb, atol=tol, rtol=tol):
            return False
    return True


# ---------------------------------------------------------------------------
#  Interpolation helpers
# ---------------------------------------------------------------------------

def _build_structured_interpolator(sf: StructuredField):
    """Return a ``RegularGridInterpolator`` for a StructuredField.

    The interpolator accepts points of shape (N, 3) where columns are
    (z, y, x) - matching the array dimension order - and returns
    interpolated temperature values.

    ``RegularGridInterpolator`` performs **trilinear** interpolation
    on a rectilinear grid.  It is exact at the original grid nodes and
    is O(N log M) for N query points on a grid with M nodes per axis
    (binary search along each axis).
    """
    from scipy.interpolate import RegularGridInterpolator

    # RegularGridInterpolator expects axes in the order of array dims,
    # i.e. (z, y, x) for T of shape (nz, ny, nx).
    return RegularGridInterpolator(
        (sf.z, sf.y, sf.x),
        sf.T,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def _clamp_query_points(query_pts: np.ndarray, sf: StructuredField) -> np.ndarray:
    """Clamp query points to structured grid bounds to handle floating-point precision.

    Points exactly at domain boundaries (Lx, Ly, Lz) can appear slightly outside
    due to floating-point rounding. This clamps them to the valid range.

    Parameters
    ----------
    query_pts : ndarray, shape (N, 3)
        Query points in (z, y, x) order (RegularGridInterpolator convention).
    sf : StructuredField
        The structured field defining the grid bounds.

    Returns
    -------
    ndarray, shape (N, 3)
        Clamped query points.
    """
    clamped = query_pts.copy()
    # Clamp z (column 0)
    clamped[:, 0] = np.clip(clamped[:, 0], sf.z.min(), sf.z.max())
    # Clamp y (column 1)
    clamped[:, 1] = np.clip(clamped[:, 1], sf.y.min(), sf.y.max())
    # Clamp x (column 2)
    clamped[:, 2] = np.clip(clamped[:, 2], sf.x.min(), sf.x.max())
    return clamped


def _build_unstructured_interpolator(uf: UnstructuredField):
    """Return a ``LinearNDInterpolator`` for an UnstructuredField.

    ``LinearNDInterpolator`` builds a Delaunay triangulation of the
    node cloud and performs **piecewise-linear** (barycentric)
    interpolation inside each simplex.

    For a P1 finite-element solution on linear tetrahedra this is
    **mathematically exact** - the FE shape functions are themselves
    piecewise linear, so the interpolation reproduces the discretised
    field with no approximation error.

    Complexity: O(N log N) triangulation + O(Q log N) per query batch.
    """
    from scipy.interpolate import LinearNDInterpolator

    return LinearNDInterpolator(uf.xyz, uf.T)


# ---------------------------------------------------------------------------
#  Common evaluation grid
# ---------------------------------------------------------------------------

def _bounding_box(fd: FieldData) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) arrays of shape (3,)."""
    if isinstance(fd, StructuredField):
        lo = np.array([fd.x.min(), fd.y.min(), fd.z.min()])
        hi = np.array([fd.x.max(), fd.y.max(), fd.z.max()])
    else:
        lo = fd.xyz.min(axis=0)
        hi = fd.xyz.max(axis=0)
    return lo, hi


def _make_common_grid(
    a: FieldData,
    b: FieldData,
    resolution: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a common structured evaluation grid (intersection of domains).

    If one of the inputs is structured, its axes are reused (clipped to
    the intersection) to avoid unnecessary re-interpolation.  Otherwise
    a uniform grid with *resolution* points per axis is created.
    """
    lo_a, hi_a = _bounding_box(a)
    lo_b, hi_b = _bounding_box(b)

    lo = np.maximum(lo_a, lo_b)
    hi = np.minimum(hi_a, hi_b)
    if np.any(hi <= lo):
        raise ValueError(
            f"Domains do not overlap.\n  A: {lo_a} -> {hi_a}\n  B: {lo_b} -> {hi_b}"
        )

    # Prefer reusing the structured axes of one input (fewer interp errors).
    # Pick the coarser grid to keep the evaluation tractable.
    def _clip_axis(coords, lo_val, hi_val):
        mask = (coords >= lo_val - 1e-12) & (coords <= hi_val + 1e-12)
        return coords[mask]

    if isinstance(a, StructuredField) and isinstance(b, StructuredField):
        # Use the coarser grid (fewer points on each axis)
        def _pick_axis(ax_a, ax_b, lo_v, hi_v):
            ca = _clip_axis(ax_a, lo_v, hi_v)
            cb = _clip_axis(ax_b, lo_v, hi_v)
            return ca if len(ca) <= len(cb) else cb

        x = _pick_axis(a.x, b.x, lo[0], hi[0])
        y = _pick_axis(a.y, b.y, lo[1], hi[1])
        z = _pick_axis(a.z, b.z, lo[2], hi[2])
        return x, y, z

    # One structured, one unstructured -> use the structured axes
    for fd in (a, b):
        if isinstance(fd, StructuredField):
            x = _clip_axis(fd.x, lo[0], hi[0])
            y = _clip_axis(fd.y, lo[1], hi[1])
            z = _clip_axis(fd.z, lo[2], hi[2])
            return x, y, z

    # Both unstructured -> create a uniform grid
    if resolution is None:
        raise ValueError("Resolution must be specified when both inputs are unstructured.")
    nx, ny, nz = resolution
    x = np.linspace(lo[0], hi[0], nx)
    y = np.linspace(lo[1], hi[1], ny)
    z = np.linspace(lo[2], hi[2], nz)
    return x, y, z


# ---------------------------------------------------------------------------
#  Evaluate a field on a structured grid
# ---------------------------------------------------------------------------

def _evaluate_on_grid(
    fd: FieldData,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    slab_max_points: int = 500_000,
) -> np.ndarray:
    """Evaluate *fd* at every node of a rectilinear (x, y, z) grid.

    Returns an array of shape ``(nz, ny, nx)``.

    For large grids the evaluation is done in z-slabs to limit peak
    memory usage (each slab holds at most ``slab_max_points`` query
    points).
    """
    nz, ny, nx = len(z), len(y), len(x)

    if isinstance(fd, StructuredField):
        # Check if the evaluation grid matches exactly
        if (
            fd.T.shape == (nz, ny, nx)
            and np.allclose(fd.x, x, atol=1e-12)
            and np.allclose(fd.y, y, atol=1e-12)
            and np.allclose(fd.z, z, atol=1e-12)
        ):
            logger.info("  Grid matches exactly - no interpolation needed")
            return fd.T.copy()

        logger.info("  Building RegularGridInterpolator...")
        interp = _build_structured_interpolator(fd)
        return _eval_structured_chunked(interp, x, y, z, slab_max_points)

    # Unstructured
    logger.info(f"  Building Delaunay triangulation ({len(fd.T)} vertices)...")
    interp = _build_unstructured_interpolator(fd)
    logger.info("  Triangulation complete")
    return _eval_unstructured_chunked(interp, x, y, z, slab_max_points)


def _eval_structured_chunked(
    interp,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    slab_max_points: int = 500_000,
) -> np.ndarray:
    """Evaluate a RegularGridInterpolator in z-slabs with progress logging.

    The interpolator expects query points in (z, y, x) order.
    """
    nz, ny, nx = len(z), len(y), len(x)
    slab_nz = max(1, slab_max_points // (ny * nx))
    result = np.empty((nz, ny, nx), dtype=np.float64)

    n_slabs = (nz + slab_nz - 1) // slab_nz
    logger.info(f"  RegularGrid interpolation: {n_slabs} slabs, grid={nx}x{ny}x{nz}")

    start_time = time.time()
    completed = 0

    for z0 in range(0, nz, slab_nz):
        z1 = min(z0 + slab_nz, nz)
        z_chunk = z[z0:z1]
        Zg, Yg, Xg = np.meshgrid(z_chunk, y, x, indexing="ij")
        pts = np.column_stack([Zg.ravel(), Yg.ravel(), Xg.ravel()])
        result[z0:z1] = interp(pts).reshape(z1 - z0, ny, nx)

        completed += 1
        elapsed = time.time() - start_time
        pct = 100 * completed / n_slabs
        if completed < n_slabs:
            eta = elapsed * (n_slabs - completed) / completed
            logger.info(f"    Progress: {completed}/{n_slabs} slabs ({pct:.1f}%) - ETA: {eta:.1f}s")

    elapsed = time.time() - start_time
    logger.info(f"  RegularGrid interpolation complete in {elapsed:.1f}s")

    return result


def _eval_unstructured_chunked(
    interp,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    slab_max_points: int = 500_000,
) -> np.ndarray:
    """Evaluate a LinearNDInterpolator in z-slabs with progress logging.

    The interpolator expects query points in (x, y, z) order.
    Result is returned in (nz, ny, nx) convention.
    """
    nz, ny, nx = len(z), len(y), len(x)
    slab_nz = max(1, slab_max_points // (ny * nx))
    result = np.empty((nz, ny, nx), dtype=np.float64)

    n_slabs = (nz + slab_nz - 1) // slab_nz
    logger.info(f"  LinearNDInterpolator: {n_slabs} slabs, grid={nx}x{ny}x{nz}")

    start_time = time.time()
    completed = 0

    for z0 in range(0, nz, slab_nz):
        z1 = min(z0 + slab_nz, nz)
        z_chunk = z[z0:z1]
        # meshgrid with (x, y, z_chunk): shape (nx, ny, chunk_nz)
        Xg, Yg, Zg = np.meshgrid(x, y, z_chunk, indexing="ij")
        pts = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
        vals = interp(pts).reshape(nx, ny, z1 - z0)
        # Transpose (nx, ny, chunk_nz) -> (chunk_nz, ny, nx)
        result[z0:z1] = vals.transpose(2, 1, 0)

        completed += 1
        elapsed = time.time() - start_time
        pct = 100 * completed / n_slabs
        if completed < n_slabs:
            eta = elapsed * (n_slabs - completed) / completed
            logger.info(f"    Progress: {completed}/{n_slabs} slabs ({pct:.1f}%) - ETA: {eta:.1f}s")

    elapsed = time.time() - start_time
    logger.info(f"  LinearNDInterpolator complete in {elapsed:.1f}s")

    return result


# ---------------------------------------------------------------------------
#  L2 norm via composite trapezoidal rule (structured grids)
# ---------------------------------------------------------------------------

def _trapezoidal_weights(coords: np.ndarray) -> np.ndarray:
    """Composite trapezoidal quadrature weights for non-uniform spacing."""
    w = np.zeros_like(coords)
    dx = np.diff(coords)
    w[:-1] += dx * 0.5
    w[1:] += dx * 0.5
    return w


def _compute_vertex_volumes(xyz: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """Compute lumped vertex volumes from tetrahedral connectivity.

    Each vertex receives 1/4 of the volume of each tetrahedron it belongs to.
    This is the standard "lumped mass" approach for P1 finite elements.

    Parameters
    ----------
    xyz : (N, 3) array
        Vertex coordinates
    connectivity : (M, 4) array
        Tetrahedral connectivity (vertex indices)

    Returns
    -------
    vertex_volumes : (N,) array
        Volume associated with each vertex
    """
    n_vertices = len(xyz)
    vertex_volumes = np.zeros(n_vertices, dtype=np.float64)

    # Vectorized tetrahedron volume computation
    # Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
    v0 = xyz[connectivity[:, 0]]
    v1 = xyz[connectivity[:, 1]]
    v2 = xyz[connectivity[:, 2]]
    v3 = xyz[connectivity[:, 3]]

    # Edge vectors
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0

    # Determinant via scalar triple product: e1 . (e2 x e3)
    cross = np.cross(e2, e3)
    det = np.einsum('ij,ij->i', e1, cross)
    tet_volumes = np.abs(det) / 6.0

    # Distribute 1/4 of each tet volume to its 4 vertices
    quarter_vol = tet_volumes / 4.0
    for i in range(4):
        np.add.at(vertex_volumes, connectivity[:, i], quarter_vol)

    return vertex_volumes


def compute_L2_structured(
    T_a: np.ndarray,
    T_b: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Dict[str, float]:
    """Compute error norms on a structured grid using trapezoidal rule."""
    err = T_a - T_b

    valid = np.isfinite(err)
    if not valid.all():
        raise ValueError("Error fields contain NaN values, indicating a severe issue (e.g. out of bounds interpolation).")

    wx = _trapezoidal_weights(x)
    wy = _trapezoidal_weights(y)
    wz = _trapezoidal_weights(z)
    W = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]

    volume = W.sum()
    L2_sq = np.sum(W * err**2)
    ref_sq = np.sum(W * T_a**2)

    L2_abs = np.sqrt(L2_sq)
    L2_rel = L2_abs / np.sqrt(ref_sq) if ref_sq > 0 else np.inf
    Linf = np.max(np.abs(err))

    n_nan = 0
    n_total = err.size
    pct_valid = 100.0

    return {
        "L2_abs": float(L2_abs),
        "L2_rel": float(L2_rel),
        "Linf": float(Linf),
        "volume": float(volume),
        "n_valid": int(valid.sum()),
        "n_total": int(n_total),
        "pct_valid": float(pct_valid),
        "n_nan": int(n_nan),
    }


# Backwards-compatible alias
compute_L2 = compute_L2_structured


def compute_L2_unstructured(
    T_a: np.ndarray,
    T_b: np.ndarray,
    vertex_volumes: np.ndarray,
) -> Dict[str, float]:
    """Compute error norms on unstructured vertices using lumped mass."""
    err = T_a - T_b

    valid = np.isfinite(err) & np.isfinite(T_a) & np.isfinite(T_b)
    if not valid.all():
        raise ValueError("Error fields contain NaN values, indicating a severe issue.")

    W = vertex_volumes

    volume = W.sum()
    L2_sq = np.sum(W * err**2)
    ref_sq = np.sum(W * T_a**2)

    L2_abs = np.sqrt(L2_sq)
    L2_rel = L2_abs / np.sqrt(ref_sq) if ref_sq > 0 else np.inf
    Linf = np.max(np.abs(err))

    n_nan = 0
    n_total = err.size
    pct_valid = 100.0

    return {
        "L2_abs": float(L2_abs),
        "L2_rel": float(L2_rel),
        "Linf": float(Linf),
        "volume": float(volume),
        "n_valid": int(valid.sum()),
        "n_total": int(n_total),
        "pct_valid": float(pct_valid),
        "n_nan": int(n_nan),
    }


# ---------------------------------------------------------------------------
#  Main comparison pipeline
# ---------------------------------------------------------------------------

def compare(
    path_a: Path,
    path_b: Path,
    attr_a: Optional[str] = None,
    attr_b: Optional[str] = None,
    output_base: Path = Path("research/error"),
    resolution: Optional[Tuple[int, int, int]] = None,
    write_error: bool = True,
) -> Dict[str, float]:
    """Full comparison pipeline: load -> align/interpolate -> L2 -> write.

    Parameters
    ----------
    path_a, path_b : str
        Paths to XDMF files. File A is treated as the reference.
    attr_a, attr_b : str, optional
        Attribute names to load from each file.
    output_base : str
        Base name for error output files.
    resolution : tuple, optional
        Grid resolution for both-unstructured case.
    write_error : bool
        If False, skip writing error XDMF/H5 files (faster).
    """
    print("=" * 70)
    print("  L2 Error Comparison")
    print("=" * 70)
    print(f"  File A : {path_a}  (reference)")
    print(f"  File B : {path_b}")
    print()

    # 1. Load both fields
    logger.info("Loading files...")
    field_a = load_xdmf(path_a, attr_name=attr_a)
    field_b = load_xdmf(path_b, attr_name=attr_b)

    print(f"  Grid A : {field_a.grid_type}", end="")
    if isinstance(field_a, StructuredField):
        print(f"  ({len(field_a.x)}x{len(field_a.y)}x{len(field_a.z)})", end="")
    else:
        print(f"  ({len(field_a.T)} vertices)", end="")
    print(f"  t = {field_a.time}")

    print(f"  Grid B : {field_b.grid_type}", end="")
    if isinstance(field_b, StructuredField):
        print(f"  ({len(field_b.x)}x{len(field_b.y)}x{len(field_b.z)})", end="")
    else:
        print(f"  ({len(field_b.T)} vertices)", end="")
    print(f"  t = {field_b.time}")
    print()

    # Determine comparison mode
    a_is_struct = isinstance(field_a, StructuredField)
    b_is_struct = isinstance(field_b, StructuredField)

    # -------------------------------------------------------------------------
    # Case 1: Both structured
    # -------------------------------------------------------------------------
    if a_is_struct and b_is_struct:
        aligned = _grids_aligned(field_a, field_b)

        if aligned:
            print("  Mode: ALIGNED - direct pointwise comparison")
            x, y, z = field_a.x, field_a.y, field_a.z
            T_a = field_a.T
            T_b = field_b.T
            interp_info = "None (grids match)"
        else:
            print("  Mode: STRUCTURED interpolation onto coarser grid")
            x, y, z = _make_common_grid(field_a, field_b)
            print(f"  Common grid: {len(x)}x{len(y)}x{len(z)}")
            print()

            logger.info("Interpolating field A...")
            T_a = _evaluate_on_grid(field_a, x, y, z)
            logger.info("Interpolating field B...")
            T_b = _evaluate_on_grid(field_b, x, y, z)
            interp_info = "RegularGridInterpolator (trilinear)"

        print()
        norms = compute_L2_structured(T_a, T_b, x, y, z)

        if write_error:
            out_path = Path(output_base).resolve()
            error = T_a - T_b
            write_structured_fields(
                out_path,
                fields={
                    "error": error,
                    "abs_error": np.abs(error),
                    "T_a": T_a,
                    "T_b": T_b,
                },
                x=x, y=y, z=z,
                time=field_a.time,
            )

    # -------------------------------------------------------------------------
    # Case 2: One structured + one unstructured (FAST PATH - no Delaunay!)
    # -------------------------------------------------------------------------
    elif a_is_struct != b_is_struct:
        # Identify which is which
        if a_is_struct:
            struct_field, unstruct_field = field_a, field_b
            struct_label, unstruct_label = "A", "B"
        else:
            struct_field, unstruct_field = field_b, field_a
            struct_label, unstruct_label = "B", "A"

        print(f"  Mode: HYBRID - interpolate structured ({struct_label}) at unstructured vertices ({unstruct_label})")
        print(f"         No Delaunay triangulation needed!")
        print()

        # Check connectivity is available
        if unstruct_field.connectivity is None:
            raise ValueError(
                f"Unstructured field {unstruct_label} has no connectivity. "
                "Cannot compute vertex volumes for L2 integration."
            )

        # Build interpolator from structured field (very fast)
        logger.info(f"Building RegularGridInterpolator from structured field...")
        interp = _build_structured_interpolator(struct_field)

        # Query at unstructured vertices
        xyz = unstruct_field.xyz
        n_pts = len(xyz)
        logger.info(f"Evaluating at {n_pts} unstructured vertices...")

        start_time = time.time()
        # RegularGridInterpolator expects (z, y, x) order
        query_pts = np.column_stack([xyz[:, 2], xyz[:, 1], xyz[:, 0]])
        # Clamp to grid bounds to handle floating-point precision at boundaries
        query_pts = _clamp_query_points(query_pts, struct_field)
        T_struct_at_unstruct = interp(query_pts)
        elapsed = time.time() - start_time
        logger.info(f"Interpolation complete in {elapsed:.2f}s")

        # Get unstructured T values directly
        T_unstruct = unstruct_field.T

        # Compute vertex volumes
        logger.info("Computing vertex volumes from connectivity...")
        vertex_volumes = _compute_vertex_volumes(xyz, unstruct_field.connectivity)
        total_vol = vertex_volumes.sum()
        logger.info(f"Total mesh volume: {total_vol:.6e} m^3")

        # Assign T_a and T_b based on which is reference
        if a_is_struct:
            T_a = T_struct_at_unstruct  # A (structured) evaluated at B's vertices
            T_b = T_unstruct            # B (unstructured) native values
        else:
            T_a = T_unstruct            # A (unstructured) native values
            T_b = T_struct_at_unstruct  # B (structured) evaluated at A's vertices

        print()
        norms = compute_L2_unstructured(T_a, T_b, vertex_volumes)
        interp_info = f"RegularGridInterpolator on {struct_label}, evaluated at {unstruct_label} vertices"

        if write_error:
            out_path = Path(output_base).resolve()
            error = T_a - T_b
            write_unstructured_fields(
                out_path,
                fields={
                    "error": error,
                    "abs_error": np.abs(error),
                    "T_a": T_a,
                    "T_b": T_b,
                },
                xyz=xyz,
                connectivity=unstruct_field.connectivity,
                time=unstruct_field.time,
            )

    # -------------------------------------------------------------------------
    # Case 3: Both unstructured (requires Delaunay - slow)
    # -------------------------------------------------------------------------
    else:
        if resolution is None:
            raise ValueError(
                "Both fields are unstructured. "
                "Use --resolution NX,NY,NZ to specify common grid."
            )

        print(f"  Mode: BOTH UNSTRUCTURED - common grid {resolution}")
        print("  WARNING: This requires Delaunay triangulation (slow)")
        print()

        x, y, z = _make_common_grid(field_a, field_b, resolution=resolution)
        print(f"  Common grid: {len(x)}x{len(y)}x{len(z)}")
        print()

        logger.info("Interpolating field A...")
        T_a = _evaluate_on_grid(field_a, x, y, z)
        logger.info("Interpolating field B...")
        T_b = _evaluate_on_grid(field_b, x, y, z)
        interp_info = "LinearNDInterpolator (Delaunay)"

        print()
        norms = compute_L2_structured(T_a, T_b, x, y, z)

        if write_error:
            out_path = Path(output_base).resolve()
            error = T_a - T_b
            write_structured_fields(
                out_path,
                fields={
                    "error": error,
                    "abs_error": np.abs(error),
                    "T_a": T_a,
                    "T_b": T_b,
                },
                x=x, y=y, z=z,
                time=field_a.time,
            )

    # Print results
    print("-" * 70)
    print(f"  L2 absolute error  : {norms['L2_abs']:.6e}")
    print(f"  L2 relative error  : {norms['L2_rel']:.6e}  ({norms['L2_rel']*100:.4f}%)")
    print(f"  L_inf (max |err|)  : {norms['Linf']:.6e}")
    print(f"  Integration volume : {norms['volume']:.6e} m^3")
    print(f"  Valid points       : {norms['n_valid']}/{norms['n_total']} ({norms['pct_valid']:.1f}%)")
    if norms["n_nan"] > 0:
        print(f"  WARNING: {norms['n_nan']} NaN points (outside domain)")
    print("-" * 70)

    if write_error:
        print()
        print(f"  Error written to: {output_base}.xdmf / .h5")

    print()
    print(f"  Method: {interp_info}")
    print("=" * 70)

    return norms
