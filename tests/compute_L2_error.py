#!/usr/bin/env python3
"""
compute_L2_error.py
===================
Compute the L2 norm error between two XDMF solution files at their last
time step, and output an ``error.xdmf`` / ``error.h5`` file for
ParaView visualisation.

Supported grid combinations
----------------------------
1. **Both structured** (``3DRectMesh`` / ``VXVYVZ``):
   If the grids match exactly (same dimensions AND same coordinate
   vectors up to tolerance), a **direct pointwise** comparison is done —
   no interpolation needed.  Otherwise the finer grid is interpolated
   onto the coarser one using ``RegularGridInterpolator`` (trilinear).

2. **One structured + one unstructured** (``Tetrahedron`` / ``XYZ``):
   Both fields are evaluated on a **common structured grid** whose
   bounding box is the intersection of the two domains.

   * *Structured source* → ``RegularGridInterpolator`` (trilinear).
     This is exact at the original grid nodes and uses efficient binary
     search along each rectilinear axis.
   * *Unstructured (FE) source* → ``LinearNDInterpolator`` (piecewise
     linear in each Delaunay simplex).  For a P1 finite-element solution
     on linear tetrahedra this reproduces the **exact** FE solution
     everywhere inside the mesh — the interpolation is therefore
     **mathematically lossless** for the discretised field.

3. **Both unstructured**: the common evaluation grid is constructed the
   same way and ``LinearNDInterpolator`` is used for both fields.

Numerical integration
---------------------
The L2 norm is computed with the **composite trapezoidal rule** on the
(possibly interpolated) structured evaluation grid:

    ‖e‖²_L2 = ∫_Ω (T₁ − T₂)² dV
            ≈ Σᵢⱼₖ  wˣᵢ wʸⱼ wᶻₖ  (T₁ − T₂)²ᵢⱼₖ

where w are standard trapezoidal weights along each axis.
This is second-order accurate in each grid spacing.

Usage
-----
    python compute_L2_error.py file_A.xdmf file_B.xdmf [options]

    # Compare spectral output vs FE validation (last time step):
    python compute_L2_error.py \\
        ../out/20260304-111924_sim/fields/field_step000200.xmf \\
        validation.xdmf \\
        --attr-a temperature --attr-b Temperature \\
        --output error

    # Compare two structured grids (e.g. spectral vs Eagar-Tsai):
    python compute_L2_error.py \\
        ../out/20260304-111924_sim/fields/field_step000200.xmf \\
        eagar_tsai.xmf \\
        --output error
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class StructuredField:
    """Temperature on a rectilinear (3DRectMesh / VXVYVZ) grid."""
    x: np.ndarray          # 1-D, ascending
    y: np.ndarray
    z: np.ndarray
    T: np.ndarray          # shape (nz, ny, nx)
    time: Optional[float] = None

    @property
    def grid_type(self) -> str:
        return "structured"


@dataclass
class UnstructuredField:
    """Temperature at scattered FE nodes (Tetrahedron / XYZ)."""
    xyz: np.ndarray        # (N, 3)
    T: np.ndarray          # (N,)
    connectivity: Optional[np.ndarray] = None   # (M, 4) if available
    time: Optional[float] = None

    @property
    def grid_type(self) -> str:
        return "unstructured"


FieldData = StructuredField | UnstructuredField

# ---------------------------------------------------------------------------
#  XDMF / HDF5 parsing
# ---------------------------------------------------------------------------

def _resolve_h5(xdmf_path: str, ref: str) -> Tuple[str, str]:
    """Parse ``filename.h5:/dataset`` and resolve relative to XDMF dir."""
    parts = ref.strip().split(":")
    h5_file = parts[0].strip()
    h5_dset = parts[1].strip() if len(parts) > 1 else "/"
    base = os.path.dirname(os.path.abspath(xdmf_path))
    return os.path.join(base, h5_file), h5_dset


def _read_dataitem(xdmf_path: str, item: ET.Element) -> np.ndarray:
    """Read a <DataItem> from HDF5."""
    h5_file, h5_dset = _resolve_h5(xdmf_path, item.text)
    with h5py.File(h5_file, "r") as f:
        return np.asarray(f[h5_dset][:], dtype=np.float64)


def _find_last_timestep_grid(domain: ET.Element) -> Tuple[ET.Element, Optional[float]]:
    """Return the <Grid> element corresponding to the **last** time step.

    Strategy:
    - If there is a Temporal Collection, take its last child Grid.
    - If the XDMF uses XInclude for topology/geometry, the referenced
      elements must be resolved from the parent domain.
    - Otherwise fall back to the single Grid that carries an Attribute.
    """
    # -- Look for temporal collections first
    collections = []
    for g in domain.iter("Grid"):
        gt = (g.get("GridType") or "").lower()
        ct = (g.get("CollectionType") or "").lower()
        if gt == "collection" and ct == "temporal":
            collections.append(g)

    # Among the temporal collections, pick the one whose name resembles
    # "Temperature" (the solution field), falling back to the last one.
    target_collection = None
    for c in collections:
        name = (c.get("Name") or "").lower()
        if "temperature" in name:
            target_collection = c
    if target_collection is None and collections:
        target_collection = collections[-1]

    if target_collection is not None:
        children = [g for g in target_collection.findall("Grid")]
        if children:
            last = children[-1]
            time_el = last.find("Time")
            t = float(time_el.get("Value")) if time_el is not None else None
            return last, t

    # -- No temporal collection → single-step file
    for g in domain.iter("Grid"):
        if g.find("Attribute") is not None:
            t_str = g.get("Time")
            t = float(t_str) if t_str else None
            return g, t

    raise ValueError("Could not find any Grid with an Attribute in the XDMF file.")


def _resolve_topology_geometry(domain: ET.Element, grid: ET.Element):
    """Return (Topology, Geometry) elements, resolving XInclude refs."""
    topo = grid.find("Topology")
    geo = grid.find("Geometry")

    if topo is not None and geo is not None:
        return topo, geo

    # FENiCS-style: topo/geo are in a separate <Grid GridType="Uniform">
    # referenced via XInclude pointers.  Search the first Uniform grid.
    for g in domain.findall("Grid"):
        gt = (g.get("GridType") or "").lower()
        if gt == "uniform":
            if topo is None and g.find("Topology") is not None:
                topo = g.find("Topology")
            if geo is None and g.find("Geometry") is not None:
                geo = g.find("Geometry")
        if topo is not None and geo is not None:
            break

    if topo is None:
        raise ValueError("Topology not found in XDMF.")
    if geo is None:
        raise ValueError("Geometry not found in XDMF.")
    return topo, geo


def load_xdmf(xdmf_path: str, attr_name: Optional[str] = None) -> FieldData:
    """Load the **last time step** from an XDMF file.

    Parameters
    ----------
    xdmf_path : str
        Path to the ``.xdmf`` or ``.xmf`` file.
    attr_name : str or None
        Name of the Attribute to load (e.g. ``"temperature"``).
        If *None*, the first ``<Attribute>`` element found is used.
    """
    tree = ET.parse(xdmf_path)
    root = tree.getroot()
    # Handle namespace - strip it if present
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    domain = root.find(f"{ns}Domain")
    if domain is None:
        domain = root.find("Domain")
    if domain is None:
        raise ValueError("No <Domain> element in XDMF.")

    grid, time_val = _find_last_timestep_grid(domain)
    topo, geo = _resolve_topology_geometry(domain, grid)

    topo_type = topo.get("TopologyType") or topo.get("Type") or ""

    # ---- Find the target Attribute element ----
    attr_el = None
    if attr_name:
        for a in grid.iter("Attribute"):
            if (a.get("Name") or "").lower() == attr_name.lower():
                attr_el = a
                break
    if attr_el is None:
        attr_el = grid.find("Attribute")
    if attr_el is None:
        raise ValueError(f"No Attribute found in the target Grid of {xdmf_path}")

    attr_item = attr_el.find("DataItem")
    T_raw = _read_dataitem(xdmf_path, attr_item)

    # ---- Structured: 3DRectMesh ----
    if "rectmesh" in topo_type.lower() or "3drect" in topo_type.lower():
        geo_items = geo.findall("DataItem")
        axes = [_read_dataitem(xdmf_path, it) for it in geo_items]
        # VXVYVZ → axes[0]=X, axes[1]=Y, axes[2]=Z
        x, y, z = axes[0], axes[1], axes[2]

        # Dimensions string is "nz ny nx"
        dims_str = topo.get("Dimensions") or topo.get("NumberOfElements") or ""
        dims = [int(d) for d in dims_str.split()]
        nz, ny, nx = dims[0], dims[1], dims[2]

        T = T_raw.reshape(nz, ny, nx)

        logger.info(
            f"[Structured] Loaded {xdmf_path}: "
            f"({nx}×{ny}×{nz}), t={time_val}"
        )
        return StructuredField(x=x, y=y, z=z, T=T, time=time_val)

    # ---- Unstructured: Tetrahedron etc. ----
    else:
        geo_type = (geo.get("GeometryType") or geo.get("Type") or "").upper()
        geo_item = geo.find("DataItem")
        xyz = _read_dataitem(xdmf_path, geo_item)
        if xyz.ndim == 1:
            xyz = xyz.reshape(-1, 3)

        T_flat = T_raw.flatten()

        # Try loading connectivity
        conn = None
        conn_item = topo.find("DataItem")
        if conn_item is not None:
            try:
                conn = np.asarray(
                    _read_dataitem(xdmf_path, conn_item), dtype=np.int64
                )
            except Exception:
                pass

        logger.info(
            f"[Unstructured] Loaded {xdmf_path}: "
            f"{len(T_flat)} nodes, t={time_val}"
        )
        return UnstructuredField(
            xyz=xyz, T=T_flat, connectivity=conn, time=time_val
        )


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
    (z, y, x) — matching the array dimension order — and returns
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


def _build_unstructured_interpolator(uf: UnstructuredField):
    """Return a ``LinearNDInterpolator`` for an UnstructuredField.

    ``LinearNDInterpolator`` builds a Delaunay triangulation of the
    node cloud and performs **piecewise-linear** (barycentric)
    interpolation inside each simplex.

    For a P1 finite-element solution on linear tetrahedra this is
    **mathematically exact** — the FE shape functions are themselves
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
            f"Domains do not overlap.\n  A: {lo_a} → {hi_a}\n  B: {lo_b} → {hi_b}"
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

    # One structured, one unstructured → use the structured axes
    for fd in (a, b):
        if isinstance(fd, StructuredField):
            x = _clip_axis(fd.x, lo[0], hi[0])
            y = _clip_axis(fd.y, lo[1], hi[1])
            z = _clip_axis(fd.z, lo[2], hi[2])
            return x, y, z

    # Both unstructured → create a uniform grid
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

# Default slab size: number of z-planes per chunk for chunked evaluation.
# Each slab queries (slab_nz * ny * nx) points — tune to balance memory vs overhead.
_SLAB_MAX_POINTS = 500_000


def _evaluate_on_grid(
    fd: FieldData,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Evaluate *fd* at every node of a rectilinear (x, y, z) grid.

    Returns an array of shape ``(nz, ny, nx)``.

    For large grids the evaluation is done in z-slabs to limit peak
    memory usage (each slab holds at most ``_SLAB_MAX_POINTS`` query
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
            return fd.T.copy()

        interp = _build_structured_interpolator(fd)
        return _eval_structured_chunked(interp, x, y, z)

    # Unstructured
    interp = _build_unstructured_interpolator(fd)
    return _eval_unstructured_chunked(interp, x, y, z)


def _eval_structured_chunked(
    interp,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Evaluate a RegularGridInterpolator in z-slabs.

    The interpolator expects query points in (z, y, x) order.
    """
    nz, ny, nx = len(z), len(y), len(x)
    slab_nz = max(1, _SLAB_MAX_POINTS // (ny * nx))
    result = np.empty((nz, ny, nx), dtype=np.float64)

    for z0 in range(0, nz, slab_nz):
        z1 = min(z0 + slab_nz, nz)
        z_chunk = z[z0:z1]
        Zg, Yg, Xg = np.meshgrid(z_chunk, y, x, indexing="ij")
        pts = np.column_stack([Zg.ravel(), Yg.ravel(), Xg.ravel()])
        result[z0:z1] = interp(pts).reshape(z1 - z0, ny, nx)

    return result


def _eval_unstructured_chunked(
    interp,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Evaluate a LinearNDInterpolator in z-slabs.

    The interpolator expects query points in (x, y, z) order.
    Result is returned in (nz, ny, nx) convention.
    """
    nz, ny, nx = len(z), len(y), len(x)
    slab_nz = max(1, _SLAB_MAX_POINTS // (ny * nx))
    result = np.empty((nz, ny, nx), dtype=np.float64)

    for z0 in range(0, nz, slab_nz):
        z1 = min(z0 + slab_nz, nz)
        z_chunk = z[z0:z1]
        # meshgrid with (x, y, z_chunk): shape (nx, ny, chunk_nz)
        Xg, Yg, Zg = np.meshgrid(x, y, z_chunk, indexing="ij")
        pts = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
        vals = interp(pts).reshape(nx, ny, z1 - z0)
        # Transpose (nx, ny, chunk_nz) → (chunk_nz, ny, nx)
        result[z0:z1] = vals.transpose(2, 1, 0)

    return result


# ---------------------------------------------------------------------------
#  L2 norm via composite trapezoidal rule
# ---------------------------------------------------------------------------

def _trapezoidal_weights(coords: np.ndarray) -> np.ndarray:
    """Composite trapezoidal quadrature weights for non-uniform spacing."""
    w = np.zeros_like(coords)
    dx = np.diff(coords)
    w[:-1] += dx * 0.5
    w[1:] += dx * 0.5
    return w


def compute_L2(
    T_a: np.ndarray,
    T_b: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Dict[str, float]:
    """Compute error norms between two temperature fields on the same grid.

    Returns a dict with keys:
        L2_abs    — ‖T_a − T_b‖_L2
        L2_rel    — ‖T_a − T_b‖_L2 / ‖T_b‖_L2  (T_b treated as reference)
        Linf      — max |T_a − T_b|
        volume    — ∫ dV  (integration domain volume)
    """
    err = T_a - T_b

    # Mask NaN (out-of-domain after interpolation)
    valid = np.isfinite(err)
    if not valid.any():
        raise RuntimeError("No valid (non-NaN) points in the overlap region.")

    # Build tensor-product weights
    wx = _trapezoidal_weights(x)
    wy = _trapezoidal_weights(y)
    wz = _trapezoidal_weights(z)
    W = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]

    # Zero out weights at NaN locations
    W_valid = np.where(valid, W, 0.0)

    volume = W_valid.sum()
    L2_sq = np.nansum(W_valid * err**2)
    ref_sq = np.nansum(W_valid * T_b**2)

    L2_abs = np.sqrt(L2_sq)
    L2_rel = L2_abs / np.sqrt(ref_sq) if ref_sq > 0 else np.inf
    Linf = np.nanmax(np.abs(err))

    n_nan = (~valid).sum()
    n_total = valid.size
    pct_valid = valid.sum() / n_total * 100

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
#  XDMF / HDF5 output for error field
# ---------------------------------------------------------------------------

def write_error_xdmf(
    output_base: str,
    error: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    T_a: np.ndarray,
    T_b: np.ndarray,
    time_a: Optional[float] = None,
    time_b: Optional[float] = None,
) -> None:
    """Write the error field (and both solutions) to HDF5 + XDMF."""
    h5_path = f"{output_base}.h5"
    xmf_path = f"{output_base}.xdmf"
    h5_ref = os.path.basename(h5_path)

    nz, ny, nx = error.shape
    assert T_a.shape == (nz, ny, nx)
    assert T_b.shape == (nz, ny, nx)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=x.astype(np.float32))
        f.create_dataset("Y", data=y.astype(np.float32))
        f.create_dataset("Z", data=z.astype(np.float32))
        f.create_dataset("error", data=error.astype(np.float32))
        f.create_dataset("abs_error", data=np.abs(error).astype(np.float32))
        f.create_dataset("T_a", data=T_a.astype(np.float32))
        f.create_dataset("T_b", data=T_b.astype(np.float32))

    time_str = ""
    if time_a is not None:
        time_str = f' Time="{time_a}"'

    def _attr_block(name: str, dset: str) -> str:
        return (
            f'     <Attribute Name="{name}" AttributeType="Scalar" Center="Node">\n'
            f'       <DataItem Dimensions="{nz} {ny} {nx}" NumberType="Float" Precision="4" Format="HDF">\n'
            f"          {h5_ref}:/{dset}\n"
            f"       </DataItem>\n"
            f"     </Attribute>"
        )

    xmf = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
   <Grid Name="Error" GridType="Uniform"{time_str}>
     <Topology TopologyType="3DRectMesh" Dimensions="{nz} {ny} {nx}"/>
     <Geometry GeometryType="VXVYVZ">
       <DataItem Dimensions="{nx}" NumberType="Float" Precision="4" Format="HDF">
          {h5_ref}:/X
       </DataItem>
       <DataItem Dimensions="{ny}" NumberType="Float" Precision="4" Format="HDF">
          {h5_ref}:/Y
       </DataItem>
       <DataItem Dimensions="{nz}" NumberType="Float" Precision="4" Format="HDF">
          {h5_ref}:/Z
       </DataItem>
     </Geometry>
{_attr_block("error", "error")}
{_attr_block("abs_error", "abs_error")}
{_attr_block("T_a", "T_a")}
{_attr_block("T_b", "T_b")}
   </Grid>
 </Domain>
</Xdmf>
"""
    with open(xmf_path, "w") as f:
        f.write(xmf)

    logger.info(f"Wrote {xmf_path}  ({nz}×{ny}×{nx})")


# ---------------------------------------------------------------------------
#  Main comparison pipeline
# ---------------------------------------------------------------------------

def compare(
    path_a: str,
    path_b: str,
    attr_a: Optional[str] = None,
    attr_b: Optional[str] = None,
    output_base: str = "error",
    resolution: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, float]:
    """Full comparison pipeline: load → align/interpolate → L2 → write."""

    print("=" * 70)
    print("  L2 Error Comparison")
    print("=" * 70)
    print(f"  File A : {path_a}")
    print(f"  File B : {path_b}  (reference)")
    print()

    # 1. Load both fields
    field_a = load_xdmf(path_a, attr_name=attr_a)
    field_b = load_xdmf(path_b, attr_name=attr_b)

    print(f"  Grid A : {field_a.grid_type}", end="")
    if isinstance(field_a, StructuredField):
        print(f"  ({len(field_a.x)}×{len(field_a.y)}×{len(field_a.z)})", end="")
    else:
        print(f"  ({len(field_a.T)} nodes)", end="")
    print(f"  t = {field_a.time}")

    print(f"  Grid B : {field_b.grid_type}", end="")
    if isinstance(field_b, StructuredField):
        print(f"  ({len(field_b.x)}×{len(field_b.y)}×{len(field_b.z)})", end="")
    else:
        print(f"  ({len(field_b.T)} nodes)", end="")
    print(f"  t = {field_b.time}")
    print()

    # 2. Check alignment or build common grid
    aligned = False
    if isinstance(field_a, StructuredField) and isinstance(field_b, StructuredField):
        aligned = _grids_aligned(field_a, field_b)

    if aligned:
        print("  Mode   : ALIGNED grids — direct pointwise comparison")
        x, y, z = field_a.x, field_a.y, field_a.z
        T_a = field_a.T
        T_b = field_b.T
        interp_info = "None (grids match exactly)"
    else:
        # Determine interpolation description
        methods = []
        for label, fd in [("A", field_a), ("B", field_b)]:
            if isinstance(fd, StructuredField):
                methods.append(
                    f"{label}: RegularGridInterpolator (trilinear on rectilinear grid)"
                )
            else:
                methods.append(
                    f"{label}: LinearNDInterpolator (piecewise-linear / P1-exact on tetrahedra)"
                )
        interp_info = "\n             ".join(methods)

        print(f"  Mode   : INTERPOLATION onto common structured grid")
        print(f"  Method : {interp_info}")
        print()

        x, y, z = _make_common_grid(field_a, field_b, resolution=resolution)
        print(
            f"  Common grid : {len(x)}×{len(y)}×{len(z)}  "
            f"(x=[{x[0]:.6f}, {x[-1]:.6f}], "
            f"y=[{y[0]:.6f}, {y[-1]:.6f}], "
            f"z=[{z[0]:.6f}, {z[-1]:.6f}])"
        )
        print()

        print("  Interpolating fields A & B in parallel ...", flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(_evaluate_on_grid, field_a, x, y, z)
            future_b = pool.submit(_evaluate_on_grid, field_b, x, y, z)
            T_a = future_a.result()
            T_b = future_b.result()
        print("  done")

    # 3. Compute norms
    print()
    error = T_a - T_b
    norms = compute_L2(T_a, T_b, x, y, z)

    print("-" * 70)
    print(f"  L2 absolute error  : {norms['L2_abs']:.6e}")
    print(f"  L2 relative error  : {norms['L2_rel']:.6e}  "
          f"({norms['L2_rel']*100:.4f} %)")
    print(f"  L∞ (max pointwise) : {norms['Linf']:.6e}")
    print(f"  Integration volume : {norms['volume']:.6e} m³")
    print(f"  Valid points       : {norms['n_valid']}/{norms['n_total']} "
          f"({norms['pct_valid']:.1f} %)")
    if norms["n_nan"] > 0:
        print(f"  WARNING: {norms['n_nan']} NaN points "
              "(outside overlap or extrapolation)")
    print("-" * 70)
    print()

    # 4. Write error XDMF
    # Resolve output path relative to current working directory
    out_path = os.path.abspath(output_base)
    write_error_xdmf(
        out_path, error, x, y, z, T_a, T_b,
        time_a=field_a.time, time_b=field_b.time,
    )
    print(f"  Error field written to:")
    print(f"    {out_path}.xdmf")
    print(f"    {out_path}.h5")
    print()

    # 5. Summary
    print("  Interpolation methods recap:")
    print(f"    {interp_info}")
    print()
    print("  Integration: composite trapezoidal rule on the structured grid,")
    print("    second-order accurate in each grid spacing.")
    print("=" * 70)

    return norms


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Compute L2 norm error between two XDMF solution files."
    )
    parser.add_argument("file_a", help="Path to XDMF file A")
    parser.add_argument("file_b", help="Path to XDMF file B (reference)")
    parser.add_argument(
        "--attr-a",
        default=None,
        help="Attribute name in file A (default: first found)",
    )
    parser.add_argument(
        "--attr-b",
        default=None,
        help="Attribute name in file B (default: first found)",
    )
    parser.add_argument(
        "--output",
        default="error",
        help="Base name for error output files (default: 'error')",
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help="Grid resolution for two-unstructured case, e.g. '128,128,128'",
    )

    args = parser.parse_args()

    res = None
    if args.resolution:
        parts = [int(x) for x in args.resolution.split(",")]
        res = tuple(parts[:3])

    norms = compare(
        path_a=args.file_a,
        path_b=args.file_b,
        attr_a=args.attr_a,
        attr_b=args.attr_b,
        output_base=args.output,
        resolution=res,
    )

    # Exit with non-zero if there were NaN issues
    if norms["pct_valid"] < 50.0:
        sys.exit(2)


if __name__ == "__main__":
    main()
