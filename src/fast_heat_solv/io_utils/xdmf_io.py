"""XDMF file loading and writing utilities.

Provides functions to read simulation results from XDMF/HDF5 format
and write error fields for visualization in ParaView.

Uses xml.etree.ElementTree for proper XML generation instead of
hardcoded string templates.
"""

# Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique
#
# Author: Théo Andrieux
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

__author__ = "Théo Andrieux"
__copyright__ = "Copyright 2026, LMS, École Polytechnique"

import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------


@dataclass
class StructuredField:
    """Temperature on a rectilinear (3DRectMesh / VXVYVZ) grid."""

    x: np.ndarray  # 1-D, ascending
    y: np.ndarray
    z: np.ndarray
    T: np.ndarray  # shape (nz, ny, nx)
    time: Optional[float] = None

    @property
    def grid_type(self) -> str:
        return "structured"


@dataclass
class UnstructuredField:
    """Temperature at scattered FE nodes (Tetrahedron / XYZ)."""

    xyz: np.ndarray  # (N, 3)
    T: np.ndarray  # (N,)
    connectivity: Optional[np.ndarray] = None  # (M, 4) if available
    time: Optional[float] = None

    @property
    def grid_type(self) -> str:
        return "unstructured"


FieldData = Union[StructuredField, UnstructuredField]


# ---------------------------------------------------------------------------
#  XDMF Builder using ElementTree
# ---------------------------------------------------------------------------


class XdmfBuilder:
    """Builder for XDMF XML documents using xml.etree.ElementTree.

    This class provides a clean API for constructing XDMF files for both
    structured (3DRectMesh) and unstructured (Tetrahedron) grids.
    """

    def __init__(self, version: str = "2.0"):
        """Initialize an XDMF document.

        Parameters
        ----------
        version : str
            XDMF version (default "2.0", use "3.0" for newer features)
        """
        self.root = ET.Element("Xdmf", Version=version)
        self.domain = ET.SubElement(self.root, "Domain")

    def add_structured_grid(
        self,
        name: str,
        dims: Tuple[int, int, int],
        h5_ref: str,
        coord_datasets: Tuple[str, str, str] = ("X", "Y", "Z"),
        attributes: Optional[Dict[str, str]] = None,
        time: Optional[float] = None,
        step: Optional[int] = None,
        precision: int = 4,
        parent: Optional[ET.Element] = None,
    ) -> ET.Element:
        """Add a structured 3DRectMesh grid to the XDMF document.

        Parameters
        ----------
        name : str
            Name of the grid (e.g., "Error", "Mesh")
        dims : tuple of int
            Grid dimensions in (nz, ny, nx) order
        h5_ref : str
            HDF5 filename reference (basename only)
        coord_datasets : tuple of str
            Dataset names for X, Y, Z coordinates in HDF5
        attributes : dict, optional
            Mapping of attribute names to HDF5 dataset names
        time : float, optional
            Time value for this grid
        step : int, optional
            Step number for this grid
        precision : int
            Float precision (4 for float32, 8 for float64)

        Returns
        -------
        ET.Element
            The created Grid element
        """
        nz, ny, nx = dims

        # Create Grid element with optional time/step
        grid_attribs = {"Name": name, "GridType": "Uniform"}
            
        p = parent if parent is not None else self.domain
        grid = ET.SubElement(p, "Grid", **grid_attribs)
        
        if time is not None:
            ET.SubElement(grid, "Time", Value=str(time))

        # Topology
        ET.SubElement(
            grid,
            "Topology",
            TopologyType="3DRectMesh",
            Dimensions=f"{nz} {ny} {nx}",
        )

        # Geometry (VXVYVZ)
        geometry = ET.SubElement(grid, "Geometry", GeometryType="VXVYVZ")

        for coord_name, size in zip(coord_datasets, [nx, ny, nz]):
            self._add_dataitem(
                geometry,
                dimensions=str(size),
                h5_ref=h5_ref,
                dataset=coord_name,
                precision=precision,
            )

        # Attributes
        if attributes:
            for attr_name, dataset_name in attributes.items():
                self._add_attribute(
                    grid,
                    name=attr_name,
                    dimensions=f"{nz} {ny} {nx}",
                    h5_ref=h5_ref,
                    dataset=dataset_name,
                    precision=precision,
                )

        return grid

    def add_unstructured_grid(
        self,
        name: str,
        n_vertices: int,
        n_elements: int,
        h5_ref: str,
        topology_dataset: str = "topology",
        geometry_dataset: str = "geometry",
        topology_type: str = "Tetrahedron",
        nodes_per_element: int = 4,
        attributes: Optional[Dict[str, str]] = None,
        time: Optional[float] = None,
        geometry_precision: int = 8,
        attribute_precision: int = 4,
    ) -> ET.Element:
        """Add an unstructured grid to the XDMF document.

        Parameters
        ----------
        name : str
            Name of the grid
        n_vertices : int
            Number of vertices in the mesh
        n_elements : int
            Number of elements (e.g., tetrahedra)
        h5_ref : str
            HDF5 filename reference (basename only)
        topology_dataset : str
            HDF5 dataset name for connectivity
        geometry_dataset : str
            HDF5 dataset name for vertex coordinates
        topology_type : str
            Element type (e.g., "Tetrahedron", "Triangle")
        nodes_per_element : int
            Nodes per element (e.g., 4 for tetrahedra)
        attributes : dict, optional
            Mapping of attribute names to HDF5 dataset names
        time : float, optional
            Time value for this grid
        geometry_precision : int
            Float precision for geometry (default 8 = float64)
        attribute_precision : int
            Float precision for attributes (default 4 = float32)

        Returns
        -------
        ET.Element
            The created Grid element
        """
        # Create Grid element
        grid_attribs = {"Name": name, "GridType": "Uniform"}

        grid = ET.SubElement(self.domain, "Grid", **grid_attribs)
        
        if time is not None:
            ET.SubElement(grid, "Time", Value=str(time))

        # Topology
        topology = ET.SubElement(
            grid,
            "Topology",
            TopologyType=topology_type,
            NumberOfElements=str(n_elements),
        )
        self._add_dataitem(
            topology,
            dimensions=f"{n_elements} {nodes_per_element}",
            h5_ref=h5_ref,
            dataset=topology_dataset,
            number_type="Int",
            precision=None,  # Int doesn't need precision
        )

        # Geometry (XYZ)
        geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
        self._add_dataitem(
            geometry,
            dimensions=f"{n_vertices} 3",
            h5_ref=h5_ref,
            dataset=geometry_dataset,
            precision=geometry_precision,
        )

        # Attributes
        if attributes:
            for attr_name, dataset_name in attributes.items():
                self._add_attribute(
                    grid,
                    name=attr_name,
                    dimensions=str(n_vertices),
                    h5_ref=h5_ref,
                    dataset=dataset_name,
                    precision=attribute_precision,
                )

        return grid

    def _add_dataitem(
        self,
        parent: ET.Element,
        dimensions: str,
        h5_ref: str,
        dataset: str,
        number_type: str = "Float",
        precision: Optional[int] = 4,
    ) -> ET.Element:
        """Add a DataItem element pointing to an HDF5 dataset."""
        attribs = {
            "Dimensions": dimensions,
            "NumberType": number_type,
            "Format": "HDF",
        }
        if precision is not None:
            attribs["Precision"] = str(precision)

        dataitem = ET.SubElement(parent, "DataItem", **attribs)
        dataitem.text = f"{h5_ref}:/{dataset}"
        return dataitem

    def _add_attribute(
        self,
        grid: ET.Element,
        name: str,
        dimensions: str,
        h5_ref: str,
        dataset: str,
        precision: int = 4,
        attr_type: str = "Scalar",
        center: str = "Node",
    ) -> ET.Element:
        """Add an Attribute element to a grid."""
        attribute = ET.SubElement(
            grid,
            "Attribute",
            Name=name,
            AttributeType=attr_type,
            Center=center,
        )
        self._add_dataitem(
            attribute,
            dimensions=dimensions,
            h5_ref=h5_ref,
            dataset=dataset,
            precision=precision,
        )
        return attribute

    def to_string(self, indent: bool = True) -> str:
        """Convert the XDMF document to a string.

        Parameters
        ----------
        indent : bool
            Whether to pretty-print with indentation

        Returns
        -------
        str
            The XDMF XML document as a string
        """
        if indent:
            ET.indent(self.root, space="  ")

        xml_str = ET.tostring(self.root, encoding="unicode")
        # Add XML declaration and DOCTYPE
        header = '<?xml version="1.0" ?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n'
        return header + xml_str

    def write(self, filepath: Path) -> None:
        """Write the XDMF document to a file.

        Parameters
        ----------
        filepath : Path
            Path to the output file
        """
        with open(filepath, "w") as f:
            f.write(self.to_string())


# ---------------------------------------------------------------------------
#  XDMF / HDF5 parsing
# ---------------------------------------------------------------------------


def _resolve_h5(xdmf_path: Path, ref: str) -> Tuple[Path, str]:
    """Parse ``filename.h5:/dataset`` and resolve relative to XDMF dir."""
    parts = ref.strip().split(":")
    h5_file = parts[0].strip()
    h5_dset = parts[1].strip() if len(parts) > 1 else "/"
    base = xdmf_path.resolve().parent
    return base / h5_file, h5_dset


def _read_dataitem(xdmf_path: Path, item: ET.Element) -> np.ndarray:
    """Read a <DataItem> from HDF5."""
    h5_file, h5_dset = _resolve_h5(xdmf_path, item.text)
    with h5py.File(h5_file, "r") as f:
        return np.asarray(f[h5_dset][:], dtype=np.float64)


def _find_last_timestep_grid(
    domain: ET.Element, attr_name: Optional[str] = None
) -> Tuple[ET.Element, Optional[float]]:
    """Return the <Grid> element corresponding to the **last** time step.

    Parameters
    ----------
    domain : ET.Element
        The <Domain> element from the XDMF file.
    attr_name : str or None
        Name of the Attribute to search for when selecting among temporal
        collections. If provided, prefers collections whose name contains
        this string (case-insensitive). Otherwise falls back to the last
        temporal collection.

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

    # Among the temporal collections, pick the one whose name matches
    # attr_name (if provided), falling back to the last one.
    target_collection = None
    if attr_name:
        for c in collections:
            name = (c.get("Name") or "").lower()
            if attr_name.lower() in name:
                target_collection = c
                break
    if target_collection is None and collections:
        target_collection = collections[-1]

    if target_collection is not None:
        children = [g for g in target_collection.findall("Grid")]
        if children:
            last = children[-1]
            time_el = last.find("Time")
            t = float(time_el.get("Value")) if time_el is not None else None
            return last, t

    # -- No temporal collection -> single-step file
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


def load_xdmf(xdmf_path: Path, attr_name: Optional[str] = None) -> FieldData:
    """Load the **last time step** from an XDMF file.

    Parameters
    ----------
    xdmf_path : Path
        Path to the ``.xdmf`` or ``.xmf`` file.
    attr_name : str or None
        Name of the Attribute to load (e.g. ``"temperature"``).
        If *None*, the first ``<Attribute>`` element found is used.
        Also used to select among multiple temporal collections.
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

    grid, time_val = _find_last_timestep_grid(domain, attr_name=attr_name)
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
        # VXVYVZ -> axes[0]=X, axes[1]=Y, axes[2]=Z
        x, y, z = axes[0], axes[1], axes[2]

        # Dimensions string is "nz ny nx"
        dims_str = topo.get("Dimensions") or topo.get("NumberOfElements") or ""
        dims = [int(d) for d in dims_str.split()]
        nz, ny, nx = dims[0], dims[1], dims[2]

        T = T_raw.reshape(nz, ny, nx)

        logger.info(
            f"[Structured] Loaded {xdmf_path}: " f"({nx}x{ny}x{nz}), t={time_val}"
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
                conn = np.asarray(_read_dataitem(xdmf_path, conn_item), dtype=np.int64)
            except Exception:
                pass

        logger.info(
            f"[Unstructured] Loaded {xdmf_path}: " f"{len(T_flat)} nodes, t={time_val}"
        )
        return UnstructuredField(xyz=xyz, T=T_flat, connectivity=conn, time=time_val)


# ---------------------------------------------------------------------------
#  XDMF / HDF5 output (using XdmfBuilder)
# ---------------------------------------------------------------------------


def write_structured_fields(
    output_base: Path,
    fields: Dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    time: Optional[float] = None,
    step: Optional[int] = None,
) -> None:
    """Write multiple structured fields to HDF5 + XDMF.

    Parameters
    ----------
    output_base : Path
        Base path for output files (without extension)
    fields : Dict[str, np.ndarray]
        Mapping of field names to 3D arrays of shape (nz, ny, nx)
    x, y, z : np.ndarray
        1D coordinate arrays
    time : float, optional
        Time value
    step : int, optional
        Step number
    """
    if not fields:
        raise ValueError("fields dict cannot be empty")

    h5_path = output_base.with_suffix(".h5")
    xmf_path = output_base.with_suffix(".xmf")
    h5_ref = h5_path.name

    # Get dimensions from first field
    first_field = next(iter(fields.values()))
    nz, ny, nx = first_field.shape

    # Write HDF5 data
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=np.asarray(x))
        f.create_dataset("Y", data=np.asarray(y))
        f.create_dataset("Z", data=np.asarray(z))
        for name, arr in fields.items():
            dset = f.create_dataset(name, data=arr)
            if time is not None:
                dset.attrs["time"] = time
            if step is not None:
                dset.attrs["step"] = step

    # Build XDMF using ElementTree
    builder = XdmfBuilder(version="2.0")
    builder.add_structured_grid(
        name="Mesh",
        dims=(nz, ny, nx),
        h5_ref=h5_ref,
        attributes={name: name for name in fields},
        time=time,
        step=step,
    )
    builder.write(xmf_path)

    logger.info(f"Wrote {xmf_path}  ({nx}x{ny}x{nz}, {len(fields)} fields)")


def write_unstructured_fields(
    output_base: Path,
    fields: Dict[str, np.ndarray],
    xyz: np.ndarray,
    connectivity: np.ndarray,
    time: Optional[float] = None,
) -> None:
    """Write multiple fields on an unstructured (tetrahedral) mesh to HDF5 + XDMF.

    Parameters
    ----------
    output_base : Path
        Base path for output files (without extension)
    fields : Dict[str, np.ndarray]
        Mapping of field names to 1D arrays of shape (n_vertices,)
    xyz : np.ndarray
        Vertex coordinates of shape (n_vertices, 3)
    connectivity : np.ndarray
        Element connectivity of shape (n_elements, 4) for tetrahedra
    time : float, optional
        Time value
    """
    if not fields:
        raise ValueError("fields dict cannot be empty")

    h5_path = output_base.with_suffix(".h5")
    xmf_path = output_base.with_suffix(".xdmf")
    h5_ref = h5_path.name

    n_vertices = len(xyz)
    n_tets = len(connectivity)

    # Write HDF5 data
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("geometry", data=xyz.astype(np.float64))
        f.create_dataset("topology", data=connectivity.astype(np.int64))
        for name, arr in fields.items():
            f.create_dataset(name, data=arr.astype(np.float32))

    # Build XDMF using ElementTree
    builder = XdmfBuilder(version="3.0")
    builder.add_unstructured_grid(
        name="Mesh",
        n_vertices=n_vertices,
        n_elements=n_tets,
        h5_ref=h5_ref,
        attributes={name: name for name in fields},
        time=time,
    )
    builder.write(xmf_path)

    logger.info(f"Wrote {xmf_path}  ({n_vertices} vertices, {n_tets} tetrahedra, {len(fields)} fields)")
