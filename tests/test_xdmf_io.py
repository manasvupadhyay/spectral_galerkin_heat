"""Tests for xdmf_io module."""
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import pytest

from fast_heat_solv.io_utils import (
    XdmfBuilder,
    StructuredField,
    load_xdmf,
    write_structured_fields,
)


class TestXdmfBuilder:
    """Test XdmfBuilder generates valid XDMF XML."""

    def test_structured_grid_xml_structure(self):
        builder = XdmfBuilder(version="2.0")
        builder.add_structured_grid(
            name="Test",
            dims=(4, 8, 16),
            h5_ref="test.h5",
            attributes={"temperature": "T"},
            time=1.5,
        )
        xml = builder.to_string()
        root = ET.fromstring(xml.split("\n", 2)[-1])  # Skip header lines

        assert root.tag == "Xdmf"
        assert root.get("Version") == "2.0"

        grid = root.find(".//Grid")
        assert grid.get("Name") == "Test"
        time_elem = grid.find("Time")
        assert time_elem is not None, "Expected a <Time> child element"
        assert time_elem.get("Value") == "1.5"

        topo = grid.find("Topology")
        assert topo.get("TopologyType") == "3DRectMesh"
        assert topo.get("Dimensions") == "4 8 16"

        geo = grid.find("Geometry")
        assert geo.get("GeometryType") == "VXVYVZ"

    def test_unstructured_grid_xml_structure(self):
        builder = XdmfBuilder(version="3.0")
        builder.add_unstructured_grid(
            name="Mesh",
            n_vertices=100,
            n_elements=50,
            h5_ref="mesh.h5",
            attributes={"error": "error"},
        )
        xml = builder.to_string()
        root = ET.fromstring(xml.split("\n", 2)[-1])

        grid = root.find(".//Grid")
        topo = grid.find("Topology")
        assert topo.get("TopologyType") == "Tetrahedron"
        assert topo.get("NumberOfElements") == "50"

        geo = grid.find("Geometry")
        assert geo.get("GeometryType") == "XYZ"


class TestWriteReadRoundtrip:
    """Test write_structured_fields + load_xdmf roundtrip."""

    def test_roundtrip_structured_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "test_field"

            x = np.linspace(0, 1, 10)
            y = np.linspace(0, 2, 15)
            z = np.linspace(0, 0.5, 5)
            field = np.random.rand(5, 15, 10).astype(np.float32)

            write_structured_fields(base, {"T": field}, x, y, z, time=2.0)

            loaded = load_xdmf(base.with_suffix(".xmf"))

            assert isinstance(loaded, StructuredField)
            np.testing.assert_allclose(loaded.x, x, rtol=1e-5)
            np.testing.assert_allclose(loaded.y, y, rtol=1e-5)
            np.testing.assert_allclose(loaded.z, z, rtol=1e-5)
            np.testing.assert_allclose(loaded.T, field, rtol=1e-5)
