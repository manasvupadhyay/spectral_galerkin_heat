"""File I/O and post-processing utilities for simulations.

**Public API:**
- **Data structures**: :class:`StructuredField`, :class:`UnstructuredField`, :class:`FieldData`
- **I/O manager**: :class:`LocalFSIOManager` (file-system implementation)
- **XDMF I/O**: :func:`load_xdmf`, :func:`write_structured_fields`, :func:`write_unstructured_fields`, :class:`XdmfBuilder`
- **Laser paths**: :class:`GCodeLaserPath` (for G-code based laser motion)
- **Simulation loading**: :func:`list_runs`, :func:`load_run`
- **Analysis**: :func:`compute_L2_structured`, :func:`compute_L2_unstructured`, :func:`compare`
- **Visualization**: :func:`generate_plots` (post-processing tool for creating cut-plane visualizations)
"""

from .spectral_fs_io import LocalFSIOManager
from .xdmf_io import (
    FieldData,
    StructuredField,
    UnstructuredField,
    XdmfBuilder,
    load_xdmf,
    write_structured_fields,
    write_unstructured_fields,
)
from .loader import SimulationResult, list_runs, load_run
from .gcode_path import GCodeLaserPath
from .compute_L2_error import compute_L2_structured, compute_L2_unstructured, compare

# Deferred: matplotlib import in cut_views is slow; only load on first use.
def __getattr__(name: str):
    if name == "generate_plots":
        from .cut_views import generate_plots
        globals()["generate_plots"] = generate_plots
        return generate_plots
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # I/O manager
    "LocalFSIOManager",
    # Data structures
    "FieldData",
    "StructuredField",
    "UnstructuredField",
    # XDMF I/O
    "XdmfBuilder",
    "load_xdmf",
    "write_structured_fields",
    "write_unstructured_fields",
    # Loading results
    "SimulationResult",
    "list_runs",
    "load_run",
    # Laser paths
    "GCodeLaserPath",
    # Analysis
    "compute_L2_structured",
    "compute_L2_unstructured",
    "compare",
    # Visualization
    "generate_plots",
]

