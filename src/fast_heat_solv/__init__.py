"""fast_heat_solv — spectral heat solver for additive manufacturing.

This package provides a fast, spectrally-accurate heat equation solver
optimized for additive manufacturing (3D printing) simulations. It supports
both CPU and GPU backends and handles complex physical phenomena including
phase change and evaporation.

**Public API:**

- **Core interfaces**: :class:`~fast_heat_solv.core.parameters.SimulationContext`,
  :class:`~fast_heat_solv.core.laser.LaserPath`, :class:`~fast_heat_solv.core.laser.LaserState`
- **Solvers**: :class:`~fast_heat_solv.solvers.HeatSolver`,
  :class:`~fast_heat_solv.solvers.SpectralSolverCPU`, :class:`~fast_heat_solv.solvers.SpectralSolverGPU`
- **I/O**: :class:`~fast_heat_solv.io_utils.IOManager`,
  :func:`~fast_heat_solv.io_utils.load_xdmf`, :func:`~fast_heat_solv.io_utils.write_structured_fields`
- **Runner**: :class:`~fast_heat_solv.runner.StandaloneHeatRunner`
"""

__version__ = "0.1.0"

from .core import (
    LaserState,
    LaserPath,
    SimulationContext,
    NumParams,
    MaterialParams,
    GeomParams,
    LaserParams,
)
from .solvers import HeatSolver, SpectralSolverCPU, SpectralSolverGPU
from .io_utils import (
    IOManager,
    LocalFSIOManager,
    load_xdmf,
    write_structured_fields,
    write_unstructured_fields,
)
from .runner import StandaloneHeatRunner

__all__ = [
    # Core
    "LaserState",
    "LaserPath",
    "SimulationContext",
    "NumParams",
    "MaterialParams",
    "GeomParams",
    "LaserParams",
    # Solvers
    "HeatSolver",
    "SpectralSolverCPU",
    "SpectralSolverGPU",
    # I/O
    "IOManager",
    "LocalFSIOManager",
    "load_xdmf",
    "write_structured_fields",
    "write_unstructured_fields",
    # Runner
    "StandaloneHeatRunner",
]
