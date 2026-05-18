"""fast_heat_solv — spectral heat solver for additive manufacturing.

Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique

Author: Théo Andrieux

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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
