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
- **Backends**: :class:`~fast_heat_solv.backends.MathBackend`,
  :func:`~fast_heat_solv.backends.get_backend`
- **Solvers**: :class:`~fast_heat_solv.solvers.HeatSolver`,
  :class:`~fast_heat_solv.solvers.spectral.SpectralSolver`
- **I/O**: :class:`~fast_heat_solv.io_utils.LocalFSIOManager`,
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
from .io_utils import (
    LocalFSIOManager,
    load_xdmf,
    write_structured_fields,
    write_unstructured_fields,
)
from .backends import (
    MathBackend,
    NumpyBackend,
    get_backend,
)

# Solver and runner imports are deferred — they pull in numba/pyfftw which
# are expensive to compile. Use __getattr__ so `from fast_heat_solv import
# SpectralSolver` still works but only loads the solvers on first access.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "HeatSolver":           (".solvers.base",    "HeatSolver"),
    "SpectralSolver":       (".solvers.spectral", "SpectralSolver"),
    "StandaloneHeatRunner": (".runner",          "StandaloneHeatRunner"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, package=__name__)
        obj = getattr(mod, attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core
    "LaserState",
    "LaserPath",
    "SimulationContext",
    "NumParams",
    "MaterialParams",
    "GeomParams",
    "LaserParams",
    # Backends
    "MathBackend",
    "NumpyBackend",
    "get_backend",
    # Solvers
    "HeatSolver",
    "SpectralSolver",
    # I/O
    "LocalFSIOManager",
    "load_xdmf",
    "write_structured_fields",
    "write_unstructured_fields",
    # Runner
    "StandaloneHeatRunner",
]
