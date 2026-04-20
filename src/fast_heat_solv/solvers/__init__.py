"""Concrete solver implementations."""

from .base import HeatSolver
from .spectral_cpu import SpectralSolverCPU

# GPU solver requires CuPy — import lazily to avoid hard dependency
try:
    from .spectral_gpu import SpectralSolverGPU
except ImportError:
    SpectralSolverGPU = None  # type: ignore[assignment,misc]

__all__ = [
    "HeatSolver",
    "SpectralSolverCPU",
    "SpectralSolverGPU",
]
