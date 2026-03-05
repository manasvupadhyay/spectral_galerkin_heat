"""Concrete solver implementations."""

from .spectral_cpu import SpectralSolverCPU

# GPU solver requires CuPy — import lazily to avoid hard dependency
try:
    from .spectral_gpu import SpectralSolverGPU
except ImportError:
    SpectralSolverGPU = None  # type: ignore[assignment,misc]

__all__ = [
    "SpectralSolverCPU",
    "SpectralSolverGPU",
]
