"""Factory implementations for CPU and GPU simulation backends."""

from .cpu_factory import CPUSimulationFactory

# GPU factory requires CuPy — import lazily to avoid hard dependency
try:
    from .gpu_factory import GPUSimulationFactory
except ImportError:
    GPUSimulationFactory = None  # type: ignore[assignment,misc]

__all__ = [
    "CPUSimulationFactory",
    "GPUSimulationFactory",
]
