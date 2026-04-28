"""Factory implementations for CPU and GPU simulation backends.

**Internal Implementation Detail**

Preferably use the high-level runner (StandaloneHeatRunner) or create solvers 
directly via the solver classes instead of using factory patterns.
"""

from .base import SimulationFactory
from .cpu_factory import CPUSimulationFactory

# GPU factory requires CuPy — import lazily to avoid hard dependency
try:
    from .gpu_factory import GPUSimulationFactory
except ImportError:
    GPUSimulationFactory = None  # type: ignore[assignment,misc]

# Not exported: internal implementation details
__all__ = []
