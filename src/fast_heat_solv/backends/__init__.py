"""Math backends: dispatch array operations and kernels to CPU or GPU.

A :class:`MathBackend` bundles an array module (NumPy or CuPy) with its
matching physics-kernel module. Inject one into
:class:`~fast_heat_solv.solvers.spectral.SpectralSolver` to run on CPU or GPU
from the same solver code. To run on a different library (PyTorch, JAX, …),
add a :class:`MathBackend` with a matching physics-kernel module, 
see the docs for more details.

Examples
--------
>>> from fast_heat_solv.backends import get_backend
>>> backend = get_backend("numpy")
>>> backend.name
'numpy'
"""

from ._registry import get_backend, register_backend
from .base import MathBackend

# Importing the module runs its ``@register_backend`` decorator.
from .numpy_backend import NumpyBackend


@register_backend("cupy")
def _make_cupy_backend() -> MathBackend:
    # Registered as a lazy factory: CuPy is an optional dependency, so its
    # module is imported only when a CuPy backend is actually requested.
    from .cupy_backend import CupyBackend

    return CupyBackend()


__all__ = [
    "MathBackend",
    "NumpyBackend",
    "get_backend",
    "register_backend",
]
