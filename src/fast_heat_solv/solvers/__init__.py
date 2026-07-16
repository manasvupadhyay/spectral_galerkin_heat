"""Concrete solver implementations."""

from .base import HeatSolver
from .spectral import SpectralSolver

__all__ = [
    "HeatSolver",
    "SpectralSolver",
    "build_solver",
]


def build_solver(context) -> HeatSolver:
    """Select and construct the heat solver for *context* from its
    ``(method, backend)``.

    Backends are imported lazily to keep CuPy/Numba off the import path until
    needed.
    """
    if context.method != "spectral":
        # Extension point: other methods (e.g. FEM) would dispatch here.
        raise ValueError(f"Unknown simulation method: {context.method!r}")

    match context.backend:
        case "cpu":
            from fast_heat_solv.backends import NumpyBackend
            return SpectralSolver(backend=NumpyBackend())
        case "gpu":
            from fast_heat_solv.backends import get_backend
            return SpectralSolver(backend=get_backend("cupy"))
        case "cpu_linear":
            from .spectral_cpu_linear import SpectralSolverCPULinear
            return SpectralSolverCPULinear()
        case _:
            raise ValueError(
                f"Unknown backend: {context.backend!r}. "
                "Choose 'cpu', 'gpu', or 'cpu_linear'."
            )
