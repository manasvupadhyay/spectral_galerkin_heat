"""
fastHeatSolv — Spectral Heat Equation Solver for Additive Manufacturing.

Public sub-packages
-------------------
core        Configuration, parameters, and standalone runner.
interfaces  Abstract contracts (solver, factory, IO, laser path).
implementations  Concrete backends (CPU / GPU spectral solvers, factories, I/O).
utils       Reconstruction helpers, comparison tools, GCode parsing.
"""

__all__ = [
    "core",
    "interfaces",
    "implementations",
    "utils",
]
