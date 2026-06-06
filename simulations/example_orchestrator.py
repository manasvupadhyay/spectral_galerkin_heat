#!/usr/bin/env python
"""
example_orchestrator.py – Demonstrates using fastHeatSolv as a library.

This script shows how an external "master" orchestrator can drive the heat solver frame-by-frame
without any disk I/O or IOManager involvement.

Usage:
    python example_orchestrator.py
"""

import sys
import os

# Ensure the package root is importable when running the script directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))); sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fast_heat_solv.core.parameters import SimulationContext

# ---------------------------------------------------------------------------
# 1. Define configuration as a plain Python dictionary
#    (An orchestrator would typically parse its own master YAML and extract
#     the heat-solver section into a dict like this one.)
# ---------------------------------------------------------------------------
config = {
    "simulation": {
        "method": "spectral",
        "backend": "cpu",
        "duration": 6e-5,      # very short run for demo
        "dt": 6e-6,
        "update_interval": 1,
    },
    "domain": {
        "size": [0.01, 0.005, 0.0025],   # Lx, Ly, Lz  [m]
        "mesh": [64, 32, 16],             # coarse mesh for speed
    },
    "material": {
        "name": "316L",
        "rho": 7850.0,
        "k": 15.0,
        "Cp": 500.0,
        "L_f": 267700.0,
        "T_solidus": 1700.0,
        "T_liquidus": 1800.0,
        "T0": 293.0,
        "DeltaH_LV": 7.41e6,
        "R_v": 150.774,
        "Pa": 101325.0,
        "T_boil": 3090.0,
    },
    "laser": {
        "radius": 60.0e-6,
        "absorptivity": 0.30,
        "power_nominal": 200.0,
        "path": {
            "type": "gcode",
            "file": "linear_track.gcode",   # resolved relative to config/paths/
        },
    },
    "io": {},   # empty – we don't use the IOManager at all
}

# ---------------------------------------------------------------------------
# 2. Build SimulationContext from the dictionary
# ---------------------------------------------------------------------------
context = SimulationContext.from_dict(config)

# ---------------------------------------------------------------------------
# 3. Instantiate the solver (bypass the factory if you want, or use it)
# ---------------------------------------------------------------------------
from fast_heat_solv.solvers.spectral import SpectralSolver
from fast_heat_solv.backends import NumpyBackend

solver = SpectralSolver(NumpyBackend())  # use get_backend("cupy") for GPU
state = solver.initialize(context)       # context injected here

# ---------------------------------------------------------------------------
# 4. Custom time loop – pure physics, no I/O
# ---------------------------------------------------------------------------
t = 0.0
dt = context.num.dt
t_end = context.num.t_end

print(f"Running heat solver in library mode: t_end={t_end:.2e} s, dt={dt:.2e} s")
print(f"Mesh: {context.geom.n.x} x {context.geom.n.y} x {context.geom.n.z}")
print("-" * 60)

step = 0
while t < t_end:
    state, metrics = solver.step(t, dt)
    t += dt
    step += 1

    # Extract whatever diagnostics you need from the metrics dict
    T_max = metrics.get("T_surface_max", float("nan"))
    P_laser = metrics.get("P_laser", 0.0)
    print(f"  step {step:>4d} | t = {t:.4e} s | T_max = {T_max:.1f} K | P_laser = {P_laser:.2f} W")

print("-" * 60)
print("Orchestrator finished – no files were written to disk.")
