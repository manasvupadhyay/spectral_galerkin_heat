#!/usr/bin/env python
"""Using fastHeatSolv as a library.

Builds a SimulationContext from a Python dictionary and advances the solver step
by step, without any disk I/O or IOManager involvement.

Usage:
    uv run python simulations/examples/orchestrator.py
"""

import sys
import os

# This file lives in simulations/examples/; the repo root is two levels up.
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXAMPLE_DIR, "..", ".."))

# Ensure the package is importable when running the script directly.
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

from fast_heat_solv.core.parameters import SimulationContext

# ---------------------------------------------------------------------------
# 1. Define the configuration as a plain Python dictionary. An external driver
#    would typically extract this from its own configuration.
# ---------------------------------------------------------------------------
config = {
    "simulation": {
        "backend": "cpu",
        "duration": 6e-5,      # short run for demonstration
        "dt": 6e-6,
        "update_interval": 1,
    },
    "domain": {
        "size": [0.01, 0.005, 0.0025],   # Lx, Ly, Lz  [m]
        "mesh": [64, 32, 16],             # coarse mesh
    },
    "material": {
        # Constant reference 316L properties at T0 = 293 K (as in example 01).
        "name": "316L",
        "rho": 7957.5,
        "k": 13.851,
        "Cp": 497.89,
        "L_f": 0.0,
        "T_solidus": 1674.15,
        "T_liquidus": 1697.15,
        "T0": 293.0,
        "DeltaH_LV": 7.416e6,
        "R_v": 150.774,
        "Pa": 101325.0,
        "T_boil": 3090.0,
    },
    "laser": {   # TODO modify to provide an exampe how to provide a laserPAth it is defined as ABC, would be better to provide a full example how to provide a laser path
        "radius": 60.0e-6,
        "absorptivity": 0.30,
        "power_nominal": 200.0,
        "path": {
            "type": "gcode",
            "file": "linear_track.gcode",   # resolved from <config_dir>/paths/ (see below)
        },
    },
    "io": {},   # empty: the IOManager is not used
}

# ---------------------------------------------------------------------------
# 2. Build SimulationContext from the dictionary
#    Passing config_dir lets the solver resolve the G-code path relative to
#    this example's own paths/ folder, regardless of the current directory.
# ---------------------------------------------------------------------------
context = SimulationContext.from_dict(config, config_dir=EXAMPLE_DIR)

# ---------------------------------------------------------------------------
# 3. Instantiate and initialize the solver
# ---------------------------------------------------------------------------
from fast_heat_solv.solvers.spectral import SpectralSolver
from fast_heat_solv.backends import NumpyBackend

solver = SpectralSolver(NumpyBackend())  # use get_backend("cupy") for GPU
state = solver.initialize(context)

# ---------------------------------------------------------------------------
# 4. Time loop
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

    # Read diagnostics from the metrics dict returned by each step.
    T_max = metrics.get("T_surface_max", float("nan"))
    P_laser = metrics.get("P_laser", 0.0)
    print(f"  step {step:>4d} | t = {t:.4e} s | T_max = {T_max:.1f} K | P_laser = {P_laser:.2f} W")

print("-" * 60)
print("Orchestrator finished: no files were written to disk.")
