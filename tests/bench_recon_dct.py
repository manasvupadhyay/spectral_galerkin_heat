#!/usr/bin/env python
"""Verify reconstruct_temperature_DCT matches the tensor-product version & benchmark."""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
from fast_heat_solv.core.parameters import SimulationContext

# ---------- tiny config --------------------------------------------------
config = {
    "simulation": {"method": "spectral", "backend": "cpu_linear",
                   "duration": 6e-6, "dt": 6e-6, "update_interval": 1},
    "domain": {"size": [0.005, 0.0025, 0.00125], "mesh": [512, 256, 1024]},
    "material": {"name": "316L", "rho": 7850.0, "k": 15.0, "Cp": 500.0,
                 "L_f": 267700.0, "T_solidus": 1700.0, "T_liquidus": 1800.0,
                 "T0": 293.0, "DeltaH_LV": 7.41e6, "R_v": 150.774,
                 "Pa": 101325.0, "T_boil": 3090.0},
    "laser": {"radius": 60e-6, "absorptivity": 0.3, "power_nominal": 200.0,
              "path": {"type": "gcode", "file": "linear_track.gcode"}},
    "io": {},
}

ctx = SimulationContext.from_dict(config)

from fast_heat_solv.solvers.spectral import SpectralSolver
from fast_heat_solv.backends import NumpyBackend
solver = SpectralSolver(NumpyBackend())
state = solver.initialize(ctx)

# Run a few steps to get non-trivial coefficients
for i in range(10):
    state, _ = solver.step(i * ctx.num.dt, ctx.num.dt)

a = state.a.copy()

# ---- Prepare full reconstruction bases (needed by tensor-product version) ----
state.grid.prepare_full_reconstruction(ctx.geom)

from fast_heat_solv.physics.spectral_helpers import reconstruct_temperature_volume, reconstruct_temperature_DCT

# ---- Correctness check ---------------------------------------------------
T_ref = reconstruct_temperature_volume(a, state)
T_dct = reconstruct_temperature_DCT(a, state)

print(f"T_ref shape: {T_ref.shape}   T_dct shape: {T_dct.shape}")
print(f"T_ref dtype: {T_ref.dtype}   T_dct dtype: {T_dct.dtype}")
print(f"T_ref range: [{T_ref.min():.4f}, {T_ref.max():.4f}]")
print(f"T_dct range: [{T_dct.min():.4f}, {T_dct.max():.4f}]")

abs_diff = np.abs(T_ref - T_dct)
rel_diff = abs_diff / (np.abs(T_ref) + 1e-10)
print(f"Max absolute diff: {abs_diff.max():.6e}")
print(f"Max relative diff: {rel_diff.max():.6e}")
print(f"Mean absolute diff: {abs_diff.mean():.6e}")

tol = 1e-1  # float32 loosened for DCT-I vs tensor-product
if abs_diff.max() < tol:
    print(f"\n  PASS  (max diff {abs_diff.max():.2e} < {tol})")
else:
    print(f"\n  FAIL  (max diff {abs_diff.max():.2e} >= {tol})")

# ---- Benchmark -----------------------------------------------------------
N_RUNS = 5
# Warmup
for _ in range(2):
    reconstruct_temperature_volume(a, state)
    reconstruct_temperature_DCT(a, state)

t0 = time.perf_counter()
for _ in range(N_RUNS):
    reconstruct_temperature_volume(a, state)
t_tensor = (time.perf_counter() - t0) / N_RUNS

t0 = time.perf_counter()
for _ in range(N_RUNS):
    reconstruct_temperature_DCT(a, state)
t_dct = (time.perf_counter() - t0) / N_RUNS

print(f"\nBenchmark ({N_RUNS} runs, mesh {a.shape}):")
print(f"  Tensor-product: {t_tensor*1e3:.1f} ms")
print(f"  DCT-I:          {t_dct*1e3:.1f} ms")
print(f"  Speedup:        {t_tensor/t_dct:.1f}x")
