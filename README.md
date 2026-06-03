# fastHeatSolv

**A semi-analytical, modular solution for the heat equation with support for CPU/GPU backends and G-code-driven laser paths.**
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[**Read the full Sphinx Documentation**](*link to be added*) (or build locally via `make -C docs html`)

---

fastHeatSolv is a modular framework designed for simulating heat transfer in additive manufacturing. It uses semi-analytical spectral methods to achieve high performance on both CPU and GPU hardware, and fully supports complex laser trajectories.

## Usage

`fastHeatSolv` can be used in two main ways: as a standalone simulation runner via CLI, or as an imported Python library.

### 1. CLI Pipeline (Standalone)

When interacting via the CLI, the solver uses `simulations/main.py` and is fully driven by a `.yaml` configuration file.

This project uses [`uv`](https://uv.io) for fast environment management.

```yaml
# Example snippet: standard_test.yaml
geom: {Lx: 0.01, Ly: 0.01, Lz: 0.005}
num: {dt: 1.0e-4, nx: 32, ny: 32, nz: 16, t_end: 0.01}
mat: {name: "Ti6Al4V", rho: 4420.0, k: 25.0, Cp: 650.0}
laser: {radius: 60.0e-6, absorptivity: 0.30, power_nominal: 200.0, path: {type: "gcode", file: "track.gcode"}}
io: {interval: 0.001, outputs: [full_volume]}
```

```bash
# Clone the repository
git clone https://github.com/TheoADX/fastHeatSolv.git
cd fastHeatSolv

# Install the standard CPU environment
uv sync

# Run the standard test simulation
uv run python simulations/main.py simulations/config/standard_test.yaml
```

*Results are automatically saved to `out/<timestamp>_<tag>/` with HDF5/XDMF formats.*

### 2. Library Integration

You can import `fastHeatSolv` as a library. 

In this mode, you pass a dictionary into `SimulationContext.from_dict(...)` and drive the steps directly.

Here is a brief demonstration (see `simulations/example_orchestrator.py` for the full script):

```python
from fast_heat_solv.core.parameters import SimulationContext
from fast_heat_solv.solvers.spectral import SpectralSolver
from fast_heat_solv.backends import NumpyBackend

config = {
    "simulation": { "method": "spectral", "backend": "cpu", "dt": 6e-6, "duration": 6e-5 },
    "domain": { "size": [0.01, 0.005, 0.0025], "mesh": [64, 32, 16] },
    "material": { "rho": 7850.0, "k": 15.0, "Cp": 500.0, "name": "316L" },
    "laser": { "radius": 60.0e-6, "absorptivity": 0.30, "power_nominal": 200.0, 
               "path": { "type": "gcode", "file": "linear_track.gcode" } },
    "io": {}, # Empty: No I/O involvement
}

# 1. Build the context
context = SimulationContext.from_dict(config)

# 2. Instantiate and initialize the solver
solver = SpectralSolver(NumpyBackend())   # use get_backend("cupy") for GPU
state = solver.initialize(context)

# 3. Time loop
t, dt = 0.0, context.num.dt
while t < context.num.t_end:
    state, metrics = solver.step(t, dt)
    t += dt
```

## Installation & Environments

Depending on your hardware, you can request `uv` to install different dependency groups:

- **CPU Core (Recommended)**: `uv sync`
- **GPU Backend**: `uv sync --group gpu` *(Requires CUDA 12.x)*
- **Visualization**: `uv sync --group viz`
- **Docs**: `uv sync --group docs`
- **Everything**: `uv sync --all-groups`

To build the documentation locally:
```bash
uv run make -C docs html
# Output: docs/_build/html/index.html
```

Alternatively, you can install the package in editable mode using standard `pip`:
```bash
python -m pip install -e .
python -m pip install -e ".[gpu]"  # With GPU support
```

## Citation

If you use this code in your research, please cite:

*(Citation to be added)*

## License

This project is licensed under the Apache License, Version 2.0. 
See the [LICENSE](LICENSE) file for the full text.

Copyright © 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique.

