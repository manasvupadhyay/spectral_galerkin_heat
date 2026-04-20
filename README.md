# fastHeatSolv

**A semi-analytical, modular solution for the heat equation with support for CPU/GPU backends and G-code-driven laser paths.**
![License](https://img.shields.io/badge/license-CC%20BY--NC--ND-blue.svg)

[**Read the full Sphinx Documentation**](*link to be added*) (or build locally via `make -C docs html`)

---

fastHeatSolv is a modular framework designed for simulating heat transfer in additive manufacturing. It uses semi-analytical spectral methods to achieve high performance on both CPU and GPU hardware, and fully supports complex laser trajectories parsed directly from G-code.

## Quickstart

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

## Installation & Environments

Depending on your hardware, you can request `uv` to install different dependency groups:

- **CPU Core (Recommended)**: `uv sync`
- **GPU Backend**: `uv sync --group gpu` *(Requires CUDA 12.x)*
- **Visualization**: `uv sync --group viz`
- **Everything**: `uv sync --all-groups`

Alternatively, you can install the package in editable mode using standard `pip`:
```bash
python -m pip install -e .
python -m pip install -e ".[gpu]"  # With GPU support
```

## Citation

If you use this code in your research, please cite:

*(Citation to be added)*

