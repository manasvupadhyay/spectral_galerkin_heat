# fastHeatSolv

**A semi-analytical, uncoupled, modal solution for the fully nonlinear heat equation with support for CPU/GPU backends and G-code-driven laser paths.**
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[**Read the full Sphinx Documentation**](https://theoadx.github.io/hsg-docs/) (or build locally via `make -C docs html`)

---

fastHeatSolv is a modular framework designed for simulating nonlinear heat transfer. It uses semi-analytical spectral methods to achieve high performance on both CPU and GPU hardware, and fully supports complex laser trajectories.

## Quickstart

fastHeatSolv is installed from source; it is not published on PyPI. The recommended path uses
[`uv`](https://github.com/astral-sh/uv), which creates an isolated environment and installs the
dependencies:

```bash
# 1. Get the code
git clone https://github.com/TheoADX/fastHeatSolv.git
cd fastHeatSolv

# 2. Install the environment (creates .venv/ and installs dependencies)
uv sync             # CPU only
uv sync --group gpu # GPU, requires CUDA 13.x

# 3. Run the smallest example
uv run python simulations/main.py simulations/examples/01_quickstart.yaml
```

`uv sync` makes `fast_heat_solv` importable and `uv run` executes inside the managed environment,
so no separate package-install step is required. Results are saved to `out/<timestamp>_<name>/` in
HDF5/XDMF format.

If `uv` is not installed, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`; see
the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for other
platforms.

## Usage

`fastHeatSolv` can be used either as a standalone CLI runner (`simulations/main.py` driven by a
`.yaml` config) or as an imported Python library (building a `SimulationContext` and calling
`solver.step(...)` directly). Both are covered by the examples, ordered from a short run to a full
non-linear case and then library mode:

- **Examples**: [`simulations/examples/`](simulations/examples/), runnable configurations and the
  library-mode `orchestrator.py`.
- **Tutorial & configuration reference**: see the
  [documentation](https://theoadx.github.io/hsg-docs/) (`examples` and `configuration` pages).

Production and GPU configurations and the study drivers are in `simulations/research/`.

## Installation & Environments

### Prerequisites

- **Python 3.12 or newer.**
- **[`uv`](https://github.com/astral-sh/uv)** (recommended): manages the virtual environment and
  dependencies. Plain `pip` also works (see below).
- **For the GPU backend only:** an NVIDIA GPU with the **CUDA 13.x** toolkit installed
  system-wide (the CPU backend needs nothing extra).

### Optional dependency groups

`uv sync` installs the CPU core. Add hardware- or task-specific extras with `--group`:

| Command | Adds |
| --- | --- |
| `uv sync` | CPU core (default; sufficient to run the examples) |
| `uv sync --group gpu` | CuPy GPU backend *(requires system CUDA 13.x)* |
| `uv sync --group viz` | Plotting / visualization helpers |
| `uv sync --group docs` | Sphinx toolchain to build the docs |
| `uv sync --group dev` | Test + lint tooling (`pytest`, `black`, …) |
| `uv sync --all-groups` | Everything above |

Build the documentation locally:
```bash
uv sync --group docs
uv run make -C docs html      # Output: docs/_build/html/index.html
```

### Installing with pip instead of uv

If you prefer a manually managed environment, install the package in editable mode from the repo
root:
```bash
python -m pip install -e .            # CPU core
python -m pip install -e ".[gpu]"     # + GPU backend (needs system CUDA 13.x)
python -m pip install -e ".[docs]"    # + docs toolchain
```

## Citation

If you use this code in your research, please cite:

*(Citation to be added)*

## License

This project is licensed under the Apache License, Version 2.0. 
See the [LICENSE](LICENSE) file for the full text.

Copyright © 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris, Route de Saclay, Palaiseau, 91128, France.

