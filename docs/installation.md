# Installation

The primary way to install and manage the environment is using [uv](https://github.com/astral-sh/uv).

## Prerequisites

- **Python**: 3.10 or higher.
- **uv**: A fast Python package installer and resolver.
- **Git**: To clone the repository.

*For GPU support (optional):*
- **CUDA Toolkit**: Compatible with CuPy.
- **NVIDIA Driver**: Matching your CUDA version.

## Installing the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TheoADX/fastHeatSolv.git
   cd fastHeatSolv
   ```

2. **Sync the environment (CPU):**
   ```bash
   uv sync
   ```
   *This creates an isolated virtual environment and installs all required dependencies defined in `pyproject.toml`.*

3. **(Optional) Sync for GPU:**
   If you have a compatible NVIDIA GPU and want to use `gpu` acceleration:
   ```bash
   uv sync --extra gpu
   ```

## Running Simulations

The project is driven by YAML configuration files.

```bash
# General invocation
uv run python simulations/main.py <path_to_config.yaml>

# Run the standard test simulation
uv run python simulations/main.py simulations/config/standard_test.yaml
```

## Building the Documentation

Requires `make` (available via your system package manager: `apt install make`, `brew install make`, etc.).

```bash
uv run make -C docs html
```

The HTML output is written to `docs/_build/html/`. Open `docs/_build/html/index.html` in a browser to view it.
