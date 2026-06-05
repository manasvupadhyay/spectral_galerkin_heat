# Installation

```{admonition} The code is not yet public
:class: important

A public release is coming soon — see {doc}`status`. The steps below apply once you have access.
```

fastHeatSolv uses [uv](https://github.com/astral-sh/uv) to manage its environment.

## Prerequisites

- **Python**: 3.12 or higher.
- **uv**: A fast Python package installer and resolver.

*For GPU support (optional):*
- **CUDA Toolkit**: Compatible with CuPy.
- **NVIDIA Driver**: Matching your CUDA version.

## Installing the Project

<!-- TODO: restore the clone step once the repository is public.
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Andrieux-LMS/spectral-galerkin-AM.git
   cd spectral-galerkin-AM
   ```
-->

1. **Enter the project directory.**

2. **Sync the environment (CPU):**
   ```bash
   uv sync
   ```
   *This creates an isolated virtual environment and installs the dependencies defined in `pyproject.toml`.*

3. **(Optional) Sync for GPU:**
   For a compatible NVIDIA GPU:
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
