# Architecture & Design

A single unified solver runs on CPU or GPU by injecting a `MathBackend`. Where an
abstraction earns its keep it lives in a co-located `base.py` (`HeatSolver`,
`MathBackend`, `LaserPath`); single-implementation interfaces are deliberately
avoided.

## Directory Structure

```text
fastHeatSolv/
├── pyproject.toml                  # Package definition and dependencies
├── README.md
├── src/
│   └── fast_heat_solv/
│       ├── __init__.py             # Package version + public API
│       ├── runner.py               # StandaloneHeatRunner: simulation loop orchestrator
│       ├── core/
│       │   ├── parameters.py       # Data classes: SimulationContext, NumParams, MaterialParams, …
│       │   ├── laser.py            # LaserState dataclass + LaserPath ABC
│       │   └── vector.py           # Vec3: immutable (x, y, z) value type
│       ├── backends/
│       │   ├── base.py             # MathBackend: bundles array module (xp) + kernels
│       │   ├── numpy_backend.py    # NumpyBackend (CPU)
│       │   ├── cupy_backend.py     # CupyBackend (GPU)
│       │   ├── _registry.py        # backend registry: register_backend + get_backend
│       │   └── __init__.py         # get_backend(name); registers built-in backends
│       ├── solvers/
│       │   ├── base.py             # HeatSolver ABC
│       │   ├── spectral.py         # SpectralSolver (unified CPU/GPU, backend-injected)
│       │   ├── spectral_cpu_linear.py
│       │   └── __init__.py         # build_solver(context): solver selector
│       ├── physics/
│       │   ├── spectral_state.py        # Backend-agnostic state (SpectralGrid, FineMeshState, …)
│       │   ├── spectral_ops.py          # Backend-agnostic free functions
│       │   ├── spectral_cpu_kernels.py  # Low-level CPU spectral kernels (NumPy / Numba)
│       │   ├── spectral_gpu_kernels.py  # Low-level GPU spectral kernels (CuPy / CUDA)
│       │   └── spectral_helpers.py      # DCT reconstruction utilities
│       └── io_utils/
│           ├── spectral_fs_io.py   # LocalFSIOManager  (HDF5 + XDMF)
│           ├── xdmf_io.py          # XdmfBuilder + structured/unstructured readers
|           ...
├── simulations/
│   ├── main.py                     # CLI entry point: parses YAML, builds solver, runs simulation
│   ├── example_orchestrator.py     # Library usage example (no I/O, frame-by-frame)
│   └── config/                     # YAML configuration files + G-code paths
└── docs/
    └── ARCHITECTURE.md
```

## Component Descriptions

- **CLI (`simulations/main.py`)**: Parses a YAML config, builds the solver via
  `build_solver(context)`, and delegates execution to `StandaloneHeatRunner`.

- **Runner (`runner.py`)**: `StandaloneHeatRunner` — the simulation loop orchestrator.
  Owns the `while t < t_end` loop; receives a `HeatSolver` and a `LocalFSIOManager`
  by injection (the I/O manager defaults to a fresh `LocalFSIOManager`), delegating
  all physics to the former and all I/O to the latter.

- **Solver selector (`solvers/__init__.py`)**: `build_solver(context)` — a single
  dispatch on `(method, backend)` that returns the configured `HeatSolver`. It
  replaces the former `SimulationFactory` hierarchy, which only ever differed in
  this one choice.

- **Solver base (`solvers/base.py`)**: `HeatSolver` ABC — declares `initialize()`,
  `step()`, and `finalize()`. The unified `SpectralSolver` runs on CPU or GPU
  purely through its injected `MathBackend`; `SpectralSolverCPULinear` is a
  separate linear variant.

- **Backends (`backends/`)**: `MathBackend` bundles an array module (`xp`, NumPy or
  CuPy) with its matching physics-kernel module. A name→factory registry
  (`_registry.py`) resolves backends: `get_backend(name)` returns `NumpyBackend`
  or `CupyBackend`, so the same solver code targets either device. Extra backends
  register themselves with the `@register_backend("name")` decorator, so adding
  one never requires editing `get_backend`.

- **I/O (`io_utils/spectral_fs_io.py`)**: `LocalFSIOManager` — the concrete
  filesystem I/O handler (HDF5 + XDMF). It owns the full I/O contract
  (`initialize`, `process_step`, `process_end`, `finalize`, `save_step`, …);
  `save_step` dispatches per output type to small `_save_*` helpers.

- **Laser (`core/laser.py`)**: `LaserState` dataclass and `LaserPath` ABC.  Used by
  the solver (to query laser position/power) and `io_utils/gcode_path.py`.

- **Physics (`physics/`)**: Backend-agnostic state and free functions
  (`spectral_state`, `spectral_ops`) plus the pure numerical kernels
  (`spectral_cpu_kernels` Numba / `spectral_gpu_kernels` CuPy).

## Adding a backend (PyTorch, JAX, …)

The solver doesn't need a specific array library to run on. It only uses
`backend.xp` (the array module) and `backend.kernels` (the math functions), and
almost all the physics code already works with any NumPy-like `xp`. So a new
backend is mostly plumbing — if needed an different backend, you need to supply four things:

**1. An array module `xp`.** It needs to behave like NumPy: `zeros`, `arange`,
`exp`, `sum`, in-place writes (`multiply(..., out=)`, `a[0,0,0] = v`),
`.astype`, etc. NumPy and CuPy work out of the box. PyTorch and JAX could be used by writing wrappers that rename a few things (and JAX arrays
can't be modified in place, so those writes need adapting).

**2. A kernels module.** Most functions are reused as-is from `spectral_ops` and
`spectral_state` — don't rewrite them. You only need to provide the ones that
truly depend on the library:
- the cosine transform (`DCT_II` / `IDCT_II`);
- the sub-pixel shift (`_ndshift`);
- a couple of simple per-cell formulas (the evaporation and latent-heat source
  terms, and the Gaussian flux `core.laser.super_gaussian_flux`).

  Then subclass `SpectralSolverState`, bind your own `xp`, and give those primitives to `BackendHooks`. Copy `spectral_cpu_kernels.py` as a template.

**3. A `MathBackend` and registration.** Subclass `MathBackend` with your name,
`xp`, and kernels, and add `@register_backend("mybackend")`.

**4. A config name.** Hook the `simulation.backend` string up to your backend in
`solvers/__init__.py` (`build_solver`).

Keep the kernels working for both float32 and float64, and check your backend
against the CPU one with `test_cpu_gpu_equivalence_e2e` in
`tests/test_integration.py`.

## Architecture Diagram

```{mermaid}
flowchart TD
    %% --- Styling ---
    classDef client   fill:#ff7675,stroke:#d63031,stroke-width:2px,color:#000;
    classDef abstract fill:#81ecec,stroke:#00cec9,stroke-width:2px,stroke-dasharray:5 5,color:#000;
    classDef concrete fill:#55efc4,stroke:#00b894,stroke-width:2px,color:#000;
    classDef product  fill:#fdcb6e,stroke:#e17055,stroke-width:2px,color:#000;
    classDef core     fill:#a29bfe,stroke:#6c5ce7,stroke-width:2px,color:#000;

    %% --- Nodes ---
    CLI["simulations/main.py"]:::client
    Runner["runner.py\nStandaloneHeatRunner"]:::client
    Context["core/parameters.py\nSimulationContext"]:::core
    Laser["core/laser.py\nLaserState · LaserPath"]:::core

    Build["solvers/__init__.py\nbuild_solver()"]:::client
    ISolv["solvers/base.py\n≪abstract≫ HeatSolver"]:::abstract
    IBack["backends/base.py\n≪abstract≫ MathBackend"]:::abstract

    Solver["solvers/spectral.py\nSpectralSolver"]:::product
    CPUBack["backends/numpy_backend.py\nNumpyBackend"]:::concrete
    GPUBack["backends/cupy_backend.py\nCupyBackend"]:::concrete
    FSIO["io_utils/spectral_fs_io.py\nLocalFSIOManager"]:::product

    %% --- Relationships ---
    CLI -->|"creates context"| Context
    CLI -->|"build_solver(context)"| Build
    CLI -->|"instantiates with solver + io"| Runner
    Build -->|"returns"| Solver
    Runner -->|"calls"| ISolv
    Runner -->|"calls"| FSIO

    Solver -.->|"implements"| ISolv
    Solver -->|"injected with"| IBack
    IBack  -.->|"implemented by"| CPUBack
    IBack  -.->|"implemented by"| GPUBack

    Solver -->|"queries"| Laser
```
