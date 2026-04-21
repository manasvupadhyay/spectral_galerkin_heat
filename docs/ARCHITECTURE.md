# Architecture & Design

Each submodule owns its own abstract base class (`base.py`) co-located with its concrete
implementations.

## Directory Structure

```text
fastHeatSolv/
├── pyproject.toml                  # Package definition and dependencies
├── README.md
├── src/
│   └── fast_heat_solv/
│       ├── __init__.py             # Package version
│       ├── runner.py               # StandaloneHeatRunner: simulation loop orchestrator
│       ├── core/
│       │   ├── parameters.py       # Data classes: SimulationContext, NumParams, MaterialParams, …
│       │   └── laser.py            # LaserState dataclass + LaserPath ABC
│       ├── solvers/
│       │   ├── base.py             # HeatSolver ABC
│       │   ├── spectral_cpu.py     # SpectralSolverCPU  (NumPy / Numba)
│       │   ├── spectral_gpu.py     # SpectralSolverGPU  (CuPy)
│       │   └── spectral_cpu_linear.py
│       ├── factories/
│       │   ├── base.py             # SimulationFactory ABC
│       │   ├── cpu_factory.py      # CPUSimulationFactory
│       │   ├── gpu_factory.py      # GPUSimulationFactory
│       │   └── cpu_linear_factory.py
│       ├── physics/
│       │   ├── spectral_cpu_kernels.py   # Low-level CPU spectral kernels
│       │   ├── spectral_gpu_kernels.py   # Low-level GPU spectral kernels
│       │   └── spectral_helpers.py       # DCT reconstruction utilities
│       └── io_utils/
│           ├── io_base.py          # IOManager ABC
│           ├── spectral_fs_io.py   # LocalFSIOManager  (HDF5 + XDMF)
|           ...
├── simulations/
│   ├── main.py                     # CLI entry point: parses YAML, dispatches factory, runs simulation
│   ├── example_orchestrator.py     # Library usage example (no I/O, frame-by-frame)
│   └── config/                     # YAML configuration files + G-code paths
└── docs/
    └── ARCHITECTURE.md
```

## Component Descriptions

- **CLI (`simulations/main.py`)**: Parses a YAML config, selects the appropriate factory, and
  delegates execution to `StandaloneHeatRunner`.

- **Runner (`runner.py`)**: `StandaloneHeatRunner` — the simulation loop orchestrator.
  Owns the `while t < t_end` loop, delegates all physics to `HeatSolver` and all I/O to
  `IOManager`. 

- **Factory base (`factories/base.py`)**: `SimulationFactory` ABC — declares `create_heat_solver()`
  and `create_io_manager()`. Concrete factories wire together solver + I/O for a specific backend.

- **Solver base (`solvers/base.py`)**: `HeatSolver` ABC — declares `initialize()`, `step()`, and
  `finalize()`. Concrete solvers contain all physics.

- **IO base (`io_utils/io_base.py`)**: `IOManager` ABC — declares the full I/O contract
  (`initialize`, `process_step`, `process_end`, `finalize`, `save_step`, …).

- **Laser (`core/laser.py`)**: `LaserState` dataclass and `LaserPath` ABC.  Used by both solvers
  (to query laser position/power) and `io_utils/gcode_path.py`.

- **Physics (`physics/`)**: Pure numerical kernels (Numba/CuPy).

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

    IFact["factories/base.py\n≪abstract≫ SimulationFactory"]:::abstract
    ISolv["solvers/base.py\n≪abstract≫ HeatSolver"]:::abstract
    IIO["io_utils/io_base.py\n≪abstract≫ IOManager"]:::abstract

    CPUFact["factories/cpu_factory.py\nCPUSimulationFactory"]:::concrete
    GPUFact["factories/gpu_factory.py\nGPUSimulationFactory"]:::concrete

    CPUSolv["solvers/spectral_cpu.py\nSpectralSolverCPU"]:::product
    GPUSolv["solvers/spectral_gpu.py\nSpectralSolverGPU"]:::product
    FSIO["io_utils/spectral_fs_io.py\nLocalFSIOManager"]:::product

    %% --- Relationships ---
    CLI -->|"creates context + factory"| Context
    CLI -->|"instantiates"| Runner
    Runner -->|"uses"| IFact
    Runner -->|"calls"| ISolv
    Runner -->|"calls"| IIO

    IFact -.->|"implemented by"| CPUFact
    IFact -.->|"implemented by"| GPUFact

    CPUFact -->|"creates"| CPUSolv
    CPUFact -->|"creates"| FSIO
    GPUFact -->|"creates"| GPUSolv
    GPUFact -->|"creates"| FSIO

    CPUSolv -.->|"implements"| ISolv
    GPUSolv -.->|"implements"| ISolv
    FSIO    -.->|"implements"| IIO

    CPUSolv -->|"queries"| Laser
    GPUSolv -->|"queries"| Laser
```

