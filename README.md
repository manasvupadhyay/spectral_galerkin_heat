# fastHeatSolv

**A semi-analytical, modular solution for the heat equation with support for CPU/GPU backends and G-code-driven laser paths.**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Directory Structure](#directory-structure)
  - [Component Descriptions](#component-descriptions)
- [Parameter Management & G-Code Integration](#parameter-management--g-code-integration)
  - [Configuration Example](#1-configuration-structure-yaml-example)
  - [Laser Path Strategy](#2-laser-path-strategy)
  - [Unified Parameter Object](#3-unified-parameter-object)
  - [Flexible Output Control](#flexible-output-control)
- [Development Roadmap](#development-roadmap)

---

## Overview

fastHeatSolv is a modular, extensible framework for simulating heat transfer using semi-analytical and numerical methods. It is designed for flexibility, supporting both CPU and GPU computation, and is capable of simulating complex laser paths (including G-code) for additive manufacturing and related applications.

## Architecture

The project is structured around the **Abstract Factory** pattern, enabling easy extension to new backends (CPU, GPU, distributed) and numerical methods (Spectral, FEM, etc.).

### Directory Structure

```
fastHeatSolv/
├── main.py                     # Entry point: parses args, instantiates SimulationFactory
├── config/                     # YAML configuration files for simulation parameters
├── data/
│   └── paths/                  # G-code files for laser paths
├── out/                        # Simulation results (per-run subfolders)
│   └── {timestamp}_{tag}/
│       ├── fields/             # 3D volumetric fields (HDF5/NPY)
│       ├── profiles/           # 1D temperature profiles
│       ├── plots/              # PNG/SVG plots
│       ├── logs/               # Run logs (stdout, errors)
│       └── diagnostics/        # Timing/convergence (JSON/CSV)
├── core/
│   ├── workflow.py             # Simulation loop (orchestrator, strategy pattern)
│   ├── parameters.py           # Data classes: PhysParams, NumParams, GeomParams, IOParams
│   └── io.py                   # Abstract base class for IOManager
├── interfaces/
│   ├── factory.py              # Abstract Factory: interface for creating solvers, IO managers
│   └── solver.py               # Abstract Product: HeatSolver interface
├── implementations/
│   ├── factories/
│   │   ├── cpu_factory.py      # Concrete Factory: SpectralSolverCPU, LocalFSIOManager
│   │   ├── gpu_factory.py      # Concrete Factory: SpectralSolverGPU, LocalFSIOManager
│   │   └── fem_factory.py      # Concrete Factory: FEMSolver, LocalFSIOManager
│   ├── solvers/
│   │   ├── spectral_cpu.py     # CPU-based Spectral Solver (numba)
│   │   ├── spectral_gpu.py     # GPU-based Spectral Solver (cupy)
│   │   └── fem_solver.py       # FEM Wrapper (FEniCS/Ansys)
│   ├── file_io/
│   │   └── fs_io.py            # Local file system IO manager
│   └── physics/
│       ├── spectral_cpu_kernels.py   # Low-level physics (latent heat, evaporation)
│       └── spectral_gpu_kernels.py   # Low-level physics (latent heat, evaporation)
└── utils/
    ├── spectral_helpers.py     # Math utilities (DCT, grid manipulation)
    └── visualisation.py        # Visualization utilities
```

### Component Descriptions

- **Client (`main.py`)**
  - Reads configuration/command-line arguments
  - Selects and injects the appropriate Factory (CPU/GPU/FEM) into the Workflow
  - Runs the Workflow

- **Workflow Strategy (`core/workflow.py`)**
  - Contains `SimulationWorkflow` (high-level director)
  - Defines *what* happens (initialize, time loop, solve, apply physics, save, post-process)
  - Uses abstract interfaces for solver and IO

- **Simulation Factory (`interfaces/factory.py`, `implementations/factories/`)**
  - Abstract interface: `create_solver()`, `create_io_manager()`, `create_physics_handler()`
  - Concrete factories for CPU, GPU, FEM

- **Solvers (`implementations/solvers/`)**
  - Mathematical engines (SpectralSolverCPU: fftw/numpy/numba, SpectralSolverGPU: cupy)

This modular structure allows easy addition of new backends (e.g., distributed MPI) or new physics (e.g., alternative latent heat models) without rewriting the main simulation loop.

---

## Parameter Management & G-Code Integration

Parameter passing and laser path definition are handled via structured configuration files (YAML/JSON) and Data Transfer Objects (DTOs).

### 1. Configuration Structure (YAML Example)

Jobs are defined by a config file, allowing selection of simulation **method** (algorithm) and **backend** (hardware):

```yaml
simulation:
  name: "spectral_test_run"
  method: "spectral"          # Options: "spectral", "fem"
  backend: "cpu"              # Options: "cpu" (standard), "gpu" (if supported)
  duration: 0.012             # s
  dt: 6.0e-6                  # s
  update_interval: 20         # steps (ETA update frequency)

domain:
  size: [0.01, 0.005, 0.0025] # [Lx, Ly, Lz] in m
  mesh: [512, 256, 1024]      # [nx, ny, nz] (dimensionless)

material:
  name: "GenericSteel"
  rho: 7850.0                 # kg/m^3
  k: 15.0                     # W/(m·K)
  Cp: 500.0                   # J/(kg·K)
  L_f: 267700.0               # J/kg (Latent heat of fusion)
  T_solidus: 1700.0           # K
  T_liquidus: 1800.0          # K
  T0: 293.0                   # K (Ambient temperature)
  
  # Evaporation parameters
  DeltaH_LV: 7.41e6           # J/kg (Specific enthalpy of vaporization)
  R_v: 150.774                # J/(kg·K) (Specific gas constant for vapor)
  Pa: 101325.0                # Pa (Ambient pressure)
  T_boil: 3090.0              # K

laser:
  radius: 60.0e-6             # m (r_b)
  absorptivity: 0.30          # (dimensionless, 0.0 to 1.0)
  power_nominal: 200.0        # W
  
  path:
    type: "gcode"
    file: "linear_track.gcode"
    initial_position: [0.0, 0.0025] # [x, y] in m

io:
  interval: 0.0012            # s (Output frequency)
  outputs: [full_volume]      
  at_end: [full_volume, profiles, cut_views]  
  profiles_locations:
    - 'laser'                 # Location in m or 'laser' for dynamic center
  cut_views_planes:
    - xy
    - yz
    - xz
```

**IO Section Details:**

- `interval`: How often to save outputs during the simulation (in seconds).
- `outputs`: List of output types to save at each interval (e.g., `full_volume`).
- `at_end`: List of output types to save at the end of the simulation (e.g., `profiles`, `cut_views`).
- `profiles_locations`: Optional, (x,y) tuple or 'laser' or 'hotspot'.
- `cut_views_planes`: Optional, planes for extracting 2D cut views.


### 2. Laser Path Strategy

The solver uses a consistent reference frame: the top surface is always $z = L_z$.

```
 +Z (BUILD DIRECTION)
            ^          ^  +y
            |         /
            |        /
            | +=====================+   <-- Top Layer, Lasered (z = Lz)
            |/=====================/|
            +====================+  |
            |                    |  |
            |        CUBOID      |  +
            |        (PART)      | /   <-- Base Layer sits on platform
(X,Y,Z=0)   |                    |/
ORIGIN >----+====================+--------------------> +X
                 ( X-Y Plane)
```

The `LaserPath` interface decouples heat source physics (e.g., Gaussian) from movement logic:

```python
class LaserState(dataclass):
    x: float
    y: float
    power: float
    is_on: bool

class LaserPath(ABC):
    @abstractmethod
    def get_state(self, time: float) -> LaserState:
        """Returns laser position and power at a given simulation time."""
        pass

class GCodeLaserPath(LaserPath):
    def __init__(self, gcode_file: str):
        self.segments = self._parse_gcode(gcode_file)
        # Segments would contain: (start_pos, end_pos, start_time, end_time, power)
    
    def _parse_gcode(self, filepath):
        # Parses G0 (move), G1 (linear cut), M words for power
        # Calculates timing based on F (feed rate)
        pass
```

### 3. Unified Parameter Object

All configuration sections are deserialized into a unified `SimulationContext` object passed to the Abstract Factory.

```python
@dataclass
class SimulationContext:
    num: NumParams
    geom: GeomParams
    mat: MaterialParams
    laser_path: LaserPath
    io: IOParams
    method: str = "spectral"
    backend: str = "cpu"
```

#### Flexible Output Control

The `io` section allows fine-grained control over what is saved and when:

- `full_volume`: Save the full 3D field at intervals and/or at the end
- `profiles`: Save 1D temperature profiles at specified locations
- `cut_views`: Save 2D slices (xy, yz, xz) at intervals or at the end

This enables efficient disk usage and post-processing tailored to your needs.

---

## Development Roadmap

- [ ] **Refactor `comparison.py`**: Use the new loader to access data cleanly
    - Update to use helper functions instead of manual filename iteration
    - Allow passing a specific `run_id` for comparison
- [ ] **Melt-Pool extraction**: An efficient routine to extract meltpool 
    - Shape and metrcis from temperature fields
- [ ] **Update Logic**: Enable the solver to load a previous known temperature field
- [ ] **Adding Material**: Change the domain definition on the fly to
    - Account for an added layer of material
---

## License

Lorem Ipsum

## Citation

Lorem Ipsum 
