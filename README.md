# fastHeatSolv
A semi analytical solution for the heat equation

## Proposed Architecture (Refactoring)

The project is moving towards a modular architecture based on the **Abstract Factory** pattern to decouple the high-level workflow from the specific implementations (CPU/GPU backends, distinct numerical methods like Spectral or FEM).

### File Structure

The layout separates core logic, concrete implementations, and simulation data/outputs:

```
fastHeatSolv/
├── main.py                     # [Client] Entry point. Parses args and instantiates the SimulationFactory.
├── config/                     # [Config] YAML configuration files defining simulation parameters.
├── data/
│   └── paths/                  # [Data] G-code files defining laser paths.
├── out/                        # [Output] Root directory for simulation results.
│   └── {timestamp}_{tag}/      # [Run] Specific run container (run_id).
│       ├── fields/             # - 3D Volumetric fields (HDF5/NPY)
│       ├── profiles/           # - 1D Temperature text profiles for plotting
│       ├── plots/              # - Generated PNG/SVG plots
│       ├── logs/               # - Run logs (stdout, errors)
│       └── diagnostics/        # - JSON/CSV timing and convergence
├── core/
│   ├── workflow.py             # [Strategy] Defines the simulation loop (Orchestrator). 
│   │                           # It uses abstract interfaces to run the timeline      
|   |                           #   independent of the backend.
│   ├── parameters.py           # Data Classes (PhysParams, NumParams, GeomParams, IOParams).
│   └── io.py                   # [Interface] Abstract base class for IOManager.
├── interfaces/
│   └── factory.py              # [Abstract Factory] Defines the interface for 
|   |                           #   creating Solvers and IO managers.
│   └── solver.py               # [Abstract Product] Interface for HeatSolver.
├── implementations/
│   ├── factories/
│   │ ├── cpu_factory.py        # [Concrete Factory] Creates SpectralSolverCPU and LocalFSIOManager.
│   │ ├── gpu_factory.py        # [Concrete Factory] Creates SpectralSolverGPU and LocalFSIOManager.
│   │ └── fem_factory.py        # [Concrete Factory] Creates FEMSolver and LocalFSIOManager.
│   ├── solvers/
│   │ ├── spectral_cpu.py       # [Concrete Product] CPU-based Spectral Solver (numba).
│   │ ├── spectral_gpu.py       # [Concrete Product] GPU-based Spectral Solver (cupy).
│   │ └── fem_solver.py         # [Concrete Product] FEM Wrapper (FEniCS/Ansys) adhering to HeatSolver interface.
│   ├── file_io/                # Concrete IO implementations
│   │   └── fs_io.py            # [Concrete Product] Local file system IO manager.
│   └── physics/
│       └── spectral_cpu_kernels.py   # Low-level physical laws (Latent heat, evaporation)
│       └── spectral_gpu_kernels.py   # Low-level physical laws (Latent heat, evaporation)
└── utils/
    └── spectral_helpers.py     # Shared mathematical utilities (DCT, grid          
    |                           #   manipulation).
    └── visualisation.py        # Viz utilities to be used standalone or defined in YAML
```

### Components Description

1.  **Client (`main.py`)**:
    *   Reads configuration/command-line arguments.
    *   Selects the appropriate Factory (e.g., `CPUSimulationFactory` vs `GPUSimulationFactory`).
    *   Injects the factory into the `Workflow`.
    *   Runs the `Workflow`.

2.  **Workflow Strategy (`core/workflow.py`)**:
    *   Contains the `SimulationWorkflow` class.
    *   **Responsibility**: The high-level director. It defines *what* happens (Initialize -> Time Loop -> Solve Step -> Apply Physics -> Save -> Post-process) but not *how*.
    *   It holds references to abstract `HeatSolver` and `IOManager`.

3.  **Simulation Factory (`interfaces/factory.py`, `implementations/factories/`)**:
    *   **Abstract Interface**: `create_solver()`, `create_io_manager()`, `create_physics_handler()`.
    *   **Concrete CPU Factory**: Returns `SpectralSolverCPU` and `XDMFManager`.
    *   **Concrete GPU Factory**: Returns `SpectralSolverGPU` (potentially with batched IO).

4.  **Solvers (`implementations/solvers/`)**:
    *   Implement the mathematical engine.
    *   `SpectralSolverCPU`: Uses `fftw`, `numpy`, and `numba` (current `main_cpu.py` logic).
    *   `SpectralSolverGPU`: Uses `cupy` (current `main_gpu.py` logic).

This structure allows us to easily add new backends (e.g., distributed MPI) or new physical strategies (e.g., different latent heat formulations) without rewriting the main simulation loop.

## Parameter Management & G-Code Integration

To clean up parameter passing and enable complex laser paths (like those defined by G-code), we propose a clear separation of concerns using Data Transfer Objects (DTOs) loaded from a structured configuration file (YAML/JSON).

### 1. Configuration Structure (YAML Example)

Jobs are defined by a config file where you can now select the simulation **method** (algorithm) and **backend** (hardware).

```yaml
simulation:
    name: "spectral_test_run"
    method: "spectral"          # Options: "spectral", "fem"
    backend: "cpu"              # Options: "cpu", "gpu" (only compatible with spectral)
    duration: 0.012             # seconds
    dt: 6.0e-6                  # seconds
    update_interval: 20         # ETA update every 20 steps (approx)

domain:
    size: [0.01, 0.005, 0.0025] # [Lx, Ly, Lz] in meters
    mesh: [10, 10, 10]          # [nx, ny, nz]

material:
    name: "GenericSteel"
    rho: 7850.0
    k: 15.0
    Cp: 500.0
    L_f: 267700.0               # Latent heat J/kg
    T_solidus: 1700.0
    T_liquidus: 1800.0
    T0: 293.0                   # Ambient temperature
    DeltaH_LV: 7.41e6           # Evaporation parameters
    R_v: 150.774
    Pa: 101325.0
    T_boil: 3090.0

laser:
    radius: 60.0e-6             # r_b in meters
    absorptivity: 0.30
    power_nominal: 200.0         # Default power if not specified in path
    path:
        type: "gcode"
        file: "linear_track.gcode"
        initial_position: [0.0, 0.0025] # [x, y] in meters

io:
    interval: 1.2e-3                # Output interval for time-stepped outputs (e.g., full_volume)
    outputs: [full_volume]          # What to save at each interval
    at_end: [full_volume, profiles, cut_views]  # What to save at the end
    profiles_locations:
        - [0.005, 0.0025]
    cut_views_planes:
        - xy
        - yz
        - xz
```

- `interval`: How often to save outputs during the simulation (in seconds).
- `outputs`: List of output types to save at each interval (e.g., `full_volume`).
- `at_end`: List of output types to save at the end of the simulation (e.g., `profiles`, `cut_views`).
- `profiles_locations`: Optional, locations for extracting 1D profiles.
- `cut_views_planes`: Optional, planes for extracting 2D cut views.

This structure is parsed as a flat dictionary and passed to the workflow and IOManager. The workflow will:

- Save all outputs in `outputs` at every interval.
- Save all outputs in `at_end` at the end of the simulation.

Additional config (like locations/planes) is passed to the IOManager for use in output routines.

```

### 2. Laser Path Strategy

We introduce an abstract `LaserPath` interface to decouple the heat source physics (Gaussian distribution) from the movement logic.

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
    laser_path: LaserPath  # The initialized path strategy object
    io: IOParams # Configuration for output behavior, e.g.:
    # io.full_volume.interval, io.profiles.at_end, io.cut_views.planes, etc.
    
    # Execution Config
    method: str = "spectral"
    backend: str = "cpu"
```

#### Flexible Output Control

The `io` section allows you to control what is saved and when:

- `full_volume`: Save the full 3D field at a given interval and/or at the end.
- `profiles`: Save 1D temperature profiles at specified locations, only at the end or at intervals.
- `cut_views`: Save 2D slices (xy, yz, xz) at the end or at intervals.

This enables efficient disk usage and post-processing tailored to your needs.



TO DO 


==> edit yaml exemple in the readme (take vizu exemple into account)

==> Dispatch functions from test+speed spectral and helpers to the kernels etc

==> We have to do that Step 6: Refactor comparison.py
Goal: Use the new loader to cleanly access data.

Update comparison.py to use the helper functions from Step 5 instead of iterating raw filenames manually.
Allow passing a specific run_id to compare against.


==> Refactoring:
reconstruct_temperature_volume_at_points: Move entirely to kernels.
save_temp_profiles: The I/O part (saving to txt) belongs in fs_io.py or the high-level workflow, but the heavy lifting of computing T_x, T_y, T_z (lines 538-568) should be a kernel function called compute_1d_profiles_from_modes.

==> do not do io in spectral_helpers.py