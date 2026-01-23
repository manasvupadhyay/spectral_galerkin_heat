# fastHeatSolv
A semi analytical solution for the heat equation

## Proposed Architecture (Refactoring)

The project is moving towards a modular architecture based on the **Abstract Factory** pattern to decouple the high-level workflow from the specific implementations (CPU/GPU backends, different solver strategies).

### File Structure

The proposed directory structure separates core interfaces, strategies, and backend-specific implementations, along with dedicated folders for configuration and path data:

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
│   │   ├── cpu_factory.py      # [Concrete Factory] Creates CPUSimulationFactory.
│   ├── solvers/
│   │   ├── spectral_cpu.py     # [Concrete Product] CPU-based Spectral Solver.
│   ├── file_io/                # Concrete IO implementations
│   │   └── fs_io.py            # [Concrete Product] Local file system IO manager.
│   └── physics/
│       └── kernels.py          # Low-level physical laws (Latent heat, evaporation)
|                                 implementation agnostic or specific.
└── utils/
    └── helpers.py              # Shared mathematical utilities (DCT, grid          
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

Instead of hardcoding values in python scripts, a job is defined by a config file:

```yaml
simulation:
  name: "single_track_test"
  duration: auto               # or explicit seconds. "auto" implies deriving from G-code path
  dt: 1.0e-5
  output_interval: 1.0e-3

domain:
  size: [0.02, 0.01, 0.005]    # [Lx, Ly, Lz] in meters
  mesh: [512, 256, 128]        # [nx, ny, nz]

material:
  name: "Ti64"
  rho: 4420.0
  k: 7.0
  Cp: 550.0
  L_f: 2.8e5                  # Latent heat
  T_solidus: 1878.0
  T_liquidus: 1928.0

laser:
  radius: 50.0e-6             # Beam radius (1/e^2 or D4sigma?)
  absorptivity: 0.35
  # The path strategy determines how (x,y) and Power evolve over time
  path:
    type: "gcode"             # Options: "gcode", "linear", "function"
    file: "paths/layer_1.gcode"
    gcode_flavor: "reprap"    # To handle different G-code dialects
    initial_position: [0.0, 0.0]
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
    num: NumericalParams
    geom: GeomParams
    mat: MaterialParams
    laser_path: LaserPath  # The initialized path strategy object
    io: IOParams # Configuration for output behavior
```

This ensures that the Solver (Product) logic remains pure:
`q_laser = laser_path.get_state(t).power * gaussian_kernel(...)`
It doesn't care if the position comes from a simple `v*t` formula or a complex G-code interpretation.



TO DO 

==>Refactor Reconstruction Logic:

Action: Update reconstruct_temperature_* functions to use the dynamic xp backend for tensor contractions (tensordot), ensuring the geometry basis vectors (geom.Bz_fine) match the device of the coefficients a.
Goal: Run the heavy reconstruction steps entirely on the GPU if the data is there.

==>Isolate Numba Kernels:

Action: Extract the @njit decorated functions (like _compute_source_term_from_temperature) from helpers.py or wrap them in a dispatcher that throws an error or uses a simplified pure-Python/Cupy fallback if GPU arrays are passed (since Numba CPU kernels crash on GPU arrays).
Goal: Ensure helpers.py doesn't force a dependency on CPU-only compiled code when running in a GPU context.

==> edit yaml exemple in the readme (take vizu exemple into account)

==> Dispatch functions from test+speed spectral and helpers to the kernels etc

==> We have to do that Step 6: Refactor comparison.py
Goal: Use the new loader to cleanly access data.

Update comparison.py to use the helper functions from Step 5 instead of iterating raw filenames manually.
Allow passing a specific run_id to compare against.