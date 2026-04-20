# fastHeatSolv

**A semi-analytical, modular solution for the heat equation with support for CPU/GPU backends and G-code-driven laser paths.**
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

fastHeatSolv is a modular framework designed for simulating heat transfer in additive manufacturing. It uses semi-analytical spectral methods to achieve high performance on both CPU and GPU hardware, and fully supports complex laser trajectories parsed directly from G-code.

## 🚀 Quickstart

This project uses [`uv`](https://uv.io) for fast, reliable dependency and virtual environment management.

```bash
# Clone the repository
git clone https://github.com/TheoADX/fastHeatSolv.git
cd fastHeatSolv

# Install the standard CPU environment
uv sync

# Run the standard test simulation
uv run python simulations/main.py simulations/config/standard_test.yaml
```

*Results are automatically saved to `out/<timestamp>_<tag>/` with HDF5/XDMF formats ready for ParaView.*

## 🔧 Installation & Environments

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

## 📚 Documentation

Detailed documentation covering architecture, configuration parameters (YAML), API references, and the mathematical methods used by the spectral solvers is generated via **Sphinx**.

To build and view the full documentation locally:
```bash
uv sync --group docs
cd docs
make html
```
Then, open `docs/_build/html/index.html` in your web browser.

## 🏗️ Architecture

fastHeatSolv is built around the **Abstract Factory pattern**, isolating physical computation kernels (`HeatSolver`) from data telemetry (`IOManager`). See the [Architecture Guide](docs/ARCHITECTURE.md) for deep-dives into how the event loop operates and how to extend the framework with new numerical backends.

## 📜 Citation

If you use this code in your research, please cite:

*(Citation to be added)*
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

- [ ] **Update Logic**: Enable the solver to load a previous known temperature field
- [ ] **Adding Material**: Change the domain definition on the fly to
    - Account for an added layer of material
- [ ] **API Design**: Enable calls from an external Orchestrator / Adapter
- [ ] **CFL and discretisation**: Add physics based warning (CFL, number of modes, discretization)
- [ ] **Custom flux**: Enable user to write custom boundary flux (less hardcoded)
- [ ] **Boundary Condition Modularity**: Enable user to choose BCs freely (less hardcoded)
- [ ] **Unit Tests**: Write and use unit tests
- [ ] **Remove CPU LINEAR**: It was for test purposes
---

Proposition 1 for Modular Boundary Conditions : (easy)
User writes boundary condition in the yaml choosing among a set of predefined BoundaryFlux class


```yaml
boundaries:
  top:
    - type: RadiationBoundary
      emissivity: 0.8
      T_inf: 293.0
    - type: custom_user_module.MyLaserPulseFlux # Example Python code hook
  bottom:
    - type: ConvectionBoundary
      h: 15.0
      T_inf: 293.0
```

Proposition 2 for Modular Boundary Condition : (harder)
User writes his main depending on needs, a parser can read the expressions. Then injected in the solver. 

```python
def main():
    config = load_config(args.config)
    context = SimulationContext.from_dict(config)
    factory = get_factory(context)
    runner = StandaloneHeatRunner(context, factory)

    # --- User defined formula parser ---
    x, y, t, T = sp.symbols('x y t T')
    
    # Example : Radiation boundary (Stefan-Boltzmann)
    # sigma = 5.67e-8, epsilon = 0.8, T_inf = 293
    # flux = -0.8 * 5.67e-8 * (T**4 - 293.0**4)
    
    # Example : Custom moving heat source
    flux_expr = 1e6 * sp.exp(-((x - 0.005)**2 + (y - 0.005)**2) / 0.0001) * sp.sin(2 * sp.pi * t * 100)**2 
    
    # Compile (see if needed)
    fast_flux_func = sp.lambdify((x, y, t, T), flux_expr, modules=['numexpr', 'numpy'])
    
    # Inject into solver
    runner.solver.set_top_flux(fast_flux_func)
    # Provide also .set_bottom_flux ...
    # -----------------------------------
```

