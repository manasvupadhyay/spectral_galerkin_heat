# Examples

`fastHeatSolv` models can be driven entirely by `YAML` configuration files.

## A Standard Test

The primary example included is {download}`standard_test.yaml <../simulations/config/standard_test.yaml>`, which sets a gaussian laser heat source hitting the surface of a cuboid domain.

Run it using:
```bash
uv run python simulations/main.py simulations/config/standard_test.yaml
```

### Configuration Structure

A typical configuration file contains the physical metrics and solver parameters:

```yaml
# simulations/config/standard_test.yaml
simulation:
  name: "spectral_test_run"
  method: "spectral"          # Options: "spectral", "fem"
  backend: "cpu"              # Options: "cpu" (standard), "gpu" (if supported)
  duration:
    value: 0.005
    unit: "s"
  dt:
    value: 2.5e-6
    unit: "s"
  update_interval: 20         # steps (ETA update frequency)

domain:
  size: [0.005, 0.0025, 0.00125]  # [Lx, Ly, Lz] in m
  mesh: [1000, 500, 1500]         # [nx, ny, nz]

material:
  name: "316L"
  rho:
    value: 7850.0
    unit: "kg/m3"
  k:
    value: 15.0
    unit: "W/(m.K)"
  Cp:
    value: 500.0
    unit: "J/(kg.K)"
  L_f:
    value: 267700.0
    unit: "J/kg"
  T_solidus:
    value: 1700
    unit: "K"
  T_liquidus:
    value: 1800.0
    unit: "K"
  T0:
    value: 293.0
    unit: "K"
  h_conv:
    value: 3000.0
    unit: "W/(m2.K)"
  DeltaH_LV:
    value: 7.41e6
    unit: "J/kg"
  R_v:
    value: 150.774
    unit: "J/(kg.K)"
  Pa:
    value: 101325.0
    unit: "Pa"
  T_boil:
    value: 3090.0
    unit: "K"

laser:
  radius:
    value: 60.0e-6
    unit: "m"
  absorptivity: 0.30
  power_nominal:
    value: 200.0
    unit: "W"
  path:
    type: "gcode"
    file: "linear_track.gcode"

io:
  output_interval: null               # (Output frequency, nb of steps)
  outputs: [full_volume]
  at_end: [full_volume, profiles]
  profiles_locations:
    - 'laser'                         # 'laser', 'hotspot', or explicit [x, y] in m
  cut_views_planes:
    - xy
    - yz
    - xz
```

## Library Integration 

While `fastHeatSolv` provides a standalone CLI, it is also designed to be fully usable as a Python library. This is useful if you want to integrate the solver in a broader codebase where you want to execute the simulation step by step.

See {download}`example_orchestrator.py <../simulations/example_orchestrator.py>` for a complete example. 

```python
from fast_heat_solv.core.parameters import SimulationContext
from fast_heat_solv.solvers.spectral_cpu import SpectralSolverCPU

# 1. Define configuration as a plain Python dictionary
config = {
    "simulation": { "method": "spectral", "backend": "cpu", "dt": 6e-6, "duration": 6e-5 },
    "domain": { "size": [0.01, 0.005, 0.0025], "mesh": [64, 32, 16] },
    "material": { "rho": 7850.0, "k": 15.0, "Cp": 500.0, "name": "316L" },
    "laser": { "radius": 60.0e-6, "absorptivity": 0.30, "power_nominal": 200.0, 
               "path": { "type": "gcode", "file": "linear_track.gcode" } },
    "io": {}, # Empty: No I/O involvement, no files written
}

# 2. Build SimulationContext
context = SimulationContext.from_dict(config)

# 3. Instantiate and initialize the solver directly
solver = SpectralSolverCPU()
state = solver.initialize(context)

# 4. Custom time loop driven by your orchestrator
t = 0.0
dt = context.num.dt
t_end = context.num.t_end

while t < t_end:
    # Step the physics
    state, metrics = solver.step(t, dt)
    t += dt
    
    # Extract needed diagnostics from memory
    T_max = metrics.get("T_surface_max")
    print(f"t = {t:.4e} s | T_max = {T_max:.1f} K")
```
