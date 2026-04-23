# Examples

fastHeatSolv models are driven entirely by YAML configuration files, allowing you to run various simulation scenarios without writing custom Python control loops.

## A Standard Test

The primary example included is the `standard_test.yaml`, which sets up a classical moving point source or distributed heat source problem based on analytical solutions.

Run it using:
```bash
uv run python simulations/main.py simulations/config/standard_test.yaml
```

### Configuration Structure

A typical configuration file contains the physical metrics and solver parameters:

```yaml
# simulations/config/standard_test.yaml
simulation:
  backend: "cpu" # Options: ["cpu", "gpu", "cpu_linear"]
  t_end: 0.05    # Simulation duration in seconds
  dt: 0.001      # Time step

output:
  base_dir: "results"
  prefix: "standard_test"
  save_interval: 10 # Save output every 10 steps

material:
  rho: 7850.0  # Density (kg/m^3)
  cp: 500.0    # Specific Heat (J/kg.K)
  k: 15.0      # Thermal Conductivity (W/m.K)

laser:
  power: 500.0   # Laser Power in Watts
  radius: 0.0001 # Beam Radius (100 µm)
  path:
    type: "constant_velocity"
    v_x: 0.8  # Velocity in X
    v_y: 0.0  # Velocity in Y
```

## Library Integration 

While `fastHeatSolv` provides a standalone CLI, it is also designed to be fully usable as a Python library. This is useful if you want to integrate the solver in a broader codebase where you want to execute the simulation step by step.

See [`simulations/example_orchestrator.py`](../simulations/example_orchestrator.py) for a complete example. 

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
