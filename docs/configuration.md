# Configuration Guide

Simulations are configured using simple YAML files mapping the physical properties and the runtime execution states.
You can find these configurations under `simulations/config/`.

## Property Reference

### `simulation`
Declares the backend behavior.
* **name**: Identifier tag for the run (`str`).
* **method**: Defines the solver type (e.g. `"spectral"`).
* **backend**: Execution device `cpu` or `gpu`.
* **duration**: Physical total execution time (`float` seconds).
* **dt**: Time step size (`float` seconds).
* **update_interval**: ETA prints to terminal threshold.

### `domain`
Sets the dimensions and grid resolutions.
* **size**: Dimensional box mapped as `[Lx, Ly, Lz]` in meters.
* **mesh**: Integer resolution grids mapped as `[nx, ny, nz]`.

### `material`
Physical material parameters needed by solvers.
* **rho**: Density (`kg/m^3`)
* **k**: Thermal conductivity (`W/(m·K)`)
* **Cp**: Specific heat capacity (`J/(kg·K)`)
* **L_f**: Latent heat of fusion (`J/kg`)
* **T_solidus**: Solidus temperature (`K`)
* **T_liquidus**: Liquidus temperature (`K`)
* **T0**: Ambient temperature (`K`)
* **DeltaH_LV**: Specific enthalpy of vaporization (`J/kg`)
* **R_v**: Specific gas constant for vapor (`J/(kg·K)`)
* **Pa**: Ambient pressure (`Pa`)
* **T_boil**: Boiling temperature (`K`)
* **h_conv**: Convective heat transfer coefficient (`W/(m^2·K)`)

### `laser`
* **radius**: Beam radius (`float`).
* **absorptivity**: Power fraction absorbed (`float`).
* **power_nominal**: Target base emitted power (`float` Watts).
* **path**: Contains string `file` directing to `.gcode` outputs.

### `io`
Export settings.
* **interval**: Time interval to trigger volumetric mesh exports (`float`).
* **outputs**: Formats accepted like `['fields']`.

**Extended Output Control**
The `io` section allows control over what is saved and when:
- `interval`: How often to save outputs during the simulation (in seconds).
- `outputs`: List of output types to save at each interval (e.g., `full_volume`).
- `at_end`: List of output types to save at the end of the simulation (e.g., `profiles`, `cut_views`).
- `profiles_locations`: Optional, (x,y) tuple or `'laser'` or `'hotspot'`.
- `cut_views_planes`: Optional, planes for extracting 2D cut views (e.g., `xy`, `yz`, `xz`).

This enables efficient disk usage and post-processing tailored to your needs.

### Global Dataclass `SimulationContext`

All configuration sections are deserialized into a unified `SimulationContext` object passed to the Abstract Factory.

```python
from dataclasses import dataclass
from fast_heat_solv.core.parameters import NumParams, GeomParams, MaterialParams, IOParams
from fast_heat_solv.core.laser import LaserPath

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
