# Configuration Guide

Simulations are configured using simple YAML files mapping the physical properties and the runtime execution states.
You can find these configurations under `simulations/config/`.

## Property Reference

### `simulation`
Declares the backend behavior.
* **name**: Identifier tag for the run (`str`).
* **method**: Defines the solver type (`str`, e.g. `"spectral"`).
* **backend**: Execution device (`str`, `"cpu"` or `"gpu"`).
* **duration**: Physical total execution time (`float`, seconds).
* **dt**: Time step size (`float`, seconds).
* **update_interval**: Number of steps between ETA prints to terminal (`int`, dimensionless).

### `domain`
Sets the dimensions and grid resolutions.
* **size**: Dimensional box mapped as `[Lx, Ly, Lz]` in meters.
* **mesh**: Integer resolution grids mapped as `[nx, ny, nz]`.

### `material`
Physical material parameters needed by solvers.
* **rho**: Density (`kg/m³`)
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
* **h_conv**: Convective heat transfer coefficient (`W/(m²·K)`)

### `laser`
* **radius**: Beam radius (`m`).
* **absorptivity**: Power fraction absorbed (dimensionless, 0–1).
* **power_nominal**: Target base emitted power (`W`).
* **path**: Contains string `file` directing to `.gcode` outputs.

### `io`
Export settings controlling what is saved and when.

- **output_interval**: Number of steps between outputs during the simulation (`int`, dimensionless). Set to `null` to disable periodic outputs.
- **outputs**: List of output types to save at each interval (e.g., `full_volume`). See {ref}`output-types` for all available types.
- **at_end**: List of output types to save at the end of the simulation (e.g., `profiles`, `cut_views`). See {ref}`output-types` for all available types.
- **profiles_locations**: Optional. `(x, y)` tuple or `'laser'` or `'hotspot'` for a dynamic center.
- **cut_views_planes**: Optional. Planes for extracting 2D cut views (e.g., `xy`, `yz`, `xz`).

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
