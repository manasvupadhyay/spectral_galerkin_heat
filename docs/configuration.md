# Configuration Guide

Simulations are configured with YAML files that map the physical properties and runtime
settings. Example configurations live under `simulations/config/`; this page mirrors
{download}`standard_test.yaml <../simulations/config/standard_test.yaml>`.

:::{note}
Physical quantities are written as a `{value, unit}` mapping, for example:

```yaml
dt:
  value: 2.5e-6
  unit: "s"
```

A plain scalar (`dt: 2.5e-6`) is also accepted; the unit then defaults to the SI unit listed
below.
:::

## Property Reference

### `simulation`
Declares the run and solver behavior.
* **name**: Identifier tag for the run (`str`).
* **method**: Solver type (`str`, e.g. `"spectral"`, `"fem"`).
* **backend**: Execution device (`str`, `"cpu"` or `"gpu"`).
* **duration**: Total physical simulation time (`s`).
* **dt**: Time step size (`s`).
* **update_interval**: Steps between ETA prints to the terminal (`int`).

### `domain`
Dimensions and grid resolution.
* **size**: Box dimensions `[Lx, Ly, Lz]` in metres.
* **mesh**: Grid resolution `[nx, ny, nz]` (integers).

### `material`
Physical material parameters.
* **name**: Material label (`str`).
* **rho**: Density (`kg/m3`)
* **k**: Thermal conductivity (`W/(m.K)`)
* **Cp**: Specific heat capacity (`J/(kg.K)`)
* **L_f**: Latent heat of fusion (`J/kg`)
* **T_solidus**: Solidus temperature (`K`)
* **T_liquidus**: Liquidus temperature (`K`)
* **T0**: Initial / ambient temperature (`K`)
* **h_conv**: Convective heat transfer coefficient (`W/(m2.K)`)
* **DeltaH_LV**: Specific enthalpy of vaporization (`J/kg`)
* **R_v**: Specific gas constant for vapor (`J/(kg.K)`)
* **Pa**: Ambient pressure (`Pa`)
* **T_boil**: Boiling temperature (`K`)

### `laser`
* **radius**: Beam radius (`m`).
* **absorptivity**: Power fraction absorbed (dimensionless, 0–1; a plain scalar).
* **power_nominal**: Nominal emitted power (`W`).
* **path**: `type` (e.g. `"gcode"`) and `file` pointing to the path file.

### `io`
Export settings controlling what is saved and when.

- **output_interval**: Steps between outputs during the run (`int`). Set the value to `null` to
  disable periodic outputs.
- **outputs**: Output types saved at each interval (e.g. `full_volume`). See {ref}`output-types`.
- **at_end**: Output types saved once at the end (e.g. `profiles`, `cut_views`). See
  {ref}`output-types`.
- **profiles_locations**: Optional. `'laser'`, `'hotspot'`, or an explicit `[x, y]` in metres.
- **cut_views_planes**: Optional. Planes for 2D cut views (`xy`, `yz`, `xz`).

### Global Dataclass `SimulationContext`

All configuration sections are deserialized into a unified `SimulationContext` object passed to
`build_solver()` and the `StandaloneHeatRunner`.

```python
from dataclasses import dataclass, field
from typing import Any, Dict
from fast_heat_solv.core.parameters import (
    NumParams, GeomParams, MaterialParams, LaserParams, FineMeshParams
)
from fast_heat_solv.core.laser import LaserPath

@dataclass
class SimulationContext:
    num: NumParams
    mat: MaterialParams
    geom: GeomParams
    laser: LaserParams
    laser_path: LaserPath
    io: Dict[str, Any]                 # flat I/O config dict from YAML
    method: str = "spectral"           # "spectral" or "fem"
    backend: str = "cpu"               # "cpu", "gpu", or "cpu_linear"
    fine: FineMeshParams = field(default_factory=FineMeshParams)
```
</content>
