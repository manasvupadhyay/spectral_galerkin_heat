# Configuration Guide

Simulations are configured with YAML files that map the physical properties and runtime
settings. Example configurations live under `simulations/examples/`; this page mirrors
{download}`02_single_track.yaml <../simulations/examples/02_single_track.yaml>`. For a
walkthrough of the examples, see {doc}`examples`.

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

The following keys control the non-linear (temperature-dependent property) solve. They are only
needed when the material properties are given as temperature-dependent branches (see `material`
below); with constant scalar properties they can be omitted.

* **max_picard_iter**: Maximum Picard fixed-point iterations per time step (`int`). The property
  correction typically needs ~80; ~30 is too low to converge.
* **picard_omega**: Picard relaxation factor (`float`); the iteration contracts at a rate of about
  `1 - picard_omega`.
* **picard_tol**: Relative convergence tolerance on the modes (`float`); the iteration stops once
  the relative change falls below it.

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

**Temperature-dependent properties.** `k`, `rho` and `Cp` may instead be given as
temperature-dependent polynomials with separate solid and liquid branches, blended by the liquid
fraction. Each branch is a list of coefficients in *ascending* powers of `T` (SI units):

```yaml
k:
  solid:  [9.248, 0.01571]     # k_s = 9.248 + 0.01571*T
  liquid: [12.41, 0.003279]    # k_l = 12.41 + 0.003279*T
Cp:
  solid:  [458.98, 0.1328]
  liquid: [769.86]             # constant branch (single coefficient)
```

Using this form makes the solve non-linear; set the Picard keys under `simulation`
(`max_picard_iter`, `picard_omega`, `picard_tol`). See example
`simulations/examples/03_nonlinear_tdep.yaml`.

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
- **at_end**: Output types saved once at the end (e.g. `profiles`, `slices`). See
  {ref}`output-types`.
- **profiles_locations**: Optional. `'laser'`, `'hotspot'`, or an explicit `[x, y]` in metres.
- **run_tag**: Optional. Suffix for the run directory `out/<timestamp>_<run_tag>/` (default `sim`).
- **output_root**: Optional. Base directory for run folders (default `out`).
- **slice_planes**: Optional. Planes for the 2-D slice images. A plane names its two in-plane axes
  (`xz`, `yz`, `xy`); the slice normal is the remaining axis (so `xz` → normal `y`, the
  longitudinal melt-pool section). A single letter is taken as the normal directly.
- **slice_width**, **slice_height**: Optional. In-plane extent of the slice image in metres
  (horizontal and vertical, centred on the laser spot).

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
