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
* **backend**: Execution device (`str`, `"cpu"`, `"gpu"`, or `"cpu_linear"`).
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
Dimensions and grid resolution. Both accept a plain list or the `{value, unit}` mapping form
(the list goes under `value`):

```yaml
domain:
  size:
    value: [0.003, 0.0025, 0.0005]   # [Lx, Ly, Lz]
    unit: "m"
  mesh:
    value: [64, 48, 24]              # [nx, ny, nz]
    unit: ""
```

* **size**: Box dimensions `[Lx, Ly, Lz]` in metres.
* **mesh**: Grid resolution `[nx, ny, nz]` (integers; dimensionless).

### `fine_mesh`
Optional; controls where the latent-heat source is evaluated (relevant only when `L_f > 0`).

- **Omitted (default): grid mode.** The latent-heat source is evaluated on the coarse main grid.
  This is lower-memory and faster for small/quick runs. 
- **Present: box mode.** A refined, laser-following sub-box resolves the sharp mushy-zone
  gradients that the coarse grid cannot. Use it on large production grids where a fine mushy
  zone matters and the melt pool is a small fraction of the domain (so the localized projection
  is cheaper than a full-grid transform).

Box-mode keys:

* **refinement**: Fine cells per coarse cell, per axis (`int`, default `4`). The fine spacing is
  the coarse spacing divided by this factor.
* **box_size**: Extent `[Lx, Ly, Lz]` of the box in metres (default `[0.9e-3, 0.9e-3, 0.04e-3]`).
  The x/y extents form a window that tracks the laser; the z extent is the near-surface depth.

In box mode the box must enclose the melt pool: the latent-heat source is projected onto the
modal basis only from inside it, so any melting beyond the box (deeper than its z extent, or
outside the x/y window) is silently dropped. The solver checks this each step and logs a warning
once if the latent-heat source reaches a box boundary — enlarge `box_size` if you see it.

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
* **h_conv_bottom**: Convective heat transfer coefficient at the bottom face z=0 (`W/(m2.K)`)
* **h_conv_top**: Convective heat transfer coefficient at the laser-facing top face z=Lz (`W/(m2.K)`)
* **DeltaH_LV**: Specific enthalpy of vaporization (`J/kg`)
* **R_v**: Specific gas constant for vapor (`J/(kg.K)`)
* **Pa**: Ambient pressure (`Pa`)
* **T_boil**: Boiling temperature (`K`)

**Temperature-dependent properties.** `k`, `rho` and `Cp` may instead be given as
temperature-dependent expressions with separate solid and liquid branches, blended by the liquid
fraction. Each branch is a string math expression of `T` (SI units, standard Python syntax 
`**` for exponentiation, not `^`), not limited to polynomials:

```yaml
k:
  solid:  "9.248 + 0.01571 * T"
  liquid: "12.41 + 0.003279 * T"
Cp:
  solid:  "458.98 + 0.1328 * T"
  liquid: "769.86"             # constant branch (a bare-number string)
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
    backend: str = "cpu"               # "cpu", "gpu", or "cpu_linear"
    fine: FineMeshParams = field(default_factory=FineMeshParams)
```
</content>
