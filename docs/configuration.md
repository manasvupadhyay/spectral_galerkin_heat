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
