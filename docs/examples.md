# Examples

`fastHeatSolv` simulations are driven by `YAML` configuration files and run through
`simulations/main.py`. The examples below are ordered by increasing complexity, from a short run
with constant properties to a full non-linear case, followed by library-mode usage. Runs are intended 
to be read and run in order.

All commands are run from the repository root with [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync   # one-time environment setup
```

The example configurations are in `simulations/examples/`.

(example-quickstart)=
## 1. Quickstart

{download}`01_quickstart.yaml <../simulations/examples/01_quickstart.yaml>` runs a single laser
track on a small, coarse cuboid with constant material properties. It is the simplest case:
latent heat of fusion and surface convection are switched off, leaving only evaporative surface
cooling. It completes in about a minute on a CPU and is annotated field by field, serving as the
configuration reference.

```bash
uv run python simulations/main.py simulations/examples/01_quickstart.yaml
```

Results are written to `out/<timestamp>_quickstart/`:

- `profiles/`: temperature along x, y and z through the laser spot (plain text).
- `fields/`: the final volume field, readable in ParaView via the `.xmf` file.
- `slices/`: the 2-D cut-plane image configured under `io.slice_planes`.
- `logs/simulation.log`: per-step metrics (`T_surface_max`, laser power).

```{figure} _images/example_01_meltpool.png
:alt: Longitudinal melt-pool section, example 01
:width: 100%

Longitudinal (x–z, normal y) section through the laser spot at the end of the run. The double
red line is the solidus/liquidus contour bounding the melt pool. This is the slice written to
`slices/` with `slice_planes: [xz]`.
```

(example-single-track)=
## 2. Single track with latent heat

{download}`02_single_track.yaml <../simulations/examples/02_single_track.yaml>` adds the two
effects that example 01 leaves off — latent heat of fusion (`L_f`) and surface convection
(`h_conv`) — at a higher resolution. Material properties are still constant (the Chadwick 316L
values evaluated at `T0 = 293 K`); example 03 makes them temperature-dependent.

```bash
uv run python simulations/main.py simulations/examples/02_single_track.yaml
```

```{figure} _images/example_02_meltpool.png
:alt: Longitudinal melt-pool section, example 02
:width: 100%

Melt-pool section for the single track. The fine z-grid resolves the mushy zone: the two red
contours are the solidus and the liquidus, the latter drawn +10 K (`io.slice_liquidus_offset`)
so the narrow 23 K band is legible.
```

(example-nonlinear)=
## 3. Non-linear, temperature-dependent properties

{download}`03_nonlinear_tdep.yaml <../simulations/examples/03_nonlinear_tdep.yaml>` lets the
conductivity, density and specific heat vary with temperature (solid and liquid polynomial
branches blended by the liquid fraction), making the diffusion problem non-linear. Each time step
is solved with a Picard iteration and a global property correction:

- `max_picard_iter`: iteration cap per step. This example uses `80`; the correction requires
  about this many iterations to converge.
- `picard_omega`: relaxation factor (contraction rate approximately `1 - omega`).

```bash
uv run python simulations/main.py simulations/examples/03_nonlinear_tdep.yaml
```

```{figure} _images/example_03_meltpool.png
:alt: Conduction-mode melt pool, example 03
:width: 100%

Melt-pool section for the slow, low-power track. The pool is nearly symmetric (quasi-stationary
conduction mode); the properties vary with temperature through the field. The two red contours are
the solidus and the (+10 K) liquidus, resolved by the fine z-grid.
```

This is a reduced-grid version of the finite-element comparison case; the full-grid GPU
configuration is `simulations/research/temp_dep_316L.yaml`.

(example-library)=
## 4. Library mode

`fastHeatSolv` can also be used as a Python library. Instead of the CLI, an external driver builds
a `SimulationContext` from a dictionary and calls `solver.step(...)` directly, which is useful
when embedding the solver in a larger codebase. No data is written to disk.

The complete script is in
{download}`orchestrator.py <../simulations/examples/orchestrator.py>`:

```bash
uv run python simulations/examples/orchestrator.py
```

```python
from fast_heat_solv.core.parameters import SimulationContext
from fast_heat_solv.solvers.spectral import SpectralSolver
from fast_heat_solv.backends import NumpyBackend

# 1. Define configuration as a plain Python dictionary
config = {
    "simulation": {"method": "spectral", "backend": "cpu", "dt": 6e-6, "duration": 6e-5},
    "domain": {"size": [0.01, 0.005, 0.0025], "mesh": [64, 32, 16]},
    "material": {"rho": 7957.5, "k": 13.851, "Cp": 497.89, "name": "316L"},  # Chadwick 316L at 293 K
    "laser": {"radius": 60.0e-6, "absorptivity": 0.30, "power_nominal": 200.0,
              "path": {"type": "gcode", "file": "linear_track.gcode"}},
    "io": {},   # empty: the IOManager is not used
}

# 2. Build the context (config_dir resolves the G-code path to <dir>/paths/)
context = SimulationContext.from_dict(config, config_dir="simulations/examples")

# 3. Instantiate and initialize the solver
solver = SpectralSolver(NumpyBackend())   # use get_backend("cupy") for GPU
state = solver.initialize(context)

# 4. Time loop
t, dt, t_end = 0.0, context.num.dt, context.num.t_end
while t < t_end:
    state, metrics = solver.step(t, dt)
    t += dt
    print(f"t = {t:.4e} s | T_max = {metrics.get('T_surface_max'):.1f} K")
```

## Further material

Every configuration key is documented in {doc}`configuration`, and the output formats in
{doc}`outputs`. Production and GPU configurations and the convergence and tolerance study drivers
are in `simulations/research/`.
