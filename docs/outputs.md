---
tocdepth: 1
---

(output-types)=
# Output Types

Output types are triggered either periodically (every `output_interval` steps, via `outputs`) or once at the end of the simulation (via `at_end`). Both keys accept any combination of the types listed below.

All outputs are written under `out/<run_id>/`, where `<run_id>` is a timestamp combined with the simulation `name`.

## `full_volume`

Reconstructs the full 3D temperature field from spectral coefficients and writes it to disk as HDF5 + XDMF.

**Files produced**

- `fields/field_step{N:06d}.h5` — temperature array
- `fields/field_step{N:06d}.xmf` — XDMF metadata
- `fields/temperature_series.xmf` — time-series index, updated after each save; open this in [ParaView](https://www.paraview.org/) to browse all steps

**Extra config keys**: none.

---

## `profiles`

Extracts 1D temperature profiles along each axis (x, y, z) through a chosen centre point. Lighter than `full_volume` when only cross-sections are needed.

**Files produced**

- `profiles/x_step{N:06d}.txt`
- `profiles/y_step{N:06d}.txt`
- `profiles/z_step{N:06d}.txt`

Each file contains two columns: coordinate (m) and temperature (K).

**`profiles_locations`** (required) — controls the centre point:

| Value | Behaviour |
|---|---|
| `'laser'` | laser position at the saved step |
| `'hotspot'` | hottest surface point (automatic) |
| `[[x, y]]` | fixed coordinate in metres |

---

## `slices`

Renders 2D PNG slices of the temperature field on chosen cross-slice planes. The full field is reconstructed internally if no XDMF file exists for that step.

**Files produced**

- `slices/slice_{plane}_step{N:06d}.png` for each plane in `slice_planes`

**`slice_planes`** (required) — any subset of `xy`, `yz`, `xz`. A plane names its two in-plane axes; the slice normal is the remaining axis (so `xz` → normal `y`, the longitudinal melt-pool section).

The slice is centred on the laser position at the saved step, with its top aligned to the surface. Its in-plane extent is set by `slice_width` and `slice_height` (metres), defaulting to 0.6 mm × 0.2 mm. See {doc}`configuration` for these keys.

---

## `modes`

Saves the raw DCT spectral coefficients instead of the reconstructed field. All steps are appended to a single resizable HDF5 file.

**Files produced**

- `fields/modes.h5` — datasets: `modes` (shape `[n_saves, nz, ny, nx]`), `time`, `step`

More compact and faster to write than `full_volume`. Useful for restarts, post-processing, or inspecting the spectral content of the solution.

**Extra config keys**: none.
