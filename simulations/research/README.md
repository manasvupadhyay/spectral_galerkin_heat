# `simulations/research/`

Production and GPU configurations and the development drivers used for the
validation and convergence studies. New users should start with the examples in
[`../examples/`](../examples/).

These configurations target large grids (often `backend: gpu`) and reproduce the
finite-element comparison and convergence studies. They can be slow and may
require a CUDA GPU.

## Configurations

| File | Purpose |
| --- | --- |
| `temp_dep_316L.yaml`, `temp_dep_316L_centered.yaml` | Temperature-dependent 316L, FE-matching grid (GPU). |
| `temp_indep_316L.yaml`, `temp_indep_316L_centered.yaml` | Constant-property 316L counterparts. |
| `fullres_tref2300.yaml`, `fullres_tref2300_full.yaml` | Full-resolution GPU runs (`property_correction.tex`, §8.3). |
| `gpu_standard_test.yaml` | Large single-track GPU test. |
| `fast_test.yaml` | High-resolution CPU configuration. |
| `ellipse_test.yaml` | Curved (ellipse) laser path; see `paths/generate_ellipse.py`. |

G-code paths are in [`paths/`](paths/) and resolve relative to this folder.

## Drivers

- `run_convergence.py`: mesh-convergence sweep against a reference field. Writes
  a scratch `tmp_convergence.yaml` next to itself (git-ignored).
- `benchmark_tol_study.py`: Picard convergence-tolerance study (iteration count
  and temperature field as a function of tolerance).

Run from the repository root:

```bash
uv run python simulations/research/benchmark_tol_study.py --help
uv run python simulations/main.py simulations/research/temp_dep_316L.yaml --backend gpu
```
