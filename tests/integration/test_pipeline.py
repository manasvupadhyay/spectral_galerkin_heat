"""End-to-end pipeline tests for the FastHeatSolv CPU & GPU backends (§2.4/§2.5).

Covers SimulationContext construction → build_solver → StandaloneHeatRunner time
loop → LocalFSIOManager XDMF / HDF5 output → physics-sanity checks on the saved
field, plus CPU↔GPU equivalence of the final field.

Slower than the unit suite (runs the full numba JIT pipeline); excluded from the
default run.  Invoke explicitly::

    pytest -m integration
    pytest -m "integration and gpu"     # CPU↔GPU equivalence (needs CUDA)

Notes
-----
- ``FixedLaser`` and the Picard helpers live in ``integration/conftest.py``.
- The YAML parser only supports the ``gcode`` path type, so the laser is set on
  ``context.laser_path`` directly.
- Full 316L material is required: the evaporation kernel divides by ``R_v`` /
  ``T_boil``.  With 10 steps the surface stays below T_liquidus, so latent heat
  and evaporation remain physically inactive.
- ``LocalFSIOManager`` always writes to ``./out/``; ``monkeypatch.chdir`` keeps
  output inside the pytest temp directory.
- Heavy solver imports (pyfftw, numba) are deferred into the test bodies.
"""

import copy

import h5py
import numpy as np
import pytest

from fast_heat_solv.core.parameters import SimulationContext

# 400 µm domain, 32×32×16 mesh → dx ≈ 4.7 µm; the 120 µm spot is resolved by
# ~25 cells.  10 steps at dt = 6 µs keeps the surface below T_liquidus.
_CONFIG = {
    "simulation": {
        "backend": "cpu",
        "duration": 6e-5, "dt": 6e-6, "update_interval": 1e-3,
    },
    "domain": {"size": [4e-4, 4e-4, 1e-4], "mesh": [32, 32, 16]},
    "material": {
        "name": "316L", "rho": 7850.0, "k": 15.0, "Cp": 500.0, "L_f": 267700.0,
        "T_solidus": 1700.0, "T_liquidus": 1800.0, "T0": 293.0,
        "DeltaH_LV": 7.41e6, "R_v": 150.774, "Pa": 101325.0, "T_boil": 3090.0,
    },
    "laser": {"radius": 60.0e-6, "absorptivity": 0.3, "power_nominal": 200.0},
    "io": {"at_end": ["full_volume"]},
}
_T0 = _CONFIG["material"]["T0"]


def _build_context(cfg, fixed_laser) -> SimulationContext:
    ctx = SimulationContext.from_dict(copy.deepcopy(cfg))
    ctx.laser_path = fixed_laser(
        ctx.geom.size.x / 2, ctx.geom.size.y / 2, cfg["laser"]["power_nominal"]
    )
    return ctx


def _run_final_field(cfg, run_dir, monkeypatch, fixed_laser) -> np.ndarray:
    """Run one full simulation under *run_dir*, return its final field array."""
    from fast_heat_solv.runner import StandaloneHeatRunner
    from fast_heat_solv.solvers import build_solver

    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(run_dir)
    context = _build_context(cfg, fixed_laser)
    StandaloneHeatRunner(context, build_solver(context)).run()

    run_dirs = list((run_dir / "out").glob("*"))
    assert len(run_dirs) == 1, f"Expected 1 run dir under {run_dir}, got {run_dirs}"
    h5_files = sorted((run_dirs[0] / "fields").glob("field_step*.h5"))
    assert h5_files, f"No HDF5 field file written under {run_dirs[0]}"
    with h5py.File(h5_files[-1], "r") as f:
        return f["temperature"][:]


@pytest.mark.integration
def test_cpu_simulation_e2e(tmp_path, monkeypatch, fixed_laser,
                            assert_field_sane, assert_final_step_converged,
                            log_picard_iterations):
    """Full CPU pipeline: config → solver → 10-step loop → XDMF output."""
    from fast_heat_solv.runner import StandaloneHeatRunner
    from fast_heat_solv.solvers import build_solver

    monkeypatch.chdir(tmp_path)
    context = _build_context(_CONFIG, fixed_laser)
    runner = StandaloneHeatRunner(context, build_solver(context))
    with log_picard_iterations(runner) as picard_counts:
        runner.run()

    assert_final_step_converged(picard_counts, runner.heat_solver.max_picard_iter)

    run_dirs = list((tmp_path / "out").glob("*"))
    assert len(run_dirs) == 1, f"Expected 1 run directory, got {run_dirs}"
    h5_files = sorted((run_dirs[0] / "fields").glob("field_step*.h5"))
    assert h5_files, "No HDF5 field file found"
    with h5py.File(h5_files[-1], "r") as f:
        assert_field_sane(f["temperature"][:], _T0)


@pytest.mark.integration
@pytest.mark.gpu
def test_gpu_simulation_e2e(tmp_path, monkeypatch, fixed_laser,
                            assert_field_sane, assert_final_step_converged,
                            log_picard_iterations):
    """GPU pipeline smoke test — skipped when CuPy / CUDA is unavailable."""
    try:
        import cupy
        cupy.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip("CuPy not available or no CUDA device found")

    from fast_heat_solv.runner import StandaloneHeatRunner
    from fast_heat_solv.solvers import build_solver

    monkeypatch.chdir(tmp_path)
    gpu_cfg = copy.deepcopy(_CONFIG)
    gpu_cfg["simulation"]["backend"] = "gpu"
    context = _build_context(gpu_cfg, fixed_laser)
    runner = StandaloneHeatRunner(context, build_solver(context))
    with log_picard_iterations(runner) as picard_counts:
        runner.run()

    assert_final_step_converged(picard_counts, runner.heat_solver.max_picard_iter)
    run_dirs = list((tmp_path / "out").glob("*"))
    assert len(run_dirs) == 1
    h5_files = sorted((run_dirs[0] / "fields").glob("field_step*.h5"))
    assert h5_files, "No HDF5 field file written for GPU run"
    with h5py.File(h5_files[-1], "r") as f:
        assert_field_sane(f["temperature"][:], _T0)


@pytest.mark.integration
@pytest.mark.gpu
def test_cpu_gpu_equivalence_e2e(tmp_path, monkeypatch, fixed_laser):
    """CPU and GPU backends must produce the same final temperature field."""
    try:
        import cupy
        cupy.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip("CuPy not available or no CUDA device found")

    cpu_cfg = copy.deepcopy(_CONFIG)
    gpu_cfg = copy.deepcopy(_CONFIG)
    gpu_cfg["simulation"]["backend"] = "gpu"

    T_cpu = _run_final_field(cpu_cfg, tmp_path / "cpu", monkeypatch, fixed_laser)
    T_gpu = _run_final_field(gpu_cfg, tmp_path / "gpu", monkeypatch, fixed_laser)

    assert T_cpu.shape == T_gpu.shape, f"shape mismatch: {T_cpu.shape} vs {T_gpu.shape}"
    # rtol absorbs FFT/reduction-order differences between backends; atol guards
    # near-T0 cells where relative error is meaningless.
    # CONFIRM: rtol/atol pending a tuning run on a CUDA box (§2.4).
    np.testing.assert_allclose(T_gpu, T_cpu, rtol=1e-5, atol=1e-3)
    max_abs_diff = float(np.max(np.abs(T_gpu - T_cpu)))
    assert max_abs_diff < 0.01, f"CPU/GPU differ by {max_abs_diff:.3e} K (> 0.01 K cap)"
