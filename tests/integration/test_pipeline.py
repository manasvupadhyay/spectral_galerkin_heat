"""End-to-end pipeline tests for the spectral_galerkin_heat CPU & GPU backends.

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
  ``T_boil``.  The surface reaches ~3860 K, so latent heat and evaporation are
  both active; see the note on ``dt`` / ``picard_omega`` above ``_CONFIG``.
- ``LocalFSIOManager`` always writes to ``./out/``; ``monkeypatch.chdir`` keeps
  output inside the pytest temp directory.
- Heavy solver imports (pyfftw, numba) are deferred into the test bodies.
"""

import copy

import h5py
import numpy as np
import pytest

from spectral_galerkin_heat.core.parameters import SimulationContext

# 400 µm domain, 32×32×16 mesh → dx ≈ 4.7 µm; the 120 µm spot is resolved by
# ~25 cells.  20 steps at dt = 3 µs over 60 µs.

_CONFIG = {
    "simulation": {
        "backend": "cpu",
        "duration": 6e-5, "dt": 3e-6, "update_interval": 1e-3,
        "picard_omega": 0.31,
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
    from spectral_galerkin_heat.runner import StandaloneHeatRunner
    from spectral_galerkin_heat.solvers import build_solver

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
    from spectral_galerkin_heat.runner import StandaloneHeatRunner
    from spectral_galerkin_heat.solvers import build_solver

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
@pytest.mark.parametrize("dtype_name, np_dtype", [
    ("float32", np.float32),
    ("float64", np.float64),
])
def test_gpu_simulation_e2e(tmp_path, monkeypatch, fixed_laser,
                            assert_field_sane, assert_final_step_converged,
                            log_picard_iterations, dtype_name, np_dtype):
    """GPU pipeline test at both precisions.

    Skipped when CuPy / CUDA is unavailable.
    """
    try:
        import cupy
        _ = cupy.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip("CuPy not available or no CUDA device found")

    from spectral_galerkin_heat.runner import StandaloneHeatRunner
    from spectral_galerkin_heat.solvers import build_solver

    monkeypatch.chdir(tmp_path)
    gpu_cfg = copy.deepcopy(_CONFIG)
    gpu_cfg["simulation"]["backend"] = "gpu"
    gpu_cfg["simulation"]["dtype"] = dtype_name
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
        T = f["temperature"][:]
    assert T.dtype == np.dtype(np_dtype), (
        f"Saved GPU field dtype {T.dtype} != configured {np_dtype}")
    assert_field_sane(T, _T0)


@pytest.mark.integration
@pytest.mark.gpu
def test_cpu_gpu_equivalence_e2e(tmp_path, monkeypatch, fixed_laser):
    """CPU and GPU backends must produce the same final temperature field."""
    try:
        import cupy
        _ = cupy.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip("CuPy not available or no CUDA device found")

    cpu_cfg = copy.deepcopy(_CONFIG)
    gpu_cfg = copy.deepcopy(_CONFIG)
    gpu_cfg["simulation"]["backend"] = "gpu"

    T_cpu = _run_final_field(cpu_cfg, tmp_path / "cpu", monkeypatch, fixed_laser)
    T_gpu = _run_final_field(gpu_cfg, tmp_path / "gpu", monkeypatch, fixed_laser)

    assert T_cpu.shape == T_gpu.shape, f"shape mismatch: {T_cpu.shape} vs {T_gpu.shape}"
    np.testing.assert_allclose(T_gpu, T_cpu, rtol=1e-5, atol=1e-3)
    max_abs_diff = float(np.max(np.abs(T_gpu - T_cpu)))
    assert max_abs_diff < 0.01, f"CPU/GPU differ by {max_abs_diff:.3e} K (> 0.01 K cap)"


@pytest.mark.integration
@pytest.mark.parametrize("dtype_name, np_dtype", [
    ("float32", np.float32),
    ("float64", np.float64),
])
def test_cpu_dtype_e2e(tmp_path, monkeypatch, fixed_laser, dtype_name, np_dtype):
    """The configured ``simulation.dtype`` flows end-to-end (state + output).
    """
    cfg = copy.deepcopy(_CONFIG)
    cfg["simulation"]["dtype"] = dtype_name

    T = _run_final_field(cfg, tmp_path / dtype_name, monkeypatch, fixed_laser)

    assert T.dtype == np.dtype(np_dtype), (
        f"Saved field dtype {T.dtype} != configured {np_dtype}"
    )
    assert not np.isnan(T).any() and not np.isinf(T).any()
    assert T.max() > _T0


@pytest.mark.integration
def test_cpu_float32_float64_consistency_e2e(tmp_path, monkeypatch, fixed_laser):
    """float64 must agree with the trusted float32 baseline.
    """
    cfg32 = copy.deepcopy(_CONFIG)
    cfg32["simulation"]["dtype"] = "float32"
    cfg64 = copy.deepcopy(_CONFIG)
    cfg64["simulation"]["dtype"] = "float64"

    T32 = _run_final_field(cfg32, tmp_path / "f32", monkeypatch, fixed_laser).astype(np.float64)
    T64 = _run_final_field(cfg64, tmp_path / "f64", monkeypatch, fixed_laser)

    assert T32.shape == T64.shape
    T_max = float(T32.max())
    max_abs_diff = float(np.max(np.abs(T64 - T32)))
    # Both runs solve the identical problem, so the fields should agree up to
    # float32 round-off
    assert max_abs_diff <= 1e-5 * T_max, (
        f"float32 and float64 fields differ by {max_abs_diff:.3e} K, "
        f"exceeding the 1e-5 * T_max = {1e-5 * T_max:.3e} K bound"
    )
