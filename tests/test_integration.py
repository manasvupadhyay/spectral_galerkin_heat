"""End-to-end integration test for the FastHeatSolv CPU pipeline.

What this test covers
---------------------
SimulationContext construction → factory → StandaloneHeatRunner time loop →
LocalFSIOManager XDMF / HDF5 output → physics sanity checks on the saved field.

The test is intentionally slower than the unit tests because each test runs the
full numba + pyfftw pipeline.  A single CPU test (``test_cpu_simulation_e2e``)
takes ~40 s; The whole file is ~4–5 min with the GPU tests (7 tests including 
the GPU/equivalence cases; the GPU tests skip when CuPy/CUDA is absent). 
It is excluded from the default ``pytest`` run and must be invoked explicitly::

    pytest -m integration               # integration tests only
    pytest -m "integration or not integration"  # everything

Design notes
------------
- ``FixedLaser`` implements :class:`~fast_heat_solv.core.laser.LaserPath`
  inline.  The YAML parser in ``SimulationContext.from_dict`` only supports
  the ``gcode`` path type, so we set ``context.laser_path`` directly.
- Full 316L material parameters are required: the evaporation-flux kernel
  divides by ``R_v`` and ``T_boil``; zero values produce NaN.  With 10 time
  steps the surface temperature stays well below T_liquidus = 1800 K, so
  latent heat and evaporation remain physically inactive.
- ``LocalFSIOManager`` always writes to ``./out/`` (``output_root`` is not
  configurable through the io dict).  ``monkeypatch.chdir(tmp_path)`` keeps
  output inside the pytest temp directory.
- Heavy solver imports (pyfftw, numba) are deferred to inside the test
  function bodies so that *file collection* remains fast when running the
  default ``pytest -m "not integration"`` suite.
"""

import contextlib
import copy
import logging
from pathlib import Path

import h5py
import numpy as np
import pytest

from fast_heat_solv.core.laser import LaserPath, LaserState
from fast_heat_solv.core.parameters import SimulationContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal stationary-laser implementation
# ---------------------------------------------------------------------------

class FixedLaser(LaserPath):
    """Stationary laser at a fixed (x, y) position with constant power.

    Parameters
    ----------
    x, y : float
        Laser position in metres.
    power : float
        Nominal laser power in watts.
    """

    def __init__(self, x: float, y: float, power: float) -> None:
        self._x = x
        self._y = y
        self._power = power

    def get_state(self, time: float, dt: float) -> LaserState:
        return LaserState(x=self._x, y=self._y, power=self._power, is_on=True)


# ---------------------------------------------------------------------------
# Reference configuration
# ---------------------------------------------------------------------------

# 150 × 150 × 50 μm domain, 32 × 32 × 16 mesh → dx ≈ 4.7 μm.
# The laser spot diameter is 2 × 60 μm = 120 μm, so the grid resolves it
# with ~25 cells across the diameter.
_CONFIG = {
    "simulation": {
        "method": "spectral",
        "backend": "cpu",
        "duration": 6e-5,   # 10 time steps at dt = 6 μs
        "dt": 6e-6,
        "update_interval": 1e-3,
    },
    "domain": {
        "size": [4e-4, 4e-4, 1e-4],
        "mesh": [32, 32, 16],
    },
    # Full 316L stainless-steel parameters.
    # R_v and T_boil must be non-zero to avoid division-by-zero in the
    # evaporation-flux kernel; values are the standard 316L literature values.
    "material": {
        "name": "316L",
        "rho": 7850.0,
        "k": 15.0,
        "Cp": 500.0,
        "L_f": 267700.0,
        "T_solidus": 1700.0,
        "T_liquidus": 1800.0,
        "T0": 293.0,
        "DeltaH_LV": 7.41e6,
        "R_v": 150.774,
        "Pa": 101325.0,
        "T_boil": 3090.0,
    },
    "laser": {
        "radius": 60.0e-6,
        "absorptivity": 0.3,
        "power_nominal": 200.0,
    },
    # Write the full temperature volume once, at the end of the run.
    "io": {"at_end": ["full_volume"]},
}

_T0 = _CONFIG["material"]["T0"]


def _build_context(cfg: dict) -> SimulationContext:
    """Parse config and attach a FixedLaser at domain centre."""
    ctx = SimulationContext.from_dict(copy.deepcopy(cfg))
    ctx.laser_path = FixedLaser(
        x=ctx.geom.size.x / 2,
        y=ctx.geom.size.y / 2,
        power=cfg["laser"]["power_nominal"],
    )
    return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_field_sane(h5_path: Path, T0: float) -> None:
    """Load the temperature dataset and verify basic physical invariants."""
    with h5py.File(h5_path, "r") as f:
        T = f["temperature"][:]
    assert not np.isnan(T).any(), "NaN values detected in temperature field"
    assert not np.isinf(T).any(), "Inf values detected in temperature field"
    assert T.max() > T0, (
        f"Laser produced no measurable heating: "
        f"T_max = {T.max():.1f} K ≤ T0 = {T0} K"
    )


def _assert_final_step_converged(picard_counts: list[int], max_iter: int) -> None:
    """Assert the final time step converged inside the Picard iteration cap.

    The spectral solver runs at most ``max_picard_iter`` fixed-point
    iterations per step.  A step that reports exactly that many iterations
    bailed out at the cap *without* meeting ``convergence_tol`` — i.e. it did
    not converge.  Early steps may legitimately hit the cap while the field
    is changing fastest, but by the end of the simulation the solution should
    have settled, so the last step must converge below the limit.

    Parameters
    ----------
    picard_counts : list of int
        Per-step Picard iteration counts, as collected by
        :func:`_log_picard_iterations`.
    max_iter : int
        The solver's ``max_picard_iter`` cap.
    """
    assert picard_counts, "No Picard iteration counts were recorded"
    final = picard_counts[-1]
    assert final < max_iter, (
        f"Solver did not converge at the end of the simulation: the final "
        f"time step ran the full {max_iter} Picard iterations (cap = "
        f"max_picard_iter), meaning the fixed-point loop bailed out before "
        f"reaching convergence_tol. Per-step counts: {picard_counts}"
    )


@contextlib.contextmanager
def _log_picard_iterations(runner):
    """Record the Picard iteration count for every time step of *runner*.

    The spectral solver reports the number of fixed-point (Picard) iterations
    it ran for a step as the ``n_evap_iter`` metric.  This wraps the solver's
    ``step`` method to collect that value, then logs a one-line summary when
    the ``with`` block exits.

    The summary is emitted through the ``logging`` module, so it is only
    visible when pytest is told to show logs, e.g.::

        pytest tests -m integration --log-cli-level=INFO
    """
    counts: list[int] = []
    solver = runner.heat_solver
    real_step = solver.step

    def step_with_count(t, dt):
        state, metrics = real_step(t, dt)
        n = metrics.get("n_evap_iter")
        if n is not None:
            counts.append(int(n))
        return state, metrics

    solver.step = step_with_count
    try:
        yield counts
    finally:
        solver.step = real_step
        if counts:
            logger.info(
                "Picard iterations per time step (%d steps): %s "
                "| min=%d max=%d mean=%.1f",
                len(counts), counts,
                min(counts), max(counts), sum(counts) / len(counts),
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_cpu_simulation_e2e(tmp_path, monkeypatch):
    """Full CPU pipeline: config → factory → 10-step loop → XDMF output.

    Assertions
    ----------
    - Runner completes without raising.
    - Final time step converges within the Picard cap (does not hit
      ``max_picard_iter``).
    - Exactly one timestamped run directory exists under ``out/``.
    - At least one HDF5 field file is present in ``fields/``.
    - Final temperature field: no NaN, no Inf, max > T0 (laser heated material).
    """
    # Defer heavy imports so collection stays fast for the default suite.
    from fast_heat_solv.solvers import build_solver
    from fast_heat_solv.runner import StandaloneHeatRunner

    # The IO manager resolves output_root to a relative './out/'; chdir keeps
    # all output inside the pytest temp directory.
    monkeypatch.chdir(tmp_path)

    context = _build_context(_CONFIG)
    runner = StandaloneHeatRunner(
        context,
        build_solver(context),
        # config / yaml_path omitted: skips the ASCII log header that would
        # require a YAML file on disk and a git-rev lookup.
    )
    with _log_picard_iterations(runner) as picard_counts:
        runner.run()

    # --- convergence ---
    _assert_final_step_converged(picard_counts, runner.heat_solver.max_picard_iter)

    # --- directory structure ---
    run_dirs = list((tmp_path / "out").glob("*"))
    assert len(run_dirs) == 1, (
        f"Expected exactly 1 run directory, got {len(run_dirs)}: {run_dirs}"
    )
    fields_dir = run_dirs[0] / "fields"
    h5_files = sorted(fields_dir.glob("field_step*.h5"))
    assert h5_files, f"No HDF5 field file found in {fields_dir}"

    # --- physics ---
    _assert_field_sane(h5_files[-1], _T0)


@pytest.mark.integration
@pytest.mark.parametrize("dtype_name, np_dtype", [
    ("float32", np.float32),
    ("float64", np.float64),
])
def test_cpu_dtype_e2e(tmp_path, monkeypatch, dtype_name, np_dtype):
    """The configured ``simulation.dtype`` flows end-to-end (state + output).

    Runs the full CPU pipeline at the requested precision and checks that the
    solver state arrays and the saved temperature field carry that dtype, and
    that the field is physically sane.
    """
    cfg = copy.deepcopy(_CONFIG)
    cfg["simulation"]["dtype"] = dtype_name

    T = _run_pipeline_final_field(cfg, tmp_path / dtype_name, monkeypatch)

    assert T.dtype == np.dtype(np_dtype), (
        f"Saved field dtype {T.dtype} != configured {np_dtype}"
    )
    assert not np.isnan(T).any() and not np.isinf(T).any()
    assert T.max() > _T0


@pytest.mark.integration
def test_cpu_float32_float64_consistency_e2e(tmp_path, monkeypatch):
    """float64 must agree with the trusted float32 baseline.

    Both precisions solve the identical problem; the final fields should match
    to within a small fraction of the peak temperature (float32 round-off sets
    the floor). This is the "float64 actually works" guard.
    """
    cfg32 = copy.deepcopy(_CONFIG)
    cfg32["simulation"]["dtype"] = "float32"
    cfg64 = copy.deepcopy(_CONFIG)
    cfg64["simulation"]["dtype"] = "float64"

    T32 = _run_pipeline_final_field(cfg32, tmp_path / "f32", monkeypatch).astype(np.float64)
    T64 = _run_pipeline_final_field(cfg64, tmp_path / "f64", monkeypatch)

    assert T32.shape == T64.shape
    T_max = float(T32.max())
    max_abs_diff = float(np.max(np.abs(T64 - T32)))
    # Both runs solve the identical problem, the fields should be equal
    # up to the float32 round-off Vs. the float64 run.
    # Bounded 1e-4 * T_max
    # (~0.39 K for T_max ~3860 K).
    assert max_abs_diff == pytest.approx(0.0, abs=1e-5 * T_max), (
        f"float32 and float64 fields differ by {max_abs_diff:.3e} K, "
        f"exceeding the 1e-4 * T_max = {1e-4 * T_max:.3e} K bound"
    )


@pytest.mark.integration
@pytest.mark.parametrize("dtype_name, np_dtype", [
    ("float32", np.float32),
    ("float64", np.float64),
])
def test_gpu_simulation_e2e(tmp_path, monkeypatch, dtype_name, np_dtype):
    """GPU pipeline smoke test — skipped when CuPy / CUDA is unavailable.

    Mirrors the CPU test but selects the GPU backend, parametrized over
    precision so the float64 GPU path is exercised on CUDA hardware (the
    ``@cuda.jit`` kernels are dtype-generic and cuFFT supports double).
    """
    try:
        import cupy
        cupy.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip("CuPy not available or no CUDA device found")

    from fast_heat_solv.solvers import build_solver
    from fast_heat_solv.runner import StandaloneHeatRunner

    monkeypatch.chdir(tmp_path)

    gpu_cfg = copy.deepcopy(_CONFIG)
    gpu_cfg["simulation"]["backend"] = "gpu"
    gpu_cfg["simulation"]["dtype"] = dtype_name

    context = _build_context(gpu_cfg)
    runner = StandaloneHeatRunner(context, build_solver(context))
    with _log_picard_iterations(runner) as picard_counts:
        runner.run()

    _assert_final_step_converged(picard_counts, runner.heat_solver.max_picard_iter)

    run_dirs = list((tmp_path / "out").glob("*"))
    assert len(run_dirs) == 1
    h5_files = sorted((run_dirs[0] / "fields").glob("field_step*.h5"))
    assert h5_files, "No HDF5 field file written for GPU run"
    _assert_field_sane(h5_files[-1], _T0)
    with h5py.File(h5_files[-1], "r") as f:
        assert f["temperature"][:].dtype == np.dtype(np_dtype)

    _assert_field_sane(h5_files[-1], _T0)


def _run_pipeline_final_field(cfg, run_dir, monkeypatch):
    """Run one full simulation under *run_dir* and return its final field array.

    The IO manager always writes to ``./out/`` relative to the current working
    directory, so each backend is given its own ``run_dir`` to keep outputs
    separate.  The solver is selected from ``cfg``'s backend via
    :func:`build_solver`.  Returns the ``temperature`` dataset of the last
    ``field_step*.h5`` written.
    """
    from fast_heat_solv.solvers import build_solver
    from fast_heat_solv.runner import StandaloneHeatRunner

    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(run_dir)

    context = _build_context(cfg)
    runner = StandaloneHeatRunner(context, build_solver(context))
    runner.run()

    run_dirs = list((run_dir / "out").glob("*"))
    assert len(run_dirs) == 1, (
        f"Expected exactly 1 run directory under {run_dir}, got {run_dirs}"
    )
    h5_files = sorted((run_dirs[0] / "fields").glob("field_step*.h5"))
    assert h5_files, f"No HDF5 field file written under {run_dirs[0]}"
    with h5py.File(h5_files[-1], "r") as f:
        return f["temperature"][:]


@pytest.mark.integration
def test_cpu_gpu_equivalence_e2e(tmp_path, monkeypatch):
    """CPU and GPU backends must produce the same final temperature field.

    Runs the identical simulation on the CPU and GPU backends (selected via
    ``build_solver`` from each config) in separate working directories, then
    compares the final saved ``temperature`` volumes.  The tolerance is loose enough to
    absorb CPU/GPU implementation shift but small enough to
    catch a genuine divergence between the two kernel implementations.

    Skipped when CuPy / CUDA is unavailable (same guard as
    :func:`test_gpu_simulation_e2e`).
    """
    try:
        import cupy
        cupy.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip("CuPy not available or no CUDA device found")

    cpu_cfg = copy.deepcopy(_CONFIG)
    cpu_cfg["simulation"]["backend"] = "cpu"
    gpu_cfg = copy.deepcopy(_CONFIG)
    gpu_cfg["simulation"]["backend"] = "gpu"

    T_cpu = _run_pipeline_final_field(cpu_cfg, tmp_path / "cpu", monkeypatch)
    T_gpu = _run_pipeline_final_field(gpu_cfg, tmp_path / "gpu", monkeypatch)

    assert T_cpu.shape == T_gpu.shape, (
        f"Field shape mismatch: CPU {T_cpu.shape} vs GPU {T_gpu.shape}"
    )
    # FFT/reduction-order shift between backends, abs guards near-T0 cells where
    # relative error is meaningless.
    assert T_gpu == pytest.approx(T_cpu, rel=1e-5, abs=1e-3)

    # Peak-relative agreement: the max absolute deviation must stay within
    # 1e-5 * T_max (T_max > 3000 K for this config).
    T_max = float(T_cpu.max())
    max_abs_diff = float(np.max(np.abs(T_gpu - T_cpu)))
    assert max_abs_diff == pytest.approx(0.0, abs=1e-5 * T_max), (
        f"CPU/GPU temperature fields differ by {max_abs_diff:.3e} K, "
        f"exceeding the 1e-5 * T_max = {1e-5 * T_max:.3e} K bound"
    )
