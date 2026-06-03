"""End-to-end integration test for the FastHeatSolv CPU pipeline.

What this test covers
---------------------
SimulationContext construction → factory → StandaloneHeatRunner time loop →
LocalFSIOManager XDMF / HDF5 output → physics sanity checks on the saved field.

The test is intentionally slower than the unit tests because it runs the full
numba JIT pipeline (warm from the on-disk cache after the first run, ~5 s
thereafter).  It is excluded from the default ``pytest`` run and must be
invoked explicitly::

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
        x=ctx.geom.Lx / 2,
        y=ctx.geom.Ly / 2,
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
    from fast_heat_solv.factories.cpu_factory import CPUSimulationFactory
    from fast_heat_solv.runner import StandaloneHeatRunner

    # The IO manager resolves output_root to a relative './out/'; chdir keeps
    # all output inside the pytest temp directory.
    monkeypatch.chdir(tmp_path)

    context = _build_context(_CONFIG)
    runner = StandaloneHeatRunner(
        context,
        CPUSimulationFactory(context),
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
def test_gpu_simulation_e2e(tmp_path, monkeypatch):
    """GPU pipeline smoke test — skipped when CuPy / CUDA is unavailable.

    Mirrors the CPU test exactly but uses GPUSimulationFactory.
    """
    try:
        import cupy
        cupy.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip("CuPy not available or no CUDA device found")

    from fast_heat_solv.factories.gpu_factory import GPUSimulationFactory
    from fast_heat_solv.runner import StandaloneHeatRunner

    monkeypatch.chdir(tmp_path)

    gpu_cfg = copy.deepcopy(_CONFIG)
    gpu_cfg["simulation"]["backend"] = "gpu"

    context = _build_context(gpu_cfg)
    runner = StandaloneHeatRunner(context, GPUSimulationFactory(context))
    with _log_picard_iterations(runner) as picard_counts:
        runner.run()

    _assert_final_step_converged(picard_counts, runner.heat_solver.max_picard_iter)

    run_dirs = list((tmp_path / "out").glob("*"))
    assert len(run_dirs) == 1
    h5_files = sorted((run_dirs[0] / "fields").glob("field_step*.h5"))
    assert h5_files, "No HDF5 field file written for GPU run"

    _assert_field_sane(h5_files[-1], _T0)
