"""Tests for StandaloneHeatRunner orchestration (runner.py)."""

import logging
import sys
from types import SimpleNamespace

import pytest

from spectral_galerkin_heat.runner import StandaloneHeatRunner, initialize_run_logging


def _ctx(n_steps=3, dt=1e-6, dt_nominal=1e-6):
    num = SimpleNamespace(
        dt=dt, dt_nominal=dt_nominal, t_end=n_steps * dt, n_steps=n_steps,
        update_interval=1e9,  # large -> ETA logging only fires on the final step
    )
    return SimpleNamespace(num=num, laser_path=None)


class _Solver:
    def __init__(self):
        self.calls = []
        self.finalized = False

    def initialize(self, ctx):
        return "state0"

    def step(self, t, dt):
        self.calls.append(t)
        return "state", {"n_evap_iter": 1}

    def finalize(self):
        self.finalized = True


class _IO:
    # base_dir = None makes the runner skip its file-logging/header block.
    def __init__(self):
        self.events = []
        self.base_dir = None

    def initialize(self, ctx):
        self.events.append("init")

    def process_step(self, t, step, state, laser):
        self.events.append(("step", step))

    def process_end(self, t, step, state, laser):
        self.events.append(("end", step))

    def finalize(self):
        self.events.append("final")


def test_runner_defaults_io_manager():
    # With no io_manager passed, the runner builds a default LocalFSIOManager.
    from spectral_galerkin_heat.io_utils.spectral_fs_io import LocalFSIOManager

    r = StandaloneHeatRunner(_ctx(), _Solver())
    assert isinstance(r.io_manager, LocalFSIOManager)
    assert isinstance(r.heat_solver, _Solver)


def test_run_executes_n_steps_plus_one_with_exact_time():
    solver, io = _Solver(), _IO()
    StandaloneHeatRunner(_ctx(n_steps=3), solver, io_manager=io).run()
    assert len(solver.calls) == 4  # n_steps + 1
    assert solver.calls == [i * 1e-6 for i in range(4)]  # t = step*dt, no drift


def test_run_io_lifecycle_order_and_finalize():
    solver, io = _Solver(), _IO()
    StandaloneHeatRunner(_ctx(n_steps=2), solver, io_manager=io).run()
    assert io.events == [
        "init", ("step", 0), ("step", 1), ("step", 2), ("end", 2), "final",
    ]
    assert solver.finalized  # solver.finalize() called


def test_dt_correction_is_logged(caplog):
    caplog.set_level(logging.INFO, logger="spectral_galerkin_heat.runner")
    StandaloneHeatRunner(_ctx(dt=1.0e-6, dt_nominal=1.1e-6), _Solver(), io_manager=_IO()).run()
    assert "dt correction" in caplog.text


def test_telemetry_runs_without_psutil(caplog, monkeypatch):
    # Make `import psutil` fail so the _psutil_proc-is-None branch is exercised.
    monkeypatch.setitem(sys.modules, "psutil", None)
    caplog.set_level(logging.INFO, logger="spectral_galerkin_heat.runner")
    io = _IO()
    StandaloneHeatRunner(_ctx(), _Solver(), io_manager=io).run()
    assert "[ETA]" in caplog.text   # telemetry produced a finite ETA line
    assert "final" in io.events     # run completed cleanly


@pytest.mark.slow
def test_initialize_run_logging_header(tmp_path):
    # Writes the log header, copies config.yaml, labels a missing g-code, and
    # degrades git hash to N/A outside a repo (tmp_path is not a git repo).
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("dummy: 1\n")
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    config = {
        "simulation": {"backend": "cpu"},
        "laser": {"path": {"file": "missing.nc"}},
    }
    initialize_run_logging(config, str(yaml_path), str(out_dir))

    log = (out_dir / "logs" / "simulation.log").read_text()
    assert "NOT FOUND" in log          # missing g-code labelled
    assert "Git Commit       : N/A" in log
    assert (out_dir / "config.yaml").exists()  # config copied in
