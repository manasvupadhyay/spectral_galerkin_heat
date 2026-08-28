"""Tests for LocalFSIOManager (io_utils/spectral_fs_io.py)."""

import h5py
import numpy as np
import pytest

from spectral_galerkin_heat.io_utils.spectral_fs_io import LocalFSIOManager, _save_field_to_hdf5
from spectral_galerkin_heat.solvers.spectral_cpu_linear import SpectralSolverCPULinear


class _Ctx:
    """Minimal stand-in: initialize() only reads context.io."""
    def __init__(self, io):
        self.io = io


@pytest.fixture
def linear_state(tiny_context):
    # Pure-NumPy state.
    return SpectralSolverCPULinear(tiny_context).initialize()


def test_get_output_path_before_init_raises():
    # Paths are only meaningful once initialize() has created the run dir.
    with pytest.raises(RuntimeError):
        LocalFSIOManager().get_output_path("x.h5", subdir="fields")


def test_initialize_creates_tree(tmp_path, monkeypatch):
    # initialize() lays out out/<run_id>/{fields,profiles,logs,diagnostics}.
    monkeypatch.chdir(tmp_path)
    m = LocalFSIOManager()
    m.initialize(_Ctx({}))
    assert m.run_id and m.base_dir
    for sub in ("fields", "profiles", "logs", "diagnostics"):
        assert (tmp_path / m.base_dir / sub).is_dir()


def test_get_output_path_known_and_custom_subdir(tmp_path, monkeypatch):
    # Known subdirs map via self.dirs; an unknown name is treated as a custom subdir.
    monkeypatch.chdir(tmp_path)
    m = LocalFSIOManager()
    m.initialize(_Ctx({}))
    assert m.get_output_path("f.h5", "fields").endswith(f"{m.base_dir}/fields/f.h5")
    assert m.get_output_path("f.h5", "custom").endswith(f"{m.base_dir}/custom/f.h5")
    assert m.get_output_path("f.h5").endswith(f"{m.base_dir}/f.h5")


def test_save_field_hdf5_shape_mismatch_raises(tmp_path):
    # Field shape must match the coord lengths, else the HDF5/XDMF would be malformed.
    field = np.zeros((2, 2, 2), dtype=np.float32)
    coords = (np.arange(3), np.arange(2), np.arange(2))  # x len 3 != nx 2
    with pytest.raises(ValueError):
        _save_field_to_hdf5(str(tmp_path / "f"), field, coords, value_name="temperature")


# --- output scheduling (fast; save_step is stubbed out) ---------------------

def test_process_step_writes_only_on_interval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = LocalFSIOManager()
    m.initialize(_Ctx({"output_interval": 2, "outputs": ["full_volume"]}))
    seen = []
    monkeypatch.setattr(m, "save_step", lambda *a, **k: seen.append(a[1]))
    for step in range(6):
        m.process_step(step * 1e-6, step, state=None, laser_path=None)
    # First write is at step == interval (the initial _next_output_step), then
    # every `interval` steps after — so step 0 is skipped.
    assert seen == [2, 4]


def test_process_step_interval_none_disables_periodic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = LocalFSIOManager()
    m.initialize(_Ctx({"output_interval": None, "outputs": ["full_volume"]}))
    seen = []
    monkeypatch.setattr(m, "save_step", lambda *a, **k: seen.append(a[1]))
    m.process_step(0.0, 0, state=None, laser_path=None)
    assert seen == []  # periodic disabled -> nothing written


def test_process_end_saves_each_at_end_type(tmp_path, monkeypatch):
    # process_end writes exactly one output per configured at_end type.
    monkeypatch.chdir(tmp_path)
    m = LocalFSIOManager()
    m.initialize(_Ctx({"at_end": ["full_volume", "modes"]}))
    seen = []
    monkeypatch.setattr(m, "save_step", lambda *a, **k: seen.append(k["output_type"]))
    m.process_end(1e-3, 9, state=None, laser_path=None)
    assert seen == ["full_volume", "modes"]


# --- load_step --------------------------------------------------------------

def test_load_step_uninitialized_returns_none():
    # Loading before initialize() returns None (no run dir), not a crash.
    assert LocalFSIOManager().load_step() is None


def test_load_step_latest_and_index(tmp_path, monkeypatch):
    # 'latest' picks the highest step; an int loads that step; missing -> None.
    monkeypatch.chdir(tmp_path)
    m = LocalFSIOManager()
    m.initialize(_Ctx({}))
    fields = m.get_output_path("", "fields")
    for step in (0, 5):
        with h5py.File(f"{fields}field_step{step:06d}.h5", "w") as f:
            d = f.create_dataset("temperature", data=np.full((2, 2, 2), float(step)))
            d.attrs["time"], d.attrs["step"] = step * 1e-6, step

    assert m.load_step("latest")["step"] == 5
    assert m.load_step(0)["temperature"][0, 0, 0] == 0.0
    assert m.load_step(99) is None


# --- full-volume / modes writing (real state, still numba-free) -------------

def test_save_full_volume_writes_h5_and_xmf(tmp_path, monkeypatch, tiny_context, linear_state):
    # full_volume writes the reconstructed field as field_stepNNNNNN.h5 + .xmf.
    monkeypatch.chdir(tmp_path)
    m = LocalFSIOManager()
    m.initialize(tiny_context)
    m.save_step(0.0, 0, linear_state, laser_path=None, output_type="full_volume")

    base = m.get_output_path("field_step000000", "fields")
    assert (tmp_path / f"{base}.h5").exists()
    assert (tmp_path / f"{base}.xmf").exists()
    assert (tmp_path / m.get_output_path("temperature_series.xmf", "fields")).exists()
    num = tiny_context.num
    with h5py.File(f"{base}.h5", "r") as f:
        t = f["temperature"]
        assert t.shape == (num.nz + 1, num.ny + 1, num.nx + 1)
        assert t.attrs["step"] == 0


def test_save_modes_appends_across_calls(tmp_path, monkeypatch, tiny_context, linear_state):
    # modes output appends each step to one resizable HDF5 dataset.
    monkeypatch.chdir(tmp_path)
    m = LocalFSIOManager()
    m.initialize(tiny_context)
    m.save_step(0.0, 0, linear_state, laser_path=None, output_type="modes")
    m.save_step(1e-6, 1, linear_state, laser_path=None, output_type="modes")

    with h5py.File(m.get_output_path("modes.h5", "fields"), "r") as f:
        assert f["modes"].shape[0] == 2  # grew 1 -> 2
        assert list(f["step"][:]) == [0, 1]


def test_save_slices_writes_png(tmp_path, monkeypatch, tiny_context, linear_state):
    # _save_slices must hand load_xdmf a Path, not a string.
    monkeypatch.chdir(tmp_path)

    class _Laser:
        # slices need the beam position to centre the cut plane.
        def get_state(self, t, dt):
            return type("S", (), {"x": 0.0, "y": 0.0})()

    m = LocalFSIOManager()
    m.initialize(tiny_context)
    m._save_slices(0.0, 0, linear_state, laser_path=_Laser(),
                   slice_planes=[{"name": "xy", "normal": "z"}])

    png = tmp_path / m.get_output_path("slice_xy_step000000.png", "slices")
    assert png.exists() and png.stat().st_size > 0


def test_profiles_roundtrip_through_loader(tmp_path, monkeypatch, tiny_context, linear_state):
    # Pofiles written by the manager must be discoverable by SimulationResult.
    from spectral_galerkin_heat.io_utils.loader import SimulationResult

    monkeypatch.chdir(tmp_path)
    m = LocalFSIOManager()
    m.initialize(tiny_context)
    m.save_step(0.0, 0, linear_state, laser_path=None, output_type="profiles")

    res = SimulationResult(m.base_dir)
    assert res.list_steps() == [0]
    coords, temps = res.get_profile(0, "x")
    assert len(coords) == len(temps) > 0
