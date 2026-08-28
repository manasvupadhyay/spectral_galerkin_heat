"""Tests for the result loader (io_utils/loader.py)."""

import h5py
import numpy as np
import pytest

from fast_heat_solv.io_utils.loader import SimulationResult, list_runs, load_run


def test_list_runs_empty_and_sorted(tmp_path):
    assert list_runs(str(tmp_path / "missing")) == []
    (tmp_path / "20260101-000000_a").mkdir()
    (tmp_path / "20260102-000000_b").mkdir()
    (tmp_path / "note.txt").write_text("x")  # files ignored
    assert list_runs(str(tmp_path)) == ["20260102-000000_b", "20260101-000000_a"]  # newest first


def test_load_run_missing_dir_raises(tmp_path):
    # Loading a non-existent run is an error.
    with pytest.raises(FileNotFoundError):
        load_run("nope", output_root=str(tmp_path))


def _run(tmp_path):
    root = tmp_path / "run"
    (root / "fields").mkdir(parents=True)
    (root / "profiles").mkdir()
    return root


def test_list_steps_parsed_from_profiles(tmp_path):
    # Steps are discovered by parsing the index out of profile filenames.
    root = _run(tmp_path)
    (root / "profiles" / "x_step000003.txt").write_text("0 0\n")
    (root / "profiles" / "y_step000010.txt").write_text("0 0\n")
    (root / "profiles" / "ignore.txt").write_text("0 0\n")
    assert SimulationResult(str(root)).list_steps() == [3, 10]


def test_get_profile_roundtrip_and_missing(tmp_path):
    # A written profile reads back as (coords, temps); a missing one -> None.
    root = _run(tmp_path)
    data = np.array([[0.0, 300.0], [1e-3, 500.0]])
    np.savetxt(root / "profiles" / "x_step000005.txt", data)
    res = SimulationResult(str(root))
    coords, temps = res.get_profile(5, "x")
    np.testing.assert_allclose(coords, data[:, 0])
    np.testing.assert_allclose(temps, data[:, 1])
    assert res.get_profile(99, "x") is None


def test_get_field_roundtrip_and_missing(tmp_path):
    # A saved field h5 reads back with its attrs; a missing step -> None.
    root = _run(tmp_path)
    with h5py.File(root / "fields" / "field_step000002.h5", "w") as f:
        d = f.create_dataset("temperature", data=np.full((2, 2, 2), 7.0))
        d.attrs["time"], d.attrs["step"] = 2e-6, 2
    res = SimulationResult(str(root))
    field = res.get_field(2)
    assert field["temperature"][0, 0, 0] == 7.0
    assert field["step"] == 2
    assert res.get_field(99) is None
