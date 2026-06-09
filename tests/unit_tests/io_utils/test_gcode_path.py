"""Tests for the G-code parser / interpolation (io_utils/gcode_path.py)."""

import numpy as np
import pytest

from fast_heat_solv.io_utils.gcode_path import GCodeLaserPath


def _write(tmp_path, text):
    p = tmp_path / "path.nc"
    p.write_text(text)
    return str(p)


# One on-segment from (0,0) to (10mm,0) at F600 (mm/min) under G21 (mm units).
# speed = 0.6 m/min / 60 = 0.01 m/s; dist = 0.01 m -> the segment spans t in [0, 1] s.
_ONE_SEGMENT = """\
G21
M3 S200
G1 X0 Y0 F600
G1 X10 Y0
"""


@pytest.mark.parametrize("code,scale", [("G21", 1.0 / 1000.0), ("G20", 25.4 / 1000.0)])
def test_units_set_scale(tmp_path, code, scale):
    path = GCodeLaserPath(_write(tmp_path, f"{code}\nM3 S100\nG1 X0 Y0 F600\n"))
    assert path.unit_scale == pytest.approx(scale)


def test_midsegment_interpolation_and_velocity(tmp_path):
    path = GCodeLaserPath(_write(tmp_path, _ONE_SEGMENT))
    dt = 0.1
    s = path.get_state(0.5, dt)  # halfway along [0,1]s -> 5mm
    assert s.is_on is True
    assert float(s.power) == pytest.approx(200.0)
    assert float(s.x) == pytest.approx(0.005, rel=1e-5)
    assert float(s.y) == pytest.approx(0.0, abs=1e-9)
    # moves +x at 10mm / 1s = 0.01 m/s; no y motion.
    assert float(s.v[0]) == pytest.approx(0.01, rel=1e-4)
    assert float(s.v[1]) == pytest.approx(0.0, abs=1e-9)


def test_after_path_holds_final_position_and_off(tmp_path):
    path = GCodeLaserPath(_write(tmp_path, _ONE_SEGMENT))
    s = path.get_state(5.0, 0.1)  # past the last segment's end time (1s)
    assert s.is_on is False
    assert float(s.power) == 0.0
    assert float(s.x) == pytest.approx(0.01, rel=1e-5)  # final x = 10mm


def test_before_path_uses_initial_position_off(tmp_path):
    path = GCodeLaserPath(_write(tmp_path, _ONE_SEGMENT), initial_position=(3.0, 4.0))
    s = path.get_state(-1.0, 0.1)  # before any segment
    assert s.is_on is False
    assert float(s.power) == 0.0
    # initial_position is scaled by unit_scale (G21 -> mm): 3mm, 4mm.
    assert float(s.x) == pytest.approx(0.003, rel=1e-5)
    assert float(s.y) == pytest.approx(0.004, rel=1e-5)
