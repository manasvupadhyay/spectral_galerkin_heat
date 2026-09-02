"""Tests for the LaserPath/LaserState contract (core/laser.py)."""

import pytest

from spectral_galerkin_heat.core.laser import LaserPath, LaserState


def test_laserpath_is_abstract():
    # get_state is abstract; neither the base nor a subclass missing it can be built.
    with pytest.raises(TypeError):
        LaserPath()

    class Incomplete(LaserPath):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_laserpath_subclass_returns_state():
    # A concrete subclass satisfying the contract is usable as a LaserPath.
    class ConstantVelocityLaser(LaserPath):
        def get_state(self, time, dt):
            return LaserState(x=0.1 * time, y=0.0, power=200.0, is_on=True, v=(0.1, 0.0))

    s = ConstantVelocityLaser().get_state(2.0, 0.1)
    assert isinstance(s, LaserState)
    assert s.x == pytest.approx(0.2)
    assert s.v == (0.1, 0.0)
