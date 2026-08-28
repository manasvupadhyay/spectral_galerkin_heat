"""Tests for the HeatSolver abstract contract (solvers/base.py)."""

import pytest

from spectral_galerkin_heat.solvers.base import HeatSolver


def test_heatsolver_is_abstract():
    # initialize/step/finalize are abstract — base and partial subclasses can't build.
    with pytest.raises(TypeError):
        HeatSolver()

    class MissingStep(HeatSolver):
        def initialize(self, context):
            ...

        def finalize(self):
            ...

    with pytest.raises(TypeError):
        MissingStep()
