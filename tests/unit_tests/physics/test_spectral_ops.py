"""Tests for shared spectral free functions (physics/spectral_ops.py)."""

import numpy as np
import pytest

from spectral_galerkin_heat.physics import spectral_cpu_kernels as k


@pytest.fixture
def state(tiny_context):
    # Building the state is pure NumPy (einsum/precompute). These tests exercise
    # the fine-box path, so build an explicit box (the default config is now
    # grid mode, i.e. fine is None).
    from spectral_galerkin_heat.core.parameters import FineMeshParams
    c = tiny_context
    return k.SpectralSolverState(c.mat, c.geom, c.num, FineMeshParams())


def test_project_box_requires_fine_mesh(state):
    # Projecting onto the moving fine box is meaningless without one -> RuntimeError.
    state.fine_mesh = None
    with pytest.raises(RuntimeError):
        k.project_box_to_modes(np.zeros((1, 1, 1), dtype=np.float32), state)


def test_reconstruct_box_requires_fine_mesh(state):
    # Same guard for the inverse (box reconstruction).
    state.fine_mesh = None
    with pytest.raises(RuntimeError):
        k._reconstruct_temperature_box(state.buffers.a_temp, state)


def test_initialize_latent_heat_sets_tprev_once(state):
    # T_prev is the fine box's temperature from the *previous* time step. The
    # latent-heat source needs it because Q ∝ (T_curr - T_prev) across the mushy
    # zone — a dT/dt term. On the very first step there is no previous field, so
    # this seeds T_prev from the current trial modes, and only once: a later
    # step's bookkeeping must not overwrite the seeded history.
    fm = state.fine_mesh
    assert fm.T_prev is None
    state.buffers.a_temp[:] = 0.0
    state.buffers.a_temp[0, 0, 0] = 1.0

    k.initialize_latent_heat_if_needed(state)
    assert fm.T_prev is not None
    assert fm.Q_prev is not None
    first = fm.T_prev

    # Second call is a no-op: T_prev is already set, so it is not recomputed.
    state.buffers.a_temp[0, 0, 0] = 999.0
    k.initialize_latent_heat_if_needed(state)
    assert fm.T_prev is first
