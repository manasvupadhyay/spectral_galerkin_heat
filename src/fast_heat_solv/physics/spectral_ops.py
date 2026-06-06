"""Backend-agnostic spectral free functions (shared by CPU and GPU kernels).

These operate on a :class:`~fast_heat_solv.physics.spectral_state.SpectralSolverState`
and are pure ``einsum`` / array-copy logic — identical on CPU and GPU once the
array module is taken from ``SsState.xp`` instead of being hard-coded to
``np`` / ``cp``. They used to exist as near-identical copies in
``spectral_cpu_kernels`` and ``spectral_gpu_kernels``; both modules now import
them from here.

Functions that genuinely differ between backends — FFT (``DCT_II`` / ``IDCT_II``),
``scipy``/``cupyx`` ndimage shifts, and the numba ``@njit`` vs ``@cuda.jit``
source-term kernel — stay in the per-backend kernel modules.
"""

# Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique
#
# Author: Théo Andrieux
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


__author__ = "Théo Andrieux"
__copyright__ = "Copyright 2026, LMS, École Polytechnique"

__all__ = [
    "project_box_to_modes",
    "reconstruct_temperature_box",
    "initialize_latent_heat_if_needed",
    "update_latent_heat_history",
]


def project_box_to_modes(field_box, SsState):
    """Project fine box field to global spectral modes."""
    if SsState.fine_mesh is None:
        raise RuntimeError("Fine mesh not initialized.")
    xp = SsState.xp
    fm = SsState.fine_mesh
    modes = xp.einsum('zyx,Zz,Yy,Xx->ZYX', field_box, fm.B_fine[2], fm.B_fine[1], fm.B_fine[0], optimize=True)
    return modes * fm.dV_fine


def reconstruct_temperature_box(a, SsState):
    """Reconstructs temperature in a small ROI around the laser."""
    if SsState.fine_mesh is None:
        raise RuntimeError("Fine mesh not initialized.")
    xp = SsState.xp
    fm = SsState.fine_mesh
    return xp.einsum('ZYX,Zz,Yy,Xx->zyx', a, fm.B_fine[2], fm.B_fine[1], fm.B_fine[0], optimize=True)


def initialize_latent_heat_if_needed(SsState):
    """Initialize fine-mesh T_prev from current trial modes on the first time step.

    Must be called after ``buffers.a_temp`` has been set to a reasonable
    estimate (e.g. the decayed modes).
    """
    fm = SsState.fine_mesh
    if fm is None or fm.T_prev is not None:
        return
    T_box = reconstruct_temperature_box(SsState.buffers.a_temp, SsState)
    fm.T_prev = T_box.copy()
    if fm.Q_prev is None:
        fm.Q_prev = SsState.xp.zeros_like(SsState.buffers.Q_latent_buffer)


def update_latent_heat_history(SsState):
    """Store the converged fine-mesh temperature as T_prev for the next step.

    Call once per time step, after the fixed-point iteration has converged
    and ``buffers.a_temp`` holds the final spectral modes.
    """
    fm = SsState.fine_mesh
    if fm is None:
        return
    T_box = reconstruct_temperature_box(SsState.buffers.a_temp, SsState)
    fm.T_prev[:] = T_box[:]
