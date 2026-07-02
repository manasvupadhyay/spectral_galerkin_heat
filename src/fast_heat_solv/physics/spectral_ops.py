"""Backend-agnostic spectral free functions (shared by CPU and GPU kernels).

These operate on a :class:`~fast_heat_solv.physics.spectral_state.SpectralSolverState`
and are pure ``einsum`` / array-copy logic: they take the array module from
``SsState.xp``, so both kernel modules import them from here.

Operations that differ between backends — FFT (``DCT_II`` / ``IDCT_II``),
``scipy``/``cupyx`` ndimage shifts, and the ``@njit`` vs ``@cuda.jit``
source-term kernel — stay in the per-backend kernel modules.
"""

# Copyright 2026 Laboratoire de Mécanique des Solides (LMS), 
# École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris, 
# Route de Saclay, Palaiseau, 91128, France.
#
# Author: Théo Andrieux, Jules Dichamp, Manas V. Upadhyay
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


__author__ = "Théo Andrieux, Jules Dichamp, Manas V. Upadhyay"
__copyright__ = "Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris"

__all__ = [
    "project_box_to_modes",
    "reconstruct_temperature_box",
    "initialize_latent_heat_if_needed",
    "update_latent_heat_history",
    "reconstruct_surface_temperature",
    "reconstruct_bottom_temperature",
    "compute_latent_heat_source",
    "shift_latent_heat_history",
    "reconstruct_volume",
    "project_volume",
    "assemble_property_correction",
]


def project_box_to_modes(field_box, SsState):
    """Project fine box field to global spectral modes."""
    if SsState.fine_mesh is None:
        raise RuntimeError("Fine mesh not initialized.")
    xp = SsState.xp
    fm = SsState.fine_mesh
    modes = xp.einsum('zyx,Zz,Yy,Xx->ZYX', field_box, fm.B_fine[2], fm.B_fine[1], fm.B_fine[0], optimize=True)
    return modes * fm.dV_fine


# ---------------------------------------------------------------------------
# Global volume transforms for the temperature-dependent property correction
# (property_correction.tex §6). The cell-centred DCT-II / IDCT-II pair below is
# consistent with the modal basis baked into the K, KK propagators — distinct
# from the node-centred DCT-I output helper in ``spectral_helpers``.
# ---------------------------------------------------------------------------

def reconstruct_volume(a, SsState):
    """Reconstruct the full cell-centred temperature field over Ω from modes *a*.

    Inverse of :func:`project_volume`: ``T = IDCT_II(a) / sqrt(dV)``. Returns a
    ``(nz, ny, nx)`` array on the same grid the modes live on.
    """
    xp = SsState.xp
    return (SsState.hooks.idct(a) / SsState.grid.sqrt_dV).astype(xp.float32, copy=False)


def project_volume(field, SsState):
    """Project a full cell-centred volume field onto the modal basis.

    Forward of :func:`reconstruct_volume`: ``modes = sqrt(dV) * DCT_II(field)``.
    """
    xp = SsState.xp
    return (SsState.hooks.dct(field) * SsState.grid.sqrt_dV).astype(xp.float32, copy=False)


def assemble_property_correction(SsState, a_trial, T_prev_full, dt, model,
                                 k_bar, a_bar):
    """Assemble the temperature-dependent property correction modes ``C_mnp``.

    Implements the forcing-assembly recipe of ``property_correction.tex`` §5/§7:
    reconstruct ``T`` over Ω from the current trial modes, read the property
    fluctuations ``k'(T)=k(T)-k̄`` and ``a'(T)=a(T)-ā`` from the tabulated model,
    form the real-space forcing ``f = ∇·(k'∇T) - a'∂_tT`` and project it to modes.
    The projection returns only the **volume** modes; the boundary contribution of
    the conductivity correction is handled by the solver as a rescaling of the
    prescribed surface flux (see below and ``SpectralSolver.step``).

    The conductivity volume term is integrated by parts (Green's first identity):
    it becomes a DCT of ``∇·(k'∇T)``, merged with the capacity source into a
    single DCT of ``f`` — ~4 FFT axis-passes.

    The boundary term ``-∮ k'∂_nT Φ dS`` is NOT assembled here. With the imposed
    Neumann flux ``-k ∂_nT = q`` it equals ``+∮ (k'/k) q Φ dS``, which combines
    with the base boundary forcing ``F^Γ = -∮ q Φ dS`` into a single rescaled-flux
    integral ``-∮ (k̄/k) q Φ dS``. The solver therefore applies the boundary part
    of the correction simply by scaling the prescribed surface flux by
    ``k̄/k(T_surface)`` (see ``SpectralSolver.step``); this is exact (uses the BC,
    not a finite-difference boundary gradient) and needs no face transforms.

    Parameters
    ----------
    a_trial : ndarray (nz, ny, nx)
        Current Picard trial modes.
    T_prev_full : ndarray (nz, ny, nx)
        Previous converged full-volume temperature field (for ∂_t T).
    dt : float
        Time step.
    model : MaterialModel
        Temperature-dependent property model.
    k_bar, a_bar : float
        Reference constants baked into the ETD1 propagators (k̄, ā).

    Returns
    -------
    C : ndarray (nz, ny, nx)
        Correction forcing modes, to be injected as ``a += KK · C``.
    """
    xp = SsState.xp
    grid = SsState.grid

    T = reconstruct_volume(a_trial, SsState)

    # Fluctuations k'(T), a'(T) in one fused pass (≈37× faster than the per-op
    # Horner/blend evaluation on GPU; see MaterialModel.k_prime_a_prime).
    k_prime, a_prime = model.k_prime_a_prime(T, k_bar, a_bar)

    # The real-space forcing f = ∇·(k'∇T) - a'∂_tT is assembled by a fused backend
    # kernel when available (GPU/CPU: two stencil passes replacing six xp.gradient
    # calls); otherwise fall back to xp finite differences.
    corr_source = getattr(SsState.hooks, "corr_source", None)
    if corr_source is not None:
        f = corr_source(T, k_prime, a_prime, T_prev_full, float(dt),
                        grid.dx, grid.dy, grid.dz)
        return project_volume(f, SsState).astype(xp.float32, copy=False)

    dT_dz, dT_dy, dT_dx = xp.gradient(T, grid.dz, grid.dy, grid.dx)
    g_x = k_prime * dT_dx
    g_y = k_prime * dT_dy
    g_z = k_prime * dT_dz
    s_a = -a_prime * ((T - T_prev_full) / xp.float32(dt))
    div_g = (xp.gradient(g_x, grid.dx, axis=2)
             + xp.gradient(g_y, grid.dy, axis=1)
             + xp.gradient(g_z, grid.dz, axis=0))
    return project_volume(s_a + div_g, SsState).astype(xp.float32, copy=False)


def reconstruct_temperature_box(a, SsState):
    """Reconstructs temperature in the refined, laser-following sub-box of the domain."""
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


# ---------------------------------------------------------------------------
# Hook-using free functions: array ops plus a backend primitive
# (FFT / ndimage shift / source-term launch) reached through ``SsState.hooks``
# (see ``spectral_state.BackendHooks``).
# ---------------------------------------------------------------------------

def reconstruct_surface_temperature(a, SsState):
    """Reconstruct the 2D temperature field at the top surface (z = Lz).

    Sums the z-modes weighted by the top-surface Cp coefficients, then applies
    the inverse DCT (via the backend ``idct`` hook).
    """
    xp = SsState.xp
    grid = SsState.grid
    A = xp.einsum('p,pij->ij', grid.Cp32_broadcast[:, 0, 0], a, optimize=True)
    return (grid.recon_scale * SsState.hooks.idct(A)).astype(SsState.dtype)


def reconstruct_bottom_temperature(a, SsState):
    """Reconstruct the 2D temperature field at the bottom surface (z = 0).

    At z=0, ``cos(p*pi*0/Lz) = 1`` so the weighting is plain Cp (no sign
    alternation).
    """
    xp = SsState.xp
    grid = SsState.grid
    A = xp.einsum('p,pij->ij', grid.Cp32_broadcast_bottom[:, 0, 0], a, optimize=True)
    return (grid.recon_scale * SsState.hooks.idct(A)).astype(SsState.dtype)


def compute_latent_heat_source(Q_buffer, phys, num, SsState):
    """Compute volumetric latent heat source Q (W/m^3) on the fine mesh.

    Uses the current trial modes (``buffers.a_temp``) and the stored
    ``T_prev``.  Does not update ``T_prev``; call
    :func:`update_latent_heat_history` after the iteration has converged.

    The latent sink is ``Q = -rho(T) * L_f * (f_l(T) - f_l(T_prev)) / dt`` with
    ``f_l`` clamped to ``[0, 1]`` — the same expression the FE reference uses
    (``Q_latent = rho_eff * L_f * (lf - lf_n)/dt``). For a temperature-dependent
    material the density is the **blended** ``rho(T) = (1-f_l)rho_s(T)+f_l
    rho_l(T)`` evaluated per cell, not the constant reference ``rho(T0)`` baked
    into the propagators — using the reference here over-weights the sink by
    ~rho(T0)/rho(T_melt) ≈ 12 % and cools the pool spuriously. Constant-property
    materials keep the fused scalar-rho kernel (validated, unchanged).
    """
    fm = SsState.fine_mesh
    if fm is None or fm.T_prev is None:
        Q_buffer.fill(0.0)
        return

    T_box = reconstruct_temperature_box(SsState.buffers.a_temp, SsState)

    model = getattr(phys, "model", None)
    if model is not None and not model.is_constant:
        # Blended-rho latent sink for temperature-dependent materials. 
        model.latent_heat_source(
            T_box, fm.T_prev, float(phys.T_solidus), float(phys.T_liquidus),
            float(phys.L_f), float(num.dt), Q_buffer,
        )
        return

    SsState.hooks.source_term(
        T_box, fm.T_prev,
        phys.T_solidus, phys.T_liquidus,
        phys.rho, phys.L_f, num.dt,
        Q_buffer,
    )


def shift_latent_heat_history(SsState, laser_state, num):
    """Shift T_prev and Q_prev to align with the current laser position.

    Call once per time step, before the fixed-point iteration begins.
    """
    fm = SsState.fine_mesh
    if fm is None or fm.T_prev is None:
        return
    shift_x = laser_state.v[0] * num.dt
    shift_y = laser_state.v[1] * num.dt
    shift_pixels = (0, -shift_y / fm.dy_fine, -shift_x / fm.dx_fine)
    ndshift = SsState.hooks.ndshift
    fm.T_prev = ndshift(fm.T_prev, shift_pixels, order=1, mode='nearest', cval=0.0)
    if fm.Q_prev is not None:
        fm.Q_prev = ndshift(fm.Q_prev, shift_pixels, order=1, mode='constant', cval=0.0)
