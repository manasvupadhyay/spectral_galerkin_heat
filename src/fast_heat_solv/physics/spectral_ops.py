"""Backend-agnostic spectral free functions (shared by CPU and GPU kernels).

These operate on a :class:`~fast_heat_solv.physics.spectral_state.SpectralSolverState`
and are pure ``einsum`` / array-copy logic: they take the array module from
``SsState.xp``, so both kernel modules import them from here.

Operations that differ between backends — FFT (``DCT_II`` / ``IDCT_II``),
``scipy``/``cupyx`` ndimage shifts, and the ``@njit`` vs ``@cuda.jit``
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
    "reconstruct_surface_temperature",
    "reconstruct_bottom_temperature",
    "compute_latent_heat_source",
    "shift_latent_heat_history",
    "reconstruct_volume",
    "project_volume",
    "conductivity_correction_modes",
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


def _mixed_sine_transform(g, axis, SsState):
    """Project *g* onto the basis differentiated along ``axis``.

    Computes ``S_d{g}`` (``property_correction.tex`` eq. Ck): a DST-II along
    ``axis`` and a DCT-II along the other two, ortho, scaled by ``sqrt(dV)``,
    with the DST-II mode-index shift applied — output cosine-mode ``m`` reads
    ``DST[m-1]`` and ``m = 0`` is set to zero (``∂`` annihilates the constant
    mode). The highest DST mode (the orthonormal special case) maps to ``m = N``,
    which is outside the kept mode range and is simply dropped.
    """
    xp = SsState.xp
    hooks = SsState.hooks
    out = g
    for ax in range(3):
        out = hooks.dst_axis(out, ax) if ax == axis else hooks.dct_axis(out, ax)

    shifted = xp.zeros_like(out)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    src[axis] = slice(0, -1)   # DST indices 0 .. N-2  (frequencies 1 .. N-1)
    dst[axis] = slice(1, None)  # cosine modes  1 .. N-1
    shifted[tuple(dst)] = out[tuple(src)]
    return shifted * SsState.grid.sqrt_dV


def conductivity_correction_modes(g_x, g_y, g_z, SsState):
    """Assemble the conductivity correction modes ``C^k_mnp`` from ``g = k' ∇T``.

    ``C^k = (mπ/Lx) S_x{g_x} + (nπ/Ly) S_y{g_y} + (pπ/Lz) S_z{g_z}`` — three
    mixed transforms scaled by their modal multipliers (axis order: x=2, y=1,
    z=0 in the ``(nz, ny, nx)`` layout).
    """
    grid = SsState.grid
    Sx = _mixed_sine_transform(g_x, 2, SsState)
    Sy = _mixed_sine_transform(g_y, 1, SsState)
    Sz = _mixed_sine_transform(g_z, 0, SsState)
    return (grid.kx[None, None, :] * Sx
            + grid.ky[None, :, None] * Sy
            + grid.kz[:, None, None] * Sz)


def assemble_property_correction(SsState, a_trial, T_prev_full, dt, model,
                                 k_bar, a_bar, mode="mixed"):
    """Assemble the temperature-dependent property correction modes ``C_mnp``.

    Implements the forcing-assembly recipe of ``property_correction.tex`` §5/§7:
    reconstruct ``T`` over Ω from the current trial modes, read the property
    fluctuations ``k'(T)=k(T)-k̄`` and ``a'(T)=a(T)-ā`` from the tabulated model,
    form ``g = k' ∇T`` (finite-difference gradient) and ``s_a = -a' ∂_t T``, then
    project. Both projections return only the **volume** modes; the boundary
    contribution of the conductivity correction is handled by the solver as a
    rescaling of the prescribed surface flux (see the divergence branch below and
    ``SpectralSolver.step``). Two projections are available:

    - ``mode="mixed"`` (reference): the weak form of ``property_correction.tex``
      Eq.(Ck), three mixed sine/cosine transforms for ``C^k`` plus one DCT for
      ``C^a`` — 15 full-volume FFT axis-passes. The sine basis vanishes on the
      faces, so the boundary content is carried entirely by the (unscaled) base
      forcing ``F^Γ``.
    - ``mode="divergence"`` (``property_correction.tex``): integrate ``C^k`` by
      parts so the volume term merges with ``C^a`` into a single DCT of
      ``f = -a'∂_tT + ∇·(k'∇T)`` — ~4 passes. The boundary term it generates is
      not assembled here; using the Neumann BC it merges with ``F^Γ`` into a
      single rescaled-flux integral ``-∮ (k̄/k) q Φ dS`` applied in the solver.

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
    mode : {"mixed", "divergence"}
        Projection scheme (default "mixed", the validated reference).

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

    if mode == "divergence":
        # Move the derivative off Φ (property_correction.tex eq. Ckdiv): the conductivity
        # volume term becomes a DCT of ∇·(k'∇T), merged with the capacity source.
        #
        # The boundary term -∮ k'∂_nT Φ dS is NOT assembled here. With the imposed
        # Neumann flux -k ∂_nT = q it equals +∮ (k'/k) q Φ dS, which combines with
        # the base boundary forcing F^Γ = -∮ q Φ dS into a single rescaled-flux
        # integral  -∮ (k̄/k) q Φ dS . The solver therefore applies the property
        # correction at the boundary simply by scaling the prescribed surface flux
        # by k̄/k(T_surface) (see SpectralSolver.step); this is exact (uses the BC,
        # not a finite-difference boundary gradient) and needs no face transforms.
        #
        # The real-space forcing f = ∇·(k'∇T) - a'∂_tT is assembled by a fused
        # backend kernel when available (GPU: two stencil passes replacing six
        # xp.gradient calls); otherwise fall back to xp finite differences.
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

    # Gradient by central differences on the reconstructed field (cheaper default
    # per tex §6.3); xp.gradient returns [∂z, ∂y, ∂x] for the (nz, ny, nx) layout.
    dT_dz, dT_dy, dT_dx = xp.gradient(T, grid.dz, grid.dy, grid.dx)
    g_x = k_prime * dT_dx
    g_y = k_prime * dT_dy
    g_z = k_prime * dT_dz

    dT_dt = (T - T_prev_full) / xp.float32(dt)
    s_a = -a_prime * dT_dt

    C_a = project_volume(s_a, SsState)
    C_k = conductivity_correction_modes(g_x, g_y, g_z, SsState)
    return (C_a + C_k).astype(xp.float32, copy=False)


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
