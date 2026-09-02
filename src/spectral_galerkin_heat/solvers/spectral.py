"""
Unified Spectral Solver (CPU and GPU).
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

import logging
import math
from typing import Any

from spectral_galerkin_heat.backends.base import MathBackend
from spectral_galerkin_heat.core.laser import LaserPath, LaserState, build_laser_profile
from spectral_galerkin_heat.core.parameters import SimulationContext
from spectral_galerkin_heat.solvers.base import HeatSolver

logger = logging.getLogger(__name__)


class SpectralSolver(HeatSolver):
    """
    Spectral method solver for the heat equation with phase change and evaporation.

    Uses fast cosine transforms (DCT) and exponential time differencing (ETD1).
    Runs on CPU (NumPy/Numba) or GPU (CuPy/CUDA) depending on the injected
    :class:`~spectral_galerkin_heat.backends.base.MathBackend`: use
    ``NumpyBackend()`` for CPU and ``get_backend("cupy")`` for GPU.

    Every numerical operation goes through ``self.backend.xp`` (the array
    module) and ``self.backend.kernels`` (the physics kernels), so the solver
    code is identical for both targets.
    """

    def __init__(
        self,
        backend: MathBackend,
        context: SimulationContext | None = None,
        *,
        mixing_omega: float | None = None,
        convergence_tol: float | None = None,
        max_picard_iter: int | None = None,
    ):
        """
        Initialize the SpectralSolver.

        Parameters
        ----------
        backend : MathBackend
            The math backend selecting CPU (NumPy) or GPU (CuPy) execution.
        context : SimulationContext, optional
            A dataclass containing complete simulation parameters. May also be
            provided (or overridden) later via :meth:`initialize`. By default
            None.
        mixing_omega, convergence_tol, max_picard_iter : float/int, optional
            Picard fixed-point controls (under-relaxation factor, relative
            convergence tolerance, iteration cap). 
        """
        self.backend: MathBackend = backend
        self.context: SimulationContext | None = context
        self.state = None
        # Picard fixed-point controls. 
        #   explicit constructor arg  >  config (NumParams)  >  built-in default.

        self._omega_req = mixing_omega
        self._tol_req = convergence_tol
        self._maxit_req = max_picard_iter
        # Re-resolved in initialize() once the configured precision is known.
        self._resolve_picard_params(backend.xp.float32)
        self.track_picard_history: bool = False
        self.picard_history = []
        # Temperature-dependent property correction.
        # Enabled in ``initialize`` when the material carries a non-constant model.
        self._property_correction: bool = False
        # Grid mode: no fine sub-box, latent heat evaluated on the coarse grid
        # (set in ``initialize`` from ``fine is None`` and ``L_f > 0``).
        self._grid_latent: bool = False
        self._T_prev_full = None   # previous converged full-volume field (∂_t T; shared
                                   # by the property correction and grid-mode latent heat)
        self._kbar = None          # reference conductivity k̄ (= propagator const)
        self._abar = None          # reference volumetric capacity ā
        self._latent_truncation_warned = False  # fine-box truncation warned once
        self._picard_cap_warned = False
        self._Q_latent_modes_prev = None

        # Beam profile (shape + energy-conserving normalization); resolved from
        # the config in initialize(). Default keeps a usable solver before then.
        self._profile = build_laser_profile("gaussian")

    def _resolve_picard_params(self, dtype) -> None:
        """
        Shared by the constructor (values available before ``initialize``) and
        :meth:`initialize`
        """
        num = self.context.num if self.context is not None else None

        def pick(explicit, cfg_attr, default):
            if explicit is not None:
                return explicit
            if num is not None and getattr(num, cfg_attr, None) is not None:
                return getattr(num, cfg_attr)
            return default

        self.mixing_omega = dtype(pick(self._omega_req, "picard_omega", 0.1))
        self.convergence_tol = dtype(pick(self._tol_req, "picard_tol", 1e-4))
        self.max_picard_iter = int(pick(self._maxit_req, "max_picard_iter", 30))

    def initialize(self, context: SimulationContext | None = None) -> Any:
        """
        Set up the spectral solver state, allocate buffers, and set the initial condition.

        Parameters
        ----------
        context : SimulationContext, optional
            If provided, replaces the stored SimulationContext. By default None.

        Returns
        -------
        Any
            The initialized solver state object (backend-specific
            ``SpectralSolverState``) containing grid buffers and spectra.

        Raises
        ------
        RuntimeError
            If SimulationContext is neither provided here nor at construction.
        """
        if context is not None:
            self.context = context
        if self.context is None:
            raise RuntimeError(
                "SimulationContext must be provided either at construction "
                "or in initialize()."
            )

        xp = self.backend.xp
        kernels = self.backend.kernels
        geom = self.context.geom
        num = self.context.num
        mat = self.context.mat

        # Initialize spectral solver state
        self.state = kernels.SpectralSolverState(mat, geom, num, self.context.fine)
        dtype = self.state.dtype

        # Resolve the beam profile (shape + normalization) from the config.
        laser = self.context.laser
        self._profile = build_laser_profile(
            laser.profile, laser.super_gaussian_order,
            cell_integrated=laser.cell_integrated,
        )

        # Mixing/convergence/iteration-cap scalars at the configured precision,
        # with optional overrides from the config (simulation block, parsed
        # onto NumParams). Absent keys keep the solver defaults.
        self._resolve_picard_params(dtype)

        # Initial condition: mean T in mode (0,0,0)
        self.state.a = xp.zeros((num.nz, num.ny, num.nx), dtype=dtype)
        T0 = mat.T0
        self.state.a[0, 0, 0] = dtype(
            T0 * math.sqrt(geom.size.x * geom.size.y * geom.size.z)
        )

        # Grid mode: no fine sub-box configured -> evaluate latent heat on the
        # coarse grid (lower memory / faster for small runs). Only needed when
        # latent heat is active (L_f > 0).
        self._grid_latent = (self.state.fine_mesh is None
                             and float(getattr(mat, "L_f", 0.0)) > 0.0)

        # --- Temperature-dependent property correction setup ----------------
        model = getattr(mat, "model", None)
        self._property_correction = model is not None and not model.is_constant
        if self._property_correction:
            # The reference constants k̄, ā used in the fluctuations
            # k'=k(T)-k̄ and a'=a(T)-ā must equal the constants used in the propagators.
            self._kbar = dtype(float(mat.k))
            self._abar = dtype(float(mat.rho) * float(mat.Cp))


        # Previous converged full-volume field
        if self._property_correction or self._grid_latent:
            self._T_prev_full = xp.full(
                (num.nz, num.ny, num.nx), dtype(T0), dtype=dtype
            )
        # No latent history yet on the first step, in either mode.
        self._Q_latent_modes_prev = None

        # Guardrail: in grid mode the sharp mushy-zone latent source is projected
        # by the full-volume DCT, which on a coarse grid can create ringing; 
        # combined with the property correction, it can destabilise 
        # the Picard iteration.
        if self._property_correction and self._grid_latent:
            logger.warning(
                "Temperature-dependent run in grid mode (no fine_mesh): the sharp "
                "latent-heat source is projected on the coarse grid and can ring. "
                " Add a fine_mesh section (box mode) if you see NaNs or ringing."
            )

        return self.state

    def step(self, t: float, dt: float) -> tuple[Any, dict[str, float]]:
        """
        Advance the spectral solution by one time step using ETD1 and nonlinear evaporation correction.

        Handles laser source, latent heat, and evaporation effects.

        Parameters
        ----------
        t : float
            Current simulation time.
        dt : float
            Time step size.

        Returns
        -------
        tuple
            SsState : Any
                The updated solver state.
            metrics : dict
                Metrics from the iteration, such as max temperature and number
                of evaporation steps.
        """
        xp = self.backend.xp
        kernels = self.backend.kernels

        # --- Unpack context ---
        context = self.context
        geom = context.geom
        num = context.num
        mat = context.mat
        laser_path: LaserPath = context.laser_path
        laser_params = context.laser
        SsState = self.state
        grid = SsState.grid
        buffers = SsState.buffers
        fm = SsState.fine_mesh
        dtype = SsState.dtype  # configured float precision (float32 / float64)

        # --- Laser state ---
        laser_state: LaserState = laser_path.get_state(t, dt)
        power = laser_state.power
        is_on = laser_state.is_on
        v_x, v_y = laser_state.v
        absorptivity = laser_params.absorptivity
        # Energy-conserving peak intensity I₀ = A·P / (f · r_x · r_y); the
        # profile owns both the f-normalization and the spatial shape below.
        laser_coef = (
            self._profile.peak_intensity(power, absorptivity, laser_params.ref_area)
            if is_on else 0.0
        )

        # ================================================================
        # 1. Compute laser flux (constant – does not depend on T)
        # ================================================================
        q_las = self._profile.flux(
            xp, grid.x, grid.y, laser_state.x, laser_state.y,
            laser_params.r_x, laser_params.r_y, laser_coef,
            hx=geom.d.x, hy=geom.d.y,
        )
        P_laser = xp.sum(q_las) * geom.d.x * geom.d.y

        # ================================================================
        # 2. Apply exponential propagator:  θ̃ = E · θ
        # ================================================================
        xp.multiply(SsState.a, SsState.K, out=SsState.a)

        # ================================================================
        # 3. Prepare latent-heat history (once per step)
        # ================================================================
        # Box mode only: reposition the fine sub-box and shift its history to
        # follow the laser. In grid mode (fm is None) these are no-ops.
        if fm is not None:
            fm.update(laser_state)
        buffers.a_temp[:] = SsState.a
        kernels.initialize_latent_heat_if_needed(SsState)
        kernels.shift_latent_heat_history(SsState, laser_state, num)

        if self.track_picard_history:
            self.picard_history = []

        # ================================================================
        # 4. Initial guesses for forcing components
        # ================================================================
        # Top surface: warm-start with shifted previous evaporation
        q_evap_shifted = kernels.shift_flux(
            buffers.q_evap_old, (v_x * dt, v_y * dt), geom
        )
        S_top = grid.dct_scale * kernels.DCT_II(q_las - q_evap_shifted)

        # Latent heat: box mode warm-starts with the shifted Q from the previous
        # step; grid mode recomputes it each iteration (no stored box history).
        Q_latent = fm.Q_prev if fm is not None else None

        # Convection at either face, independently switched by h_conv_top /
        # h_conv_bottom (0.0 disables that face, same gating as evaporation).
        h_conv_top = mat.h_conv_top
        h_conv_bottom = mat.h_conv_bottom
        T0 = dtype(mat.T0)
        S_bot = xp.zeros((num.ny, num.nx), dtype=dtype) if h_conv_bottom > 0 else None

        # Build initial a_temp = θ̃ + Q_mnp · F  with all forcing guesses
        kernels.update_modes_etd1(
            SsState.a, SsState.KK, grid.Cp32_broadcast, S_top, buffers.a_temp
        )
        if fm is not None and fm.T_prev is not None and Q_latent is not None:
            kernels.add_source_term_modes(
                buffers.a_temp, SsState.KK,
                kernels.project_box_to_modes(Q_latent, SsState),
            )
        elif self._grid_latent and self._Q_latent_modes_prev is not None:
            # Grid-mode counterpart: replay the previous step's converged latent
            # modes. Stored already projected (no moving window to re-shift), so
            # this costs an axpy instead of box mode's re-projection.
            kernels.add_source_term_modes(
                buffers.a_temp, SsState.KK, self._Q_latent_modes_prev,
            )
        if h_conv_top > 0:
            # Same face as the laser/evaporation flux. add_bottom_surface_source
            # is a generic Cp-weighted 2D-surface accumulator (nothing
            # bottom-specific in the kernel itself) -- pass the top Cp
            # weighting to add a convective loss at z=Lz instead of z=0.
            T_surface0 = kernels.reconstruct_surface_temperature(buffers.a_temp, SsState)
            q_conv_top0 = dtype(-h_conv_top) * (T_surface0 - T0)
            S_top_conv = grid.dct_scale * kernels.DCT_II(q_conv_top0)
            kernels.add_bottom_surface_source(
                buffers.a_temp, SsState.KK, grid.Cp32_broadcast, S_top_conv
            )
        if h_conv_bottom > 0:
            T_bottom = kernels.reconstruct_bottom_temperature(buffers.a_temp, SsState)
            q_conv = dtype(-h_conv_bottom) * (T_bottom - T0)
            S_bot[:] = grid.dct_scale * kernels.DCT_II(q_conv)
            kernels.add_bottom_surface_source(
                buffers.a_temp, SsState.KK, grid.Cp32_broadcast_bottom, S_bot
            )

        # ================================================================
        # Reuse the Picard scratch buffers allocated once in SolverBuffers
        # ================================================================
        a_old = buffers.a_old
        residual_curr = buffers.residual_curr
        n_elements = buffers.n_elements

        # ================================================================
        # Hoist linear/constant forcing terms
        # ================================================================
        S_las = grid.dct_scale * kernels.DCT_II(q_las)

        # Divergence-mode property correction handles the conductivity boundary
        # term by rescaling the prescribed surface flux by k̄/k(T_surface): using
        # the Neumann BC (-k ∂_nT = q), the boundary piece -∮k'∂_nT Φ dS merges
        # with F^Γ = -∮qΦ dS into -∮(k̄/k)qΦ dS. This is
        # exact and replaces the finite-difference face term.
        rescale_flux = self._property_correction

        S_bot_raw = None
        if h_conv_bottom > 0:
            T_bottom = kernels.reconstruct_bottom_temperature(buffers.a_temp, SsState)
            q_conv = dtype(-h_conv_bottom) * (T_bottom - T0)
            if rescale_flux:
                q_conv = q_conv * (self._kbar / mat.model.k(T_bottom))
            S_bot_raw = grid.dct_scale * kernels.DCT_II(q_conv)

        # ================================================================
        # 5. Fixed-point (Picard) iteration
        # ================================================================
        iter_k, true_rel_err, picard_converged = -1, float("nan"), False
        Q_latent_raw = Q_latent_modes = None   # defined even if the cap is 0
        for iter_k in range(self.max_picard_iter):
            xp.copyto(a_old, buffers.a_temp)

            T_surface = kernels.reconstruct_surface_temperature(a_old, SsState)
            kernels.compute_evaporation_flux(
                T_surface, buffers.q_evap_buffer,
                mat.Pa, mat.T_boil, mat.DeltaH_LV, mat.R_v, mat.T_liquidus,
            )
            # Top convection is just another top-face flux term, recomputed
            # from the current iterate exactly like evaporation.
            q_conv_top = (dtype(h_conv_top) * (T_surface - T0)) if h_conv_top > 0 else dtype(0.0)
            if rescale_flux:
                # Exact Neumann-BC boundary correction: scale the net top flux by
                # k̄/k(T_surface) before projecting (folds -∮k'∂_nT Φ dS into F^Γ).
                q_top = ((q_las - buffers.q_evap_buffer - q_conv_top)
                         * (self._kbar / mat.model.k(T_surface)))
                S_top_raw = grid.dct_scale * kernels.DCT_II(q_top)
            else:
                S_evap = grid.dct_scale * kernels.DCT_II(buffers.q_evap_buffer)
                S_top_raw = S_las - S_evap
                if h_conv_top > 0:
                    S_top_raw = S_top_raw - grid.dct_scale * kernels.DCT_II(q_conv_top)

            # Latent-heat source at the current iterate a_old, projected to modes.
            # Box mode: reconstruct on the fine sub-box and contract against the
            # box basis. Grid mode: reconstruct on the coarse grid and project by
            # the full-volume DCT (reuses the previous converged _T_prev_full).
            Q_latent_raw = None
            Q_latent_modes = None
            if fm is not None and fm.T_prev is not None:
                xp.copyto(buffers.a_temp, a_old)
                buffers.Q_latent_buffer.fill(0.0)
                kernels.compute_latent_heat_source(
                    buffers.Q_latent_buffer, mat, num, SsState
                )
                Q_latent_raw = buffers.Q_latent_buffer
                Q_latent_modes = kernels.project_box_to_modes(Q_latent_raw, SsState)
            elif self._grid_latent:
                T_full = kernels.reconstruct_volume(a_old, SsState)
                buffers.Q_latent_buffer.fill(0.0)
                kernels.compute_latent_heat_source_grid(
                    buffers.Q_latent_buffer, T_full, self._T_prev_full, mat, num, SsState
                )
                Q_latent_modes = kernels.project_volume(buffers.Q_latent_buffer, SsState)

            kernels.update_modes_etd1(
                SsState.a, SsState.KK, grid.Cp32_broadcast,
                S_top_raw, buffers.a_temp,
            )
            if Q_latent_modes is not None:
                kernels.add_source_term_modes(
                    buffers.a_temp, SsState.KK, Q_latent_modes,
                )
            if S_bot_raw is not None:
                kernels.add_bottom_surface_source(
                    buffers.a_temp, SsState.KK,
                    grid.Cp32_broadcast_bottom, S_bot_raw,
                )

            # Temperature-dependent property correction: re-evaluated
            # from the current iterate, folded into the forcing before relaxation
            # so the fixed point resums the perturbation series to all orders.
            if self._property_correction:
                C_corr = kernels.assemble_property_correction(
                    SsState, buffers.a_temp, self._T_prev_full,
                    num.dt, mat.model, self._kbar, self._abar,
                )
                kernels.add_source_term_modes(buffers.a_temp, SsState.KK, C_corr)

            # Under-relax in residual form: ω·a_raw + (1-ω)·a_old is identically
            # a_old + ω·(a_raw - a_old), and the residual is already needed for the
            # convergence test. Written this way the iteration needs neither a
            # verbatim copy of a_temp nor a full-grid temporary for (1-ω)·a_old.
            omega = self.mixing_omega
            xp.subtract(buffers.a_temp, a_old, out=residual_curr)
            xp.multiply(residual_curr, omega, out=buffers.a_temp)
            buffers.a_temp += a_old

            rms_diff = xp.sqrt(xp.vdot(residual_curr, residual_curr) / n_elements)
            rms_old = xp.sqrt(xp.vdot(a_old, a_old) / n_elements)
            # Form the ratio on the device so the convergence check costs a single
            # host sync per Picard iteration instead of two.
            true_rel_err = float(rms_diff / xp.maximum(rms_old, xp.asarray(1e-9, dtype=rms_old.dtype)))

            if self.track_picard_history:
                rho_k = None
                if iter_k > 0 and self.picard_history:
                    prev_rms = self.picard_history[-1]["rms_diff"]
                    if prev_rms > 0:
                        rho_k = float(rms_diff) / prev_rms
                self.picard_history.append({
                    "iter": iter_k,
                    "rms_diff": float(rms_diff),
                    "rho": rho_k,
                    "true_rel_err": float(true_rel_err),
                })

            if true_rel_err < float(self.convergence_tol):
                picard_converged = True
                break
        else:
            # Fell through the cap without meeting the tolerance. The step still
            # returns a field, but it is not the fixed point, so say so instead
            # of letting a wrong answer look like a converged one.
            picard_converged = False
            if not self._picard_cap_warned:
                self._picard_cap_warned = True
                logger.warning(
                    "Picard iteration hit max_picard_iter=%d at t=%.6g s without "
                    "reaching picard_tol=%g (last relative residual %.3g). The "
                    "returned field is not converged. Raise max_picard_iter, lower "
                    "picard_omega, or shorten dt. Warned once per run.",
                    self.max_picard_iter, t, float(self.convergence_tol), true_rel_err,
                )

        # Restore variables needed below
        T_temp = kernels.reconstruct_surface_temperature(buffers.a_temp, SsState)
        Q_latent = Q_latent_raw
        # Carry the latent modes into the next step's initial guess (grid mode;
        # box mode carries the real-space Q_prev instead, which it must re-shift).
        if self._grid_latent:
            self._Q_latent_modes_prev = Q_latent_modes

        # ================================================================
        # 6. Commit converged state
        # ================================================================
        buffers.q_evap_old[:] = buffers.q_evap_buffer
        # Copy into the existing modes array rather than rebinding to a fresh one:
        # `.copy()` allocated a full-grid array every step and orphaned the old.
        # a_temp and SsState.a are distinct buffers, so an in-place copy is safe.
        xp.copyto(SsState.a, buffers.a_temp)

        # Store the converged full-volume field for the next step's ∂_t T (used
        # by both the property correction and grid-mode latent heat).
        if self._property_correction or self._grid_latent:
            self._T_prev_full[:] = kernels.reconstruct_volume(SsState.a, SsState)

        # ================================================================
        # 7. Update latent-heat history with converged temperature
        # ================================================================
        # Box mode keeps its own moving-window history; grid mode reuses
        # _T_prev_full (updated above), so nothing more to do here.
        if fm is not None:
            kernels.update_latent_heat_history(SsState)
            if fm.Q_prev is None:
                fm.Q_prev = xp.zeros_like(buffers.Q_latent_buffer)
            if Q_latent is not None:
                fm.Q_prev[:] = Q_latent[:]
                self._check_latent_box_truncation(Q_latent, fm)

        metrics = {
            'T_surface_max': xp.max(T_temp),
            'P_laser': P_laser,
            'n_evap_iter': iter_k + 1,
            'picard_converged': picard_converged,
            'picard_rel_err': true_rel_err,
        }
        return SsState, metrics

    def _check_latent_box_truncation(self, Q_latent, fm) -> None:
        """Warn (once per run) if the latent-heat source reaches a boundary of
        the refined sub-box.

        The latent-heat source is projected onto the modal basis only from
        within the laser-following fine box (``project_box_to_modes``). If the
        melt front reaches the box floor (the deepest fine layer, at depth
        ``Lz - Lz_box``) or a moving x/y window edge, the latent heat beyond it
        is silently dropped. A non-negligible source on such a face signals that
        the box no longer encloses the pool and ``fine_mesh.box_size`` should be
        enlarged. Faces clamped to the domain boundary are not flagged (there is
        nothing beyond them to truncate).
        """
        if self._latent_truncation_warned or Q_latent is None:
            return
        xp = self.backend.xp
        peak = float(xp.max(xp.abs(Q_latent)))
        if peak <= 0.0:
            return
        rel = 1e-2  # fraction of the peak source treated as non-negligible

        faces = {"floor (depth)": Q_latent[0, :, :]}            # z: always the box floor
        if fm.nx_box < fm.n_fine_totals[0]:                     # x: real moving window
            faces["x window edge"] = Q_latent[:, :, (0, -1)]
        if fm.ny_box < fm.n_fine_totals[1]:                     # y: real moving window
            faces["y window edge"] = Q_latent[:, (0, -1), :]

        hit = [name for name, face in faces.items()
               if float(xp.max(xp.abs(face))) > rel * peak]
        if not hit:
            return
        box_mm = (fm.nx_box * fm.dx_fine * 1e3,
                  fm.ny_box * fm.dy_fine * 1e3,
                  fm.nz_box * fm.dz_fine * 1e3)
        logger.warning(
            "Latent-heat source reaches the fine-mesh box boundary (%s): the melt "
            "pool extends beyond the refined sub-box, so its latent heat is "
            "truncated. Enlarge fine_mesh.box_size (currently %.3g x %.3g x %.3g mm) "
            "to enclose the pool.",
            ", ".join(hit), *box_mm,
        )
        self._latent_truncation_warned = True

    def set_state(self, temperature_field) -> None:
        """
        Overwrite the internal spectral coefficients from a spatial temperature field.

        Converts the cell-centred temperature array into spectral (DCT) modes
        so that the solver can continue stepping from the injected state.

        Parameters
        ----------
        temperature_field : ndarray
            A 3-D array (shape ``(N_z, N_y, N_x)``) containing temperatures.
            May be a NumPy or CuPy array.

        Raises
        ------
        RuntimeError
            If called before the solver is initialized.
        """
        if self.state is None or self.context is None:
            raise RuntimeError(
                "Solver must be initialized before calling set_state()."
            )
        xp = self.backend.xp
        kernels = self.backend.kernels
        geom = self.context.geom
        dtype = self.state.dtype
        T = xp.asarray(temperature_field, dtype=dtype)
        # Forward DCT-II (ortho) converts the spatial field to ortho-normalised
        # coefficients. The solver's internal modes use a scaling of
        # sqrt(dx*dy*dz) relative to the standard ortho DCT coefficients.
        scale = dtype(math.sqrt(float(geom.d.x * geom.d.y * geom.d.z)))
        self.state.a = kernels.DCT_II(T) * scale

    def finalize(self) -> None:
        """Release nothing and return.

        The spectral solver holds no resources requiring explicit release
        (CuPy frees device memory on garbage collection). This override is
        present only to satisfy the abstract :meth:`HeatSolver.finalize`
        interface so the class is instantiable.
        """
