"""
Unified Spectral Solver (CPU and GPU).
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

import math
from typing import Optional, Any, Tuple, Dict

from fast_heat_solv.backends.base import MathBackend
from fast_heat_solv.core.parameters import SimulationContext
from fast_heat_solv.core.laser import LaserState, LaserPath
from fast_heat_solv.solvers.base import HeatSolver


class SpectralSolver(HeatSolver):
    """
    Spectral method solver for the heat equation with phase change and evaporation.

    Uses fast cosine transforms (DCT) and exponential time differencing (ETD1).
    Runs on CPU (NumPy/Numba) or GPU (CuPy/CUDA) depending on the injected
    :class:`~fast_heat_solv.backends.base.MathBackend`: use
    ``NumpyBackend()`` for CPU and ``get_backend("cupy")`` for GPU.

    Every numerical operation goes through ``self.backend.xp`` (the array
    module) and ``self.backend.kernels`` (the physics kernels), so the solver
    code is identical for both targets.
    """

    def __init__(
        self,
        backend: MathBackend,
        context: Optional[SimulationContext] = None,
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
        """
        self.backend: MathBackend = backend
        self.context: Optional[SimulationContext] = context
        self.state = None
        xp = backend.xp
        self.mixing_omega = xp.float32(0.1)
        self.convergence_tol = xp.float32(1e-4)
        self.max_picard_iter: int = 30
        self.track_picard_history: bool = False
        self.picard_history = []

    def initialize(self, context: Optional[SimulationContext] = None) -> Any:
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

        # Initial condition: mean T in mode (0,0,0)
        self.state.a = xp.zeros((num.nz, num.ny, num.nx), dtype=xp.float32)
        T0 = mat.T0
        self.state.a[0, 0, 0] = xp.float32(
            T0 * math.sqrt(geom.size.x * geom.size.y * geom.size.z)
        )

        return self.state

    def step(self, t: float, dt: float) -> Tuple[Any, Dict[str, float]]:
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

        # --- Laser state ---
        laser_state: LaserState = laser_path.get_state(t, dt)
        power = laser_state.power
        is_on = laser_state.is_on
        v_x, v_y = laser_state.v
        absorptivity = laser_params.absorptivity
        laser_coef = (
            absorptivity * 2.0 * power / (math.pi * laser_params.radius ** 2)
            if is_on else 0.0
        )

        # ================================================================
        # 1. Compute laser flux (constant – does not depend on T)
        # ================================================================
        q_las = kernels.compute_gaussian_laser_flux(
            grid.x, grid.y, laser_state.x, laser_state.y,
            laser_params.radius, laser_coef,
        )
        P_laser = xp.sum(q_las) * geom.d.x * geom.d.y

        # ================================================================
        # 2. Apply exponential propagator:  θ̃ = E · θ
        # ================================================================
        xp.multiply(SsState.a, SsState.K, out=SsState.a)

        # ================================================================
        # 3. Prepare latent-heat history (once per step)
        # ================================================================
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

        # Latent heat: warm-start with shifted Q from previous step
        Q_latent = xp.zeros_like(buffers.Q_latent_buffer)
        if fm.Q_prev is not None:
            Q_latent[:] = fm.Q_prev

        # Bottom convection
        h_conv = mat.h_conv
        T0 = xp.float32(mat.T0)
        S_bot = xp.zeros((num.ny, num.nx), dtype=xp.float32) if h_conv > 0 else None

        # Build initial a_temp = θ̃ + Q_mnp · F  with all forcing guesses
        kernels.update_modes_etd1(
            SsState.a, SsState.KK, grid.Cp32_broadcast, S_top, buffers.a_temp
        )
        if fm.T_prev is not None and Q_latent is not None:
            kernels.add_source_term_modes(
                buffers.a_temp, SsState.KK,
                kernels.project_box_to_modes(Q_latent, SsState),
            )
        if h_conv > 0:
            T_bottom = kernels.reconstruct_bottom_temperature(buffers.a_temp, SsState)
            q_conv = xp.float32(-h_conv) * (T_bottom - T0)
            S_bot[:] = grid.dct_scale * kernels.DCT_II(q_conv)
            kernels.add_bottom_surface_source(
                buffers.a_temp, SsState.KK, grid.Cp32_broadcast_bottom, S_bot
            )

        # ================================================================
        # Pre-allocate arrays to avoid per-iteration allocations
        # ================================================================
        a_old = xp.empty_like(buffers.a_temp)
        a_raw = xp.empty_like(buffers.a_temp)
        residual_curr = xp.empty_like(buffers.a_temp)
        n_elements = xp.float32(buffers.a_temp.size)

        # ================================================================
        # Hoist linear/constant forcing terms
        # ================================================================
        S_las = grid.dct_scale * kernels.DCT_II(q_las)

        S_bot_raw = None
        if h_conv > 0:
            T_bottom = kernels.reconstruct_bottom_temperature(buffers.a_temp, SsState)
            q_conv = xp.float32(-h_conv) * (T_bottom - T0)
            S_bot_raw = grid.dct_scale * kernels.DCT_II(q_conv)

        # ================================================================
        # 5. Fixed-point (Picard) iteration
        # ================================================================
        for iter_k in range(self.max_picard_iter):
            xp.copyto(a_old, buffers.a_temp)

            T_surface = kernels.reconstruct_surface_temperature(a_old, SsState)
            kernels.compute_evaporation_flux(
                T_surface, buffers.q_evap_buffer,
                mat.Pa, mat.T_boil, mat.DeltaH_LV, mat.R_v, mat.T_liquidus,
            )
            S_evap = grid.dct_scale * kernels.DCT_II(buffers.q_evap_buffer)
            S_top_raw = S_las - S_evap

            Q_latent_raw = None
            if fm.T_prev is not None:
                xp.copyto(buffers.a_temp, a_old)
                buffers.Q_latent_buffer.fill(0.0)
                kernels.compute_latent_heat_source(
                    buffers.Q_latent_buffer, mat, num, SsState
                )
                Q_latent_raw = buffers.Q_latent_buffer

            kernels.update_modes_etd1(
                SsState.a, SsState.KK, grid.Cp32_broadcast,
                S_top_raw, buffers.a_temp,
            )
            if Q_latent_raw is not None:
                kernels.add_source_term_modes(
                    buffers.a_temp, SsState.KK,
                    kernels.project_box_to_modes(Q_latent_raw, SsState),
                )
            if S_bot_raw is not None:
                kernels.add_bottom_surface_source(
                    buffers.a_temp, SsState.KK,
                    grid.Cp32_broadcast_bottom, S_bot_raw,
                )

            xp.copyto(a_raw, buffers.a_temp)
            xp.subtract(a_raw, a_old, out=residual_curr)

            omega = self.mixing_omega
            xp.multiply(a_raw, omega, out=buffers.a_temp)
            buffers.a_temp += (xp.float32(1.0) - omega) * a_old

            rms_diff = xp.sqrt(xp.vdot(residual_curr, residual_curr) / n_elements)
            rms_old = xp.sqrt(xp.vdot(a_old, a_old) / n_elements)
            true_rel_err = float(rms_diff) / max(float(rms_old), 1e-9)

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
                break

        # Restore variables needed below
        T_temp = kernels.reconstruct_surface_temperature(buffers.a_temp, SsState)
        Q_latent = Q_latent_raw

        # ================================================================
        # 6. Commit converged state
        # ================================================================
        buffers.q_evap_old[:] = buffers.q_evap_buffer
        SsState.a = buffers.a_temp.copy()

        # ================================================================
        # 7. Update latent-heat history with converged temperature
        # ================================================================
        kernels.update_latent_heat_history(SsState)
        if fm.Q_prev is None:
            fm.Q_prev = xp.zeros_like(buffers.Q_latent_buffer)
        if Q_latent is not None:
            fm.Q_prev[:] = Q_latent[:]

        metrics = {
            'T_surface_max': xp.max(T_temp),
            'P_laser': P_laser,
            'n_evap_iter': iter_k + 1,
        }
        return SsState, metrics

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
        T = xp.asarray(temperature_field, dtype=xp.float32)
        # Forward DCT-II (ortho) converts the spatial field to ortho-normalised
        # coefficients. The solver's internal modes use a scaling of
        # sqrt(dx*dy*dz) relative to the standard ortho DCT coefficients.
        scale = xp.float32(math.sqrt(float(geom.d.x * geom.d.y * geom.d.z)))
        self.state.a = kernels.DCT_II(T) * scale

    def finalize(self) -> None:
        """
        Clean up resources (GPU memory, thread pools) if necessary.
        """
        pass
