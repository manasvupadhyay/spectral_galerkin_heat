"""
Spectral GPU Solver Implementation.
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

import numpy as np
import cupy as cp
import fast_heat_solv.physics.spectral_gpu_kernels as kernels
from fast_heat_solv.core.parameters import SimulationContext
from fast_heat_solv.solvers.base import HeatSolver
from fast_heat_solv.core.laser import LaserState, LaserPath
from typing import Optional, Any, Tuple, Dict

class SpectralSolverGPU(HeatSolver):
    """
    SpectralSolverGPU implements a spectral method for solving the heat equation on the GPU.
    
    It manages the solver state, initialization, and time-stepping logic, including laser source,
    latent heat, and evaporation effects. The solver is designed for modularity and performance.
    """
    def __init__(self, context: Optional[SimulationContext] = None):
        """
        Initializes the SpectralSolverGPU.

        Optionally attach a SimulationContext at construction time.
        The context can also be provided (or overridden) later via
        ``initialize(context)``.

        Parameters
        ----------
        context : SimulationContext, optional
            A dataclass containing complete simulation parameters, by default None.
        """
        self.context: Optional[SimulationContext] = context
        self.state: Optional[kernels.SpectralSolverState] = None
        self.mixing_omega: float = 0.1
        self.convergence_tol: float = 1e-4
        self.max_picard_iter: int = 30

    def initialize(self, context: SimulationContext) -> Any:
        """
        Set up the spectral solver state, allocate buffers, and set the initial condition.

        Parameters
        ----------
        context : SimulationContext
            Replaces the stored SimulationContext.

        Returns
        -------
        fast_heat_solv.physics.spectral_gpu_kernels.SpectralSolverState
            The initialized GPU solver state object.
        """
        self.context = context

        geom = self.context.geom
        num = self.context.num
        mat = self.context.mat

        # Initialize spectral solver state
        self.state = kernels.SpectralSolverState(mat, geom, num)

        # Initial condition: mean T in mode (0,0,0)
        self.state.a = cp.zeros((num.nz, num.ny, num.nx), dtype=cp.float32)
        # Use T0 if present, else default to 293.0
        T0 = getattr(mat, 'T0', 293.0)
        self.state.a[0, 0, 0] = cp.float32(T0 * np.sqrt(geom.Lx * geom.Ly * geom.Lz))

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
            SsState : fast_heat_solv.physics.spectral_gpu_kernels.SpectralSolverState
                 The updated GPU solver state.
            metrics : dict
                 Metrics from the iteration, such as max temperature and number of evaporation steps.
        """
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

        laser_state: LaserState = laser_path.get_state(t, dt)
        power = laser_state.power
        is_on = laser_state.is_on
        v_x, v_y = laser_state.v
        absorptivity = laser_params.absorptivity
        laser_coef = absorptivity * 2.0 * power / (np.pi * laser_params.radius ** 2) if is_on else 0.0

        # ================================================================
        # 1. Compute laser flux (constant – does not depend on T)
        # ================================================================
        q_las = kernels.compute_gaussian_laser_flux(
            grid.x, grid.y, laser_state.x, laser_state.y,
            laser_params.radius, laser_coef
        )
        P_laser = cp.sum(q_las) * geom.dx * geom.dy

        # ================================================================
        # 2. Apply exponential propagator:  θ̃ = E · θ
        # ================================================================
        cp.multiply(SsState.a, SsState.K, out=SsState.a)

        # ================================================================
        # 3. Prepare latent-heat history (once per step)
        # ================================================================
        if fm:
            fm.update(laser_state)
            buffers.a_temp[:] = SsState.a
            kernels.initialize_latent_heat_if_needed(SsState)
            kernels.shift_latent_heat_history(fm, laser_state, num)

        # ================================================================
        # 4. Initial guesses for forcing components
        # ================================================================
        q_evap_shifted = kernels.shift_flux(buffers.q_evap_old, (v_x * dt, v_y * dt), geom)
        S_top = grid.dct_scale * kernels.DCT_II(q_las - q_evap_shifted)

        Q_latent = None
        if fm:
            Q_latent = cp.zeros_like(buffers.Q_latent_buffer)
            if fm.Q_prev is not None:
                Q_latent[:] = fm.Q_prev

        h_conv = getattr(mat, 'h_conv', 0.0)
        T0 = cp.float32(getattr(mat, 'T0', 293.0))

        kernels.update_modes_etd1(SsState.a, SsState.KK, grid.Cp32_broadcast, S_top, buffers.a_temp)
        if fm and fm.T_prev is not None and Q_latent is not None:
            kernels.add_source_term_modes(
                buffers.a_temp, SsState.KK,
                kernels.project_box_to_modes(Q_latent, SsState))
        if h_conv > 0:
            T_bottom = kernels.reconstruct_bottom_temperature(buffers.a_temp, SsState)
            q_conv = cp.float32(-h_conv) * (T_bottom - T0)
            S_bot_init = grid.dct_scale * kernels.DCT_II(q_conv)
            kernels.add_bottom_surface_source(
                buffers.a_temp, SsState.KK,
                grid.Cp32_broadcast_bottom, S_bot_init)

        # ================================================================
        # Pre-allocate arrays to avoid per-iteration allocations
        # ================================================================
        a_old = cp.empty_like(buffers.a_temp)
        a_raw = cp.empty_like(buffers.a_temp)
        residual_curr = cp.empty_like(buffers.a_temp)
        n_elements = cp.float32(buffers.a_temp.size)

        # ================================================================
        # Hoist linear/constant forcing terms
        # ================================================================
        S_las = grid.dct_scale * kernels.DCT_II(q_las)

        S_bot_raw = None
        if h_conv > 0:
            T_bottom = kernels.reconstruct_bottom_temperature(buffers.a_temp, SsState)
            q_conv = cp.float32(-h_conv) * (T_bottom - T0)
            S_bot_raw = grid.dct_scale * kernels.DCT_II(q_conv)

        # ================================================================
        # 5. Fixed-point iteration
        # ================================================================
        for iter_k in range(self.max_picard_iter):
            cp.copyto(a_old, buffers.a_temp)

            T_surface = kernels.reconstruct_surface_temperature(a_old, SsState)
            kernels.compute_evaporation_flux(
                T_surface, buffers.q_evap_buffer,
                mat.Pa, mat.T_boil, mat.DeltaH_LV, mat.R_v, mat.T_liquidus)
            S_evap = grid.dct_scale * kernels.DCT_II(buffers.q_evap_buffer)
            S_top_raw = S_las - S_evap

            Q_latent_raw = None
            if fm and fm.T_prev is not None:
                cp.copyto(buffers.a_temp, a_old)
                buffers.Q_latent_buffer.fill(0.0)
                kernels.compute_latent_heat_source(buffers.Q_latent_buffer, mat, num, SsState)
                Q_latent_raw = buffers.Q_latent_buffer

            kernels.update_modes_etd1(SsState.a, SsState.KK, grid.Cp32_broadcast, S_top_raw, buffers.a_temp)
            if Q_latent_raw is not None:
                kernels.add_source_term_modes(
                    buffers.a_temp, SsState.KK,
                    kernels.project_box_to_modes(Q_latent_raw, SsState))
            if S_bot_raw is not None:
                kernels.add_bottom_surface_source(
                    buffers.a_temp, SsState.KK,
                    grid.Cp32_broadcast_bottom, S_bot_raw)

            cp.copyto(a_raw, buffers.a_temp)
            cp.subtract(a_raw, a_old, out=residual_curr)

            omega = cp.float32(self.mixing_omega)
            # if iter_k == 0:
            #     cp.copyto(buffers.a_temp, a_raw)
            # else:
            #     cp.multiply(a_raw, omega, out=buffers.a_temp)
            #     buffers.a_temp += (cp.float32(1.0) - omega) * a_old
            np.multiply(a_raw, omega, out=buffers.a_temp)
            buffers.a_temp += (np.float32(1.0) - omega) * a_old

            rms_diff = float(cp.sqrt(cp.vdot(residual_curr, residual_curr) / n_elements))
            rms_old = float(cp.sqrt(cp.vdot(a_old, a_old) / n_elements))
            true_rel_err = rms_diff / max(rms_old, 1e-9)

            if true_rel_err < float(self.convergence_tol):
                break

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
        if fm:
            kernels.update_latent_heat_history(SsState)
            if fm.Q_prev is None:
                fm.Q_prev = cp.zeros_like(buffers.Q_latent_buffer)
            if Q_latent is not None:
                fm.Q_prev[:] = Q_latent[:]

        metrics = {
            'T_surface_max': cp.max(T_temp),
            'P_laser': P_laser,
            'n_evap_iter': iter_k + 1
        }
        return SsState, metrics

    def set_state(self, temperature_field: np.ndarray) -> None:
        """
        Overwrite the internal spectral coefficients from a spatial temperature field.

        Converts the cell-centred temperature array into spectral (DCT) modes
        so that the solver can continue stepping from the injected state.

        Parameters
        ----------
        temperature_field : np.ndarray or cupy.ndarray
            A 3-D array (shape ``(N_z, N_y, N_x)``) containing temperatures.
        
        Raises
        ------
        RuntimeError
            If called before the solver is initialized.
        """
        if self.state is None or self.context is None:
            raise RuntimeError("Solver must be initialized before calling set_state().")
        geom = self.context.geom
        T = cp.asarray(temperature_field, dtype=cp.float32)
        scale = cp.float32(np.sqrt(np.float64(geom.dx * geom.dy * geom.dz)))
        self.state.a = kernels.DCT_II(T) * scale

    def finalize(self) -> None:
        """
        Clean up GPU resources if necessary.
        """
        pass
