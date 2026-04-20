"""
StandaloneHeatRunner – self-contained simulation runner.

This class extracts the time-stepping loop, I/O management, and telemetry
from ``SimulationWorkflow``. It is the recommended entry-point for running
fastHeatSolv in **standalone** mode (i.e. driven by a YAML config file).
"""

import time
import logging
import os
import platform
from typing import Optional, Dict, Any

from fast_heat_solv.factories.base import SimulationFactory
from fast_heat_solv.solvers.base import HeatSolver
from fast_heat_solv.io_utils.io_base import IOManager
from fast_heat_solv.core.parameters import SimulationContext

logger = logging.getLogger(__name__)


class StandaloneHeatRunner:
    """
    Orchestrates the full simulation lifecycle in standalone mode.

    Responsibilities
    ----------------
    * Creates solver and I/O manager via the supplied factory.
    * Runs the time-stepping loop (``while t < t_end``).
    * Delegates periodic / end-of-run output to the ``IOManager``.
    * Prints ETA / telemetry to the logger.

    Parameters
    ----------
    context : SimulationContext
        Fully populated simulation parameters.
    factory : SimulationFactory
        Abstract factory that produces backend-specific components.
    """

    def __init__(self, context: SimulationContext, factory: SimulationFactory):
        self.context = context
        self.factory = factory

        # Create components via factory
        self.heat_solver: HeatSolver = self.factory.create_heat_solver()
        self.io_manager: IOManager = self.factory.create_io_manager()

    # ------------------------------------------------------------------
    #  Telemetry helpers
    # ------------------------------------------------------------------
    def _init_telemetry(self) -> None:
        """
        Initialises wall-clock tracking and optional memory probe.

        Sets up the internal clock mechanisms required to track the processing 
        time per step and optionally hooks into the `psutil` package to track
        RAM consumption during the loop's execution.
        """
        self._start_wall_time = time.time()
        self._last_eta_log_time = self._start_wall_time
        self._eta_log_interval = self.context.num.update_interval
        self._total_step_time = 0.0
        self._n_steps_timed = 0
        try:
            import psutil
            self._psutil_proc = psutil.Process()
        except Exception:
            self._psutil_proc = None

    def _log_progress(
        self,
        t: float,
        step: int,
        step_elapsed: float,
        metrics: Optional[Dict[str, Any]],
    ) -> None:
        """
        Logs ETA, memory usage, and solver metrics at the configured interval.

        Parameters
        ----------
        t : float
            Current simulation physical time in seconds.
        step : int
            Current time step iteration index.
        step_elapsed : float
            Wall-time duration of the previous simulation step in seconds.
        metrics : dict of str to any, optional
            A dictionary containing any internal solving metrics mapped by string ID.
        """
        self._total_step_time += step_elapsed
        self._n_steps_timed += 1

        t_end = self.context.num.t_end
        now = time.time()
        if (now - self._last_eta_log_time < self._eta_log_interval) and (t < t_end):
            return  # not yet time to log

        elapsed = now - self._start_wall_time
        frac_done = min(t / t_end, 1.0) if t_end > 0 else 0.0

        if frac_done > 0:
            est_remaining = (elapsed / frac_done) - elapsed
            eta_str = time.strftime("%H:%M:%S", time.gmtime(est_remaining))
            mem_mb = None
            if self._psutil_proc is not None:
                try:
                    mem_mb = self._psutil_proc.memory_info().rss / (1024.0 ** 2)
                except Exception:
                    pass
            avg_step = self._total_step_time / self._n_steps_timed
            logger.info(
                f"[ETA] Step {step} | t={t:.6e}s | Elapsed: {elapsed:.1f}s | "
                f"Remaining: {eta_str} | step_time={step_elapsed:.3f}s | "
                f"avg_step={avg_step:.3f}s | mem_mb={mem_mb if mem_mb is not None else 'NA'}"
            )
        else:
            logger.info(
                f"[ETA] Step {step} | t={t:.6e}s | Elapsed: {elapsed:.1f}s | Remaining: unknown"
            )

        # Format metrics
        parts = []
        for k, v in (metrics or {}).items():
            try:
                if isinstance(v, int):
                    parts.append(f"{k}={v}")
                else:
                    parts.append(f"{k}={float(v):.3f}")
            except Exception:
                parts.append(f"{k}={v}")
        logger.info(f"[METRICS] {', '.join(parts)}")
        self._last_eta_log_time = now

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Executes the main simulation loop.

        It initializes the logging mechanism and file outputs, builds the initial 
        state via the solver interface, and proceeds to perform the chronological
        advancement loop until `t_end` is reached. Final states are serialized 
        before cleanup.
        """
        logger.info("Initializing standalone simulation runner...")

        # 1. Initialize IO System
        self.io_manager.initialize(self.context)

        # --- LOG FILE SETUP ---
        run_dir = self.io_manager.base_dir
        if run_dir is not None:
            log_file_path = os.path.join(run_dir, "logs", "simulation.log")
            # Remove previous file handlers to avoid duplicates
            for h in logger.handlers[:]:
                if isinstance(h, logging.FileHandler):
                    logger.removeHandler(h)
            file_handler = logging.FileHandler(log_file_path, mode="a")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(file_handler)
            logger.info(f"File logging enabled: {log_file_path}")

        # 2. Initialize Solver (pass context for dependency injection)
        state = self.heat_solver.initialize(self.context)

        # --- Startup context & environment summary ---
        try:
            import numpy as _np
            try:
                import cupy as _cp
                cupy_ver = getattr(_cp, "__version__", "unknown")
            except Exception:
                cupy_ver = None
        except Exception:
            _np = None
            cupy_ver = None

        geom = getattr(self.context, "geom", None)
        mat = getattr(self.context, "mat", None)
        num = getattr(self.context, "num", None)

        logger.info(
            f"Config Summary: method={self.context.method}, backend={self.context.backend}, "
            f"mesh={getattr(geom, 'nx', None)}x{getattr(geom, 'ny', None)}x{getattr(geom, 'nz', None)}, "
            f"dt={getattr(num, 'dt', None):.3e}, t_end={getattr(num, 't_end', None):.3e}, "
            f"material={getattr(mat, 'name', None)}"
        )
        logger.info(
            f"Environment: python={platform.python_version()}, "
            f"numpy={getattr(_np, '__version__', None)}, cupy={cupy_ver}"
        )

        # 3. Time Loop
        t = 0.0
        step = 0
        dt = self.context.num.dt
        t_end = self.context.num.t_end
        laser_path = getattr(self.context, "laser_path")

        logger.info(f"Starting time loop: 0 -> {t_end:.4e} s (dt={dt:.2e})")
        self._init_telemetry()

        while t < t_end:
            # A. Periodic output (IOManager decides internally)
            self.io_manager.process_step(t, step, state, laser_path)

            # B. Evolve state (pure physics – no I/O inside step)
            step_start = time.time()
            state, metrics = self.heat_solver.step(t, dt)
            step_elapsed = time.time() - step_start

            # C. Advance time
            t += dt
            step += 1

            # D. Telemetry (internally rate-limited)
            self._log_progress(t, step, step_elapsed, metrics)

        # End-of-simulation outputs
        self.io_manager.process_end(t, step, state, laser_path)

        # 4. Finalize
        self.heat_solver.finalize()
        self.io_manager.finalize()

        # Optionally log profiler diagnostics path if present
        try:
            run_dir = getattr(self.io_manager, "base_dir", None)
            if run_dir:
                prof_path = os.path.join(run_dir, "diagnostics", "profiler.txt")
                if os.path.exists(prof_path):
                    logger.info(f"Profiler output written to: {prof_path}")
        except Exception:
            pass

        logger.info("Simulation completed successfully.")
