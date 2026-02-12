import time
import logging
import sys
import platform
from typing import Optional, Dict, Any
from interfaces.factory import SimulationFactory
from interfaces.solver import HeatSolver
from interfaces.io import IOManager
from core.parameters import SimulationContext

logger = logging.getLogger(__name__)

class SimulationWorkflow:
    """
    Orchestrates the entire simulation lifecycle.

    This class implements the Strategy pattern by delegating specific tasks
    (solving, I/O) to components created by the SimulationFactory. It manages
    the time-stepping loop, logging, and coordinating outputs.

    Attributes:
        context (SimulationContext): Global configuration and state parameters.
        factory (SimulationFactory): Factory for creating backend-specific components.
        heat_solver (HeatSolver): The numerical solver instance.
        io_manager (IOManager): The input/output manager instance.
    """
    def __init__(self, context: SimulationContext, factory: SimulationFactory):
        self.context = context
        self.factory = factory
        
        # Instantiate core components via Factory
        # The factory decides WHICH implementation (CPU vs GPU, Spectral vs FEM) is used.
        self.heat_solver: HeatSolver = self.factory.create_heat_solver()
        
        # Create IO Manager
        self.io_manager: IOManager = self.factory.create_io_manager()



    def run(self):
        """
        Execute the main simulation loop.
        """
        logger.info("Initializing simulation workflow...")

        # 1. Initialize IO System
        self.io_manager.initialize(self.context)

        # --- LOG FILE SETUP ---
        run_dir = self.io_manager.base_dir
        if run_dir is not None:
            import os
            log_file_path = os.path.join(run_dir, "logs", "simulation.log")
            # Remove previous file handlers if any (avoid duplicate logs)
            for h in logger.handlers[:]:
                if isinstance(h, logging.FileHandler):
                    logger.removeHandler(h)
            file_handler = logging.FileHandler(log_file_path, mode="a")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
            logger.info(f"File logging enabled: {log_file_path}")

        # 2. Initialize Solver State
        state = self.heat_solver.initialize()

        # --- Startup context & environment summary (useful for reproducibility) ---
        try:
            import numpy as _np
            try:
                import cupy as _cp
                cupy_ver = getattr(_cp, '__version__', 'unknown')
            except Exception:
                _cp = None
                cupy_ver = None
        except Exception:
            _np = None
            cupy_ver = None

        geom = getattr(self.context, 'geom', None)
        mat = getattr(self.context, 'mat', None)
        num = getattr(self.context, 'num', None)

        logger.info(f"Config Summary: method={self.context.method}, backend={self.context.backend}, "
                    f"mesh={(getattr(geom,'nx',None))}x{getattr(geom,'ny',None)}x{getattr(geom,'nz',None)}, "
                    f"dt={getattr(num,'dt',None):.3e}, t_end={getattr(num,'t_end',None):.3e}, "
                    f"material={getattr(mat,'name', None)}")
        logger.info(f"Environment: python={platform.python_version()}, numpy={getattr(_np,'__version__',None)}, cupy={cupy_ver}")

        # 3. Time Loop
        t = 0.0
        step = 0
        dt = self.context.num.dt
        t_end = self.context.num.t_end

        # IO Timers
        io_cfg = self.context.io if hasattr(self.context, 'io') else {}
        interval = io_cfg.get('interval')
        outputs = io_cfg.get('outputs', [])
        at_end = io_cfg.get('at_end', [])
        profiles_locations = io_cfg.get('profiles_locations', [])
        cut_views_planes = io_cfg.get('cut_views_planes', [])

        # Defensive handling: if interval is None (YAML null) we disable periodic outputs.
        if interval is None:
            logger.info("io.interval is None: periodic outputs disabled; only 'at_end' outputs will be saved.")
            next_output_step = float('inf')
        else:
            # Ensure integral number of steps
            try:
                next_output_step = int(interval)
            except Exception:
                logger.warning(f"Invalid io.interval '{interval}' - disabling periodic outputs.")
                next_output_step = float('inf')
        logger.info(f"Starting time loop: 0 -> {t_end:.4e} s (dt={dt:.2e})")

        # Check for dynamic laser_path in context
        laser_path = getattr(self.context, 'laser_path')
        
        # ETA logging & telemetry setup
        start_wall_time = time.time()
        last_eta_log_time = start_wall_time
        eta_log_interval = self.context.num.update_interval
        # telemetry counters
        total_step_time = 0.0
        n_steps_timed = 0

        # optional memory measurement
        try:
            import psutil
            _psutil_proc = psutil.Process()
        except Exception:
            _psutil_proc = None

        while t < t_end:
            # A. Output Check (step-based)
            if step >= next_output_step:
                for output_type in outputs:
                    self.io_manager.save_step(
                        t, step, state, laser_path,
                        output_type=output_type,
                        profiles_locations=profiles_locations,
                        cut_views_planes=cut_views_planes
                    )
                logger.info(f"Step {step} | t={t:.6e}s | Output(s) saved: {outputs}")
                if interval is not None:
                    try:
                        next_output_step += int(interval)
                    except Exception:
                        next_output_step = float('inf')
                else:
                    next_output_step = float('inf')
            # B. Evolve State (timed)
            step_start = time.time()
            state, metrics = self.heat_solver.step(t, dt)
            step_elapsed = time.time() - step_start
            total_step_time += step_elapsed
            n_steps_timed += 1

            # C. Advance Time
            t += dt
            step += 1

            # ETA logging (every eta_log_interval seconds or at end)
            now = time.time()
            if (now - last_eta_log_time >= eta_log_interval) or (t >= t_end):
                elapsed = now - start_wall_time
                frac_done = min(t / t_end, 1.0) if t_end > 0 else 0.0
                if frac_done > 0:
                    est_total = elapsed / frac_done
                    est_remaining = est_total - elapsed
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(est_remaining))
                    # Memory usage (RSS in MB) if available
                    if _psutil_proc is not None:
                        try:
                            mem_mb = _psutil_proc.memory_info().rss / (1024.0 ** 2)
                        except Exception:
                            mem_mb = None
                    else:
                        mem_mb = None

                    logger.info(f"[ETA] Step {step} | t={t:.6e}s | Elapsed: {elapsed:.1f}s | Remaining: {eta_str} | "
                                f"step_time={step_elapsed:.3f}s | avg_step={ (total_step_time / n_steps_timed):.3f}s | mem_mb={mem_mb if mem_mb is not None else 'NA'}")

                    # Format metrics into plain Python scalars and controlled precision
                    metrics_items = []
                    for k, v in (metrics or {}).items():
                        try:
                            if isinstance(v, (int,)):
                                metrics_items.append(f"{k}={int(v)}")
                            else:
                                metrics_items.append(f"{k}={float(v):.3f}")
                        except Exception:
                            metrics_items.append(f"{k}={v}")
                    metrics_str = ", ".join(metrics_items)
                    logger.info(f"[METRICS] {metrics_str}")
                else:
                    logger.info(f"[ETA] Step {step} | t={t:.6e}s | Elapsed: {elapsed:.1f}s | Remaining: unknown")
                    metrics_items = []
                    for k, v in (metrics or {}).items():
                        try:
                            if isinstance(v, (int,)):
                                metrics_items.append(f"{k}={int(v)}")
                            else:
                                metrics_items.append(f"{k}={float(v):.3f}")
                        except Exception:
                            metrics_items.append(f"{k}={v}")
                    metrics_str = ", ".join(metrics_items)
                    logger.info(f"[METRICS] {metrics_str}")
                last_eta_log_time = now

        # At end: save all requested outputs
        for output_type in at_end:
            self.io_manager.save_step(
                t, step, state, laser_path,
                output_type=output_type,
                profiles_locations=profiles_locations,
                cut_views_planes=cut_views_planes
            )
        logger.info(f"Final output(s) saved at end: {at_end}")


        # 4. Finalize
        self.io_manager.finalize()
        # Optionally log profiler diagnostics path if present
        try:
            run_dir = getattr(self.io_manager, 'base_dir', None)
            if run_dir:
                prof_path = os.path.join(run_dir, 'diagnostics', 'profiler.txt')
                import os
                if os.path.exists(prof_path):
                    logger.info(f"Profiler output written to: {prof_path}")
        except Exception:
            pass

        logger.info("Simulation completed successfully.")