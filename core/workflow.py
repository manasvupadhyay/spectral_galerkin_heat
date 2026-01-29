import time
import logging
from typing import Optional, Dict, Any
from interfaces.factory import SimulationFactory
from interfaces.solver import HeatSolver
from interfaces.io import IOManager
from core.parameters import SimulationContext

logger = logging.getLogger(__name__)

class SimulationWorkflow:
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

        # Defensive handling: if interval is None (YAML null) we disable periodic outputs
        if interval is None:
            logger.info("io.interval is None: periodic outputs disabled; only 'at_end' outputs will be saved.")
            next_output_time = float('inf')
        else:
            # Ensure numeric
            try:
                next_output_time = float(interval)
            except Exception:
                logger.warning(f"Invalid io.interval '{interval}' - disabling periodic outputs.")
                next_output_time = float('inf')
        logger.info(f"Starting time loop: 0 -> {t_end:.4e} s (dt={dt:.2e})")

        # Check for dynamic laser_path in context
        laser_path = getattr(self.context, 'laser_path')
        
        # ETA logging setup
        start_wall_time = time.time()
        last_eta_log_time = start_wall_time
        eta_log_interval = self.context.num.update_interval

        while t < t_end:
            # A. Output Check
            if t >= next_output_time:
                for output_type in outputs:
                    self.io_manager.save_step(
                        t, step, state, laser_path,
                        output_type=output_type,
                        profiles_locations=profiles_locations,
                        cut_views_planes=cut_views_planes
                    )
                logger.info(f"Step {step} | t={t:.6e}s | Output(s) saved: {outputs}")
                next_output_time += interval
            # B. Evolve State
            state, metrics = self.heat_solver.step(t, dt)
            
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
                    logger.info(f"[ETA] Step {step} | t={t:.6e}s | Elapsed: {elapsed:.1f}s | Remaining: {eta_str}")
                    logger.info(f"[METRICS] {metrics}")
                else:
                    logger.info(f"[ETA] Step {step} | t={t:.6e}s | Elapsed: {elapsed:.1f}s | Remaining: unknown")
                    logger.info(f"[METRICS] {metrics}")
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
        logger.info("Simulation completed successfully.")   