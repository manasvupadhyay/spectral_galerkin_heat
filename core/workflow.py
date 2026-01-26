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

        # 2. Initialize Solver State
        state = self.heat_solver.initialize()

        # 3. Time Loop
        t = 0.0
        step = 0
        dt = self.context.num.dt
        t_end = self.context.num.t_end

        # IO Timers
        next_output_time = 0.0

        logger.info(f"Starting time loop: 0 -> {t_end:.4e} s (dt={dt:.2e})")

        # Check for dynamic laser_path in context
        laser_path = getattr(self.context, 'laser_path', None)

        # ETA logging setup
        start_wall_time = time.time()
        last_eta_log_time = start_wall_time
        eta_log_interval = 10.0  # seconds

        while t < t_end:
            # A. Output Check
            if t >= next_output_time:
                self.io_manager.save_step(t, step, state)
                # fs_io save not implemented yet 
                logger.info(f"Step {step} | t={t:.6e}s | Output saved")
                next_output_time += self.context.io.output_interval

            # B. Evolve State
            state, metrics = self.heat_solver.step(t, dt, state)

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
                else:
                    logger.info(f"[ETA] Step {step} | t={t:.6e}s | Elapsed: {elapsed:.1f}s | Remaining: unknown")
                last_eta_log_time = now

            if step % 100 == 0:
                logger.debug(f"Step {step}/{int(t_end/dt)}")

        # 4. Finalize
        self.io_manager.finalize()
        logger.info("Simulation completed successfully.")
