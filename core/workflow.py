import time
import logging
from typing import Optional, Dict, Any
from ..interfaces.factory import SimulationFactory
from ..interfaces.solver import HeatSolver
from ..interfaces.io import IOManager
from .parameters import SimulationContext

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
        # This creates the run directory and sets up file handlers
        self.io_manager.initialize(self.context)
        
        # 2. Initialize Solver State
        # Assuming solver.initialize() returns the initial field (e.g. spectral coefficients or temp array)
        state = self.heat_solver.initialize()
        
        # 3. Time Loop
        t = 0.0
        step = 0
        dt = self.context.num.dt
        t_end = self.context.num.t_end
        
        # IO Timers
        next_output_time = 0.0
        
        logger.info(f"Starting time loop: 0 -> {t_end:.4e} s (dt={dt:.2e})")

        while t < t_end:
            # A. Output Check
            # We save at the *start* of the step for t=0, or when threshold passed
            if t >= next_output_time:
                # Pass data to IO manager
                # 'metrics' can be a dict returned by solver containing scalar diagnostics (Power, MaxT, etc)
                # For now we pass the raw state. The IO manager decides what to write.
                self.io_manager.save_step(t, step, state)
                
                logger.info(f"Step {step} | t={t:.6e}s | Output saved")
                next_output_time += self.context.io.output_interval

            # B. Evolve State
            # Solver returns new state and potentially a dictionary of metrics (e.g., {'P_laser': ..., 'max_T': ...})
            # Adjust signature based on your specific HeatSolver interface definition
            state, metrics = self.heat_solver.step(t, dt, state)
            
            # C. Advance Time
            t += dt
            step += 1
            
            # Optional: Log progress periodically
            if step % 100 == 0:
                logger.debug(f"Step {step}/{int(t_end/dt)}")

        # 4. Finalize
        self.io_manager.finalize()
        logger.info("Simulation completed successfully.")
