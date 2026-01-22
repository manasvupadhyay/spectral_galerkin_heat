import time
import logging
from typing import Optional
from ..interfaces.factory import SimulationFactory
from ..interfaces.solver import HeatSolver
from ..interfaces.io import IOManager
from ..interfaces.laser import LaserPath
from .parameters import SimulationContext

logger = logging.getLogger(__name__)

class SimulationWorkflow:
    def __init__(self, context: SimulationContext, factory: SimulationFactory):
        self.context = context
        self.factory = factory
        
        # Instantiate core components via Factory
        # The factory decides WHICH implementation (CPU vs GPU, Spectral vs FEM) is used.
        self.heat_solver: HeatSolver = self.factory.create_heat_solver()
        self.io: IOManager = self.factory.create_io_manager()
        
        # --- EXTENSION POINT: Future Solvers ---
        # If another solver is added later, it will be instantiated here:
        # self.another_solver = self.factory.create_another_solver()
        # --------------------------------------
        
        self.current_time = 0.0
        self.step_index = 0

    def run(self):
        """
        Main execution method. Orchestrates the initialization, 
        time-stepping loop, and finalization.
        """
        logger.info(f"Initializing simulation: {self.context.mat.name}")
        
        # 1. Initialization phase
        self.heat_solver.initialize()
        self.io.initialize(self.context)
        
        # Setup time loop variables
        t_end = self.context.num.t_end
        dt = self.context.num.dt
        next_output_time = 0.0
        
        # Access laser path strategy
        # Note: In the future if 'laser_path' is not in context, 
        # it might need to be passed or created via factory too.
        laser_path = self.context.laser_path
        
        start_wall_time = time.time()
        
        try:
            while self.current_time < t_end:
                # A. Get Boundary Conditions / Source Terms
                # The laser state depends on time (handled by the Strategy)
                # This decouples the movement logic (G-code vs Linear) from the solver.
                laser_state = laser_path.get_state(self.current_time)
                
                # B. Physics Solve Step
                # Advance heat equation by one step 'dt'
                # The solver implementation handles strictly calculation (FFT, matmul, etc.)
                temp_field = self.heat_solver.solve_step(self.current_time, dt, laser_state)
                
                # --- EXTENSION POINT: Coupled Physics ---
                # This is where you would plug in other physics modules that depend on T.
                # Example:
                # self.grain_solver.update(temp_field, dt)
                # self.fluid_solver.update(...)
                # --------------------------------------
                
                # C. IO / Analysis
                # Decouples saving frequency from calculation frequency
                if self.current_time >= next_output_time or self.context.num.save_all:
                    self.io.save_step(self.current_time, self.step_index, temp_field)
                    
                    # Advance output timer
                    next_output_time += self.context.num.output_interval
                    
                    # Simple progress log to console
                    elapsed = time.time() - start_wall_time
                    percent = (self.current_time / t_end) * 100 if t_end > 0 else 0
                    print(f"Time: {self.current_time:.4e}s ({percent:.1f}%) - Wall: {elapsed:.2f}s", end='\r')

                # D. Advance Time
                self.current_time += dt
                self.step_index += 1
                
        except KeyboardInterrupt:
            logger.warning("\nSimulation interrupted by user.")
            
        except Exception as e:
            logger.error(f"\nSimulation failed with error: {e}", exc_info=True)
            raise e
            
        finally:
            # 3. Finalization & Cleanup
            logger.info("Finalizing simulation...")
            self.io.finalize()
            total_time = time.time() - start_wall_time
            logger.info(f"Done. Total wall time: {total_time:.2f}s")
