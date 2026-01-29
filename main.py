import argparse
import yaml
import logging
import sys
import os
import numpy as np
from typing import Dict, Any

# Ensure we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.workflow import SimulationWorkflow
from core.parameters import SimulationContext, NumParams, MaterialParams, GeomParams, LaserParams
from utils.cut_views import generate_plots

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_context(cfg: Dict[str, Any]) -> SimulationContext:
    """Construct the SimulationContext object from dictionary configuration."""
    
    # 1. Numerical & Simulation Meta-Parameters
    sim_cfg = cfg.get('simulation', {})
    domain_cfg = cfg.get('domain', {})
    
    # Capture Method and Backend from config
    sim_method = sim_cfg.get('method', 'spectral').lower()
    sim_backend = sim_cfg.get('backend', 'cpu').lower()
    
    # Calculate mesh geometry
    Lx, Ly, Lz = domain_cfg['size']
    nx, ny, nz = domain_cfg['mesh']
    
    # Calculate Time Steps
    t_end = sim_cfg.get('duration', 0.01) # Default 10ms
    dt = float(sim_cfg['dt'])
    
    num_params = NumParams(
        dt=dt,
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        t_end=float(t_end),
        update_interval=float(sim_cfg.get('update_interval'))
    )

    geom_params = GeomParams(
        Lx=float(Lx), Ly=float(Ly), Lz=float(Lz),
        nx=int(nx), ny=int(ny), nz=int(nz)
    )

    # 2. Material Parameters
    mat_cfg = cfg.get('material', {})
    mat_params = MaterialParams(
        name=mat_cfg.get('name', 'Material'),
        rho=float(mat_cfg['rho']),
        k=float(mat_cfg['k']),
        Cp=float(mat_cfg['Cp']),
        L_f=float(mat_cfg.get('L_f')),
        T_solidus=float(mat_cfg.get('T_solidus')),
        T_liquidus=float(mat_cfg.get('T_liquidus')),
        Pa=float(mat_cfg.get('Pa')),
        R_v=float(mat_cfg.get('R_v')),
        T_boil=float(mat_cfg.get('T_boil')),
        DeltaH_LV=float(mat_cfg.get('DeltaH_LV')),
        T0=float(mat_cfg.get('T0')) # Reference temperature
    )

    # 3. Laser Parameters
    laser_cfg = cfg.get('laser', {})
    laser_params = LaserParams(
        radius=float(laser_cfg['radius']),
        absorptivity=float(laser_cfg['absorptivity']),
        power=float(laser_cfg.get('power_nominal'))
    )

    # 4. Laser Path Strategy
    path_cfg = laser_cfg.get('path', {})
    laser_path = None
    if path_cfg.get('type', '').lower() == 'gcode':
        from implementations.io.gcode_path import GCodeLaserPath
        gcode_file = path_cfg.get('file', None)
        initial_position = tuple(path_cfg.get('initial_position', [0.0, 0.0]))
        if gcode_file is not None:
            # Assume relative to data/paths if not absolute
            if not os.path.isabs(gcode_file):
                gcode_file = os.path.join(os.path.dirname(__file__), 'data', 'paths', gcode_file)
            laser_path = GCodeLaserPath(gcode_file, initial_position=initial_position)

    # 5. IO Parameters (flat dict: interval, outputs, at_end, etc.)
    io_cfg = cfg.get('io', {})
    # Optionally validate/normalize io_cfg here
    ctx = SimulationContext(
        num=num_params,
        mat=mat_params,
        geom=geom_params,
        laser=laser_params,
        laser_path=laser_path,
        io=io_cfg,  # Flat dict for new IO config
        method=sim_method,
        backend=sim_backend
    )
    return ctx

def get_factory(context: SimulationContext):
    """
    Select and instantiate the appropriate SimulationFactory based on context.
    Dispatches between Spectral (CPU/GPU) and FEM implementations.
    """
    method = context.method
    backend = context.backend
    
    logger.info(f"Factory Selector: Method='{method}', Backend='{backend}'")

    if method == "fem":
        # Import local to avoid loading dependencies if not used
        try:
            from implementations.factories.fem_factory import FEMSimulationFactory
            return FEMSimulationFactory(context)
        except ImportError as e:
            logger.error(f"Failed to import FEM factory: {e}")
            raise

    elif method == "spectral":
        if backend == "gpu":
            try:
                from implementations.factories.gpu_factory import GPUSimulationFactory
                return GPUSimulationFactory(context)
            except ImportError as e:
                logger.error(f"Failed to import GPU factory (check cupy installation): {e}")
                raise
        else:
            # Default to CPU Spectral
            from implementations.factories.cpu_factory import CPUSimulationFactory
            return CPUSimulationFactory(context)
            
    else:
        raise ValueError(f"Unknown simulation method: {method}")

def main():
    parser = argparse.ArgumentParser(description="FastHeatSolv: Spectral Heat Equation Solver")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--backend", default=None, choices=["cpu", "gpu"], help="Override backend (cpu/gpu)")
    parser.add_argument("--viz", action="store_true", help="Force visualization after simulation")
    parser.add_argument("--no-viz", action="store_true", help="Disable automatic visualization")
    args = parser.parse_args()

    # 1. Load Config & Context
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    context = build_context(config)

    # 2. Determine Computing Backend
    # Priority: CLI Argument > Config File > Default (CPU)
    backend = args.backend
    if not backend:
        backend = config.get('simulation', {}).get('backend', 'cpu')
    
    logger.info(f"Using Backend: {backend.upper()}")
    
    run_dir = None

    # 3. Instantiate Factory & Workflow

    try:
        # get_factory only needs context
        factory = get_factory(context) 
        workflow = SimulationWorkflow(context, factory)

        # 4. Run Simulation
        workflow.run()

    except Exception as e:
        logger.exception("Simulation Failed")
        sys.exit(1)

    # 5. Post-Processing / Visualization
    viz_cfg = config.get('post_processing', {})
    
    
    # Determine if we should visualize TO DO : The function save_step already exports profiles and cut view, this is redundant (in workflow)
    should_visualize = viz_cfg.get('auto_visualize', False)
    if args.viz: should_visualize = True
    if args.no_viz: should_visualize = False

    if should_visualize:
        logger.info("Starting Visualization...")
        
        if run_dir and os.path.exists(run_dir):
            logger.info(f"Visualizing results from: {run_dir}")
            
            try:
                 # Passing the specific run directory to generate_plots.
                 # The visualization tool should now load data using the loader utils
                 # or scan the profiles/ folder within run_dir.
                 generate_plots(
                     run_dir=run_dir,
                     show_ui=viz_cfg.get('show_gui', True),
                     save_images=viz_cfg.get('save_images', True),
                     output_dir=run_dir
                 )
            except Exception as e:
                 logger.error(f"Visualization failed: {str(e)}")
        else:
             logger.warning("Output directory not found or IOManager not active. Visualization skipped.")


if __name__ == "__main__":
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    # Try to write profiler output to diagnostics if possible
    try:
        # run_dir is set in main() as a global variable
        run_dir = None
        # Try to get run_dir from the main function's local scope
        import inspect
        frame = inspect.currentframe()
        while frame:
            if 'run_dir' in frame.f_locals:
                run_dir = frame.f_locals['run_dir']
                break
            frame = frame.f_back
        if run_dir and os.path.exists(run_dir):
            diag_dir = os.path.join(run_dir, "diagnostics")
            os.makedirs(diag_dir, exist_ok=True)
            prof_path = os.path.join(diag_dir, "profiler.txt")
            with open(prof_path, "w") as f:
                stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
                stats.print_stats(40)
            print(f"[cProfile] Top 40 functions written to {prof_path}")
        else:
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.print_stats(40)
    except Exception as e:
        print(f"[cProfile] Failed to write diagnostics: {e}")
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(40)
