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
from utils.visualisation import generate_plots

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
    
    # 1. Numerical Parameters
    sim_cfg = cfg.get('simulation', {})
    domain_cfg = cfg.get('domain', {})
    
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
        output_interval=float(sim_cfg.get('output_interval', 1e-3))
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
        L_f=float(mat_cfg.get('L_f', 0.0)),
        T_solidus=float(mat_cfg.get('T_solidus', 0.0)),
        T_liquidus=float(mat_cfg.get('T_liquidus', 0.0))
        # Note: Additional params like DeltaH_LV could be added to MaterialParams or passed via extra dict if needed
    )

    # 3. Laser Parameters
    laser_cfg = cfg.get('laser', {})
    laser_params = LaserParams(
        radius=float(laser_cfg['radius']),
        absorptivity=float(laser_cfg['absorptivity']),
        power=float(laser_cfg.get('power_nominal', 0.0))
    )

    # 4. Laser Path Strategy (Placeholder for now)
    # In a real implementation, this would instantiate GCodeLaserPath or LinearLaserPath
    path_cfg = laser_cfg.get('path', {})
    laser_path = None # To be implemented: basic path object or G-Code loader
    
    
    ctx = SimulationContext(
        num=num_params,
        mat=mat_params,
        geom=geom_params,
        laser=laser_params,
        laser_path=laser_path
    )
    
    return ctx

def get_factory(backend: str, context: SimulationContext):
    """Factory Selector."""
    if backend.lower() == 'gpu':
        try:
            from implementations.factories.gpu_factory import GPUSimulationFactory
            return GPUSimulationFactory(context)
        except ImportError as e:
            logger.error(f"Failed to import GPU Factory. Ensure 'implementations/factories/gpu_factory.py' exists and dependencies (cupy) are installed.")
            raise e
    elif backend.lower() == 'cpu':
        try:
            from implementations.factories.cpu_factory import CPUSimulationFactory
            return CPUSimulationFactory(context)
        except ImportError as e:
            logger.error(f"Failed to import CPU Factory. Ensure 'implementations/factories/cpu_factory.py' exists.")
            raise e
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: 'cpu', 'gpu'")

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

    # 3. Instantiate Factory & Workflow
    try:
        factory = get_factory(backend, context)
        workflow = SimulationWorkflow(context, factory)
        
        # 4. Run Simulation
        workflow.run()
        
    except Exception as e:
        logger.exception("Simulation Failed")
        sys.exit(1)

    # 5. Post-Processing / Visualization
    viz_cfg = config.get('post_processing', {})
    
    # Determine if we should visualize
    should_visualize = viz_cfg.get('auto_visualize', False)
    if args.viz: should_visualize = True
    if args.no_viz: should_visualize = False

    if should_visualize:
        logger.info("Starting Visualization...")
        
        # Construct output filename based on config
        # Assuming workflow uses these conventions. 
        # Ideally, workflow.run() returns the path to the result file.
        output_dir = config.get('output', {}).get('directory', 'out')
        # This is a bit brittle, workflow needs to expose where it saved data
        # For now, we assume a standard name or derived from config
        
        # Heuristic: find the latest xdmf in output dir?
        # Or require config to specify filename
        # Let's check config used in build_context, but I didn't store it in context completely...
        # For prototype, let's assume 'out/simulation_result.xdmf' or scan folder.
        
        # Better approach: 
        # Assume workflow saves to {output_dir}/validation.xdmf or similar. 
        # Let's try to locate the most recently modified .xdmf file in output dir
        
        try:
             files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.xdmf')]
             if not files:
                 logger.warning(f"No XDMF files found in {output_dir} for visualization.")
             else:
                 latest_file = max(files, key=os.path.getmtime)
                 logger.info(f"Visualizing result: {latest_file}")
                 
                 generate_plots(
                     xdmf_path=latest_file,
                     show_ui=viz_cfg.get('show_gui', True),
                     save_images=viz_cfg.get('save_images', True),
                     output_dir=output_dir
                 )
        except Exception as e:
             logger.error(f"Visualization failed: {str(e)}")

if __name__ == "__main__":
    main()
