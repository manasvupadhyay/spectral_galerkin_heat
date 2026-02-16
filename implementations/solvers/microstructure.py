import logging
import numpy as np
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from core.parameters import SimulationContext
from interfaces.microstructure import MicrostructureSolver

logger = logging.getLogger(__name__)

@dataclass
class Seed:
    """Represents a single grain seed with position and orientation."""
    position: np.ndarray  # [x, y] or [x, y, z]
    orientation: np.ndarray  # Euler angles [phi1, Phi, phi2]
    phase: int = 0

class TreeMicroSolver(MicrostructureSolver):
    """
    Concrete implementation of MicrostructureSolver using a Tree-based spatial index.
    Uses Neper (via subprocess) for microstructure generation.
    """

    def __init__(self, context: SimulationContext):
        self.context = context
        self.params = context.micro
        self.seeds: Optional[List[Any]] = None
        self.tree: Any = None
        self.active = self.params.enabled

    def initialize(self, output_dir: Optional[str] = None) -> Any:
        """
        Initialize the microstructure state.
        
        Args:
           output_dir (str): Directory to save intermediate files.
        """
        if not self.active:
            logger.info("Microstructure solver is disabled.")
            return None
            
        self.output_dir = output_dir

        logger.info(f"Initializing TreeMicroSolver (Type: {self.params.initial_type})")

        # 1. GENERATE or LOCATE seeds
        seeds_file = None
        
        if self.params.initial_type == 'from_file':
            # Use provided file as source
            seeds_file = self.params.input_file
            if not seeds_file or not os.path.exists(seeds_file):
                logger.error(f"Input seed file not found: {seeds_file}")
                return None
                
        elif self.params.initial_type == 'synthetic_voronoi':
            # Generate seeds using Neper and save to output_dir/seeds/seeds.txt
            seeds_dir = os.path.join(self.output_dir, "seeds")
            seeds_file = os.path.join(seeds_dir, "seeds.txt")
            self._generate_and_save_seeds(seeds_file, seeds_dir)
        
        else:
             logger.warning(f"Unknown type {self.params.initial_type}")
             return None

        # 2. LOAD seeds (Standardizes input format)
        if seeds_file and os.path.exists(seeds_file):
            self._load_seeds_from_file(seeds_file)
        else:
            logger.error("Failed to initialize seeds.")
            return None

        # 3. Create Spatial Tree
        if self.seeds:
            self.tree = self.create_tree(self.seeds)
            count = len(self.seeds)
            logger.info(f"Microstructure initialized with {count} seeds.")
        
        return self.seeds

    def _generate_and_save_seeds(self, filepath: str, work_dir: str):
        """
        Generate synthetic seeds using Neper and write to file.
        """
        gen_params = self.params.generation_params
        # Default n_grains if not provided
        n_grains = gen_params.get('n_grains', 100)
        
        # Use simulation bounds
        Lx = self.context.geom.Lx
        Ly = self.context.geom.Ly
        domain_size = (Lx, Ly)
        
        logger.info(f"Generating Neper microstructure: n={n_grains}, Domain=[{Lx}x{Ly}]")

        neper_data = self._run_neper(n_grains, domain_size, work_dir)
        
        if neper_data:
            centers = neper_data['centers']
            orientations = neper_data['orientations']
            
            # Save to simple text file: x y phi1 Phi phi2
            # 2D centers: (N, 2), Orientations: (N, 3)
            # Combine
            try:
                data = np.hstack((centers, orientations))
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                header = "x y phi1 Phi phi2"
                np.savetxt(filepath, data, header=header)
                logger.info(f"Saved seeds to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save seeds: {e}")

    def _run_neper(self, n_grains: int, domain_size: Tuple[float, float], output_dir: str):
        """
        Runs Neper to generate seeds.
        """
        os.makedirs(output_dir, exist_ok=True)
        file_prefix = os.path.join(output_dir, "poly_2d")

        # 1. CONSTRUCT NEPER COMMAND
        # -dim 2
        # -domain "square(Lx,Ly)"
        Lx, Ly = domain_size
        
        cmd_gen = [
            "neper", "-T",
            "-n", str(n_grains),
            "-dim", "2",
            "-domain", f"square({Lx},{Ly})",
            "-ori", "uniform",
            "-format", "tess",
        ]
        # Adding -o argument
        cmd_gen.extend(["-o", file_prefix])

        try:
            logger.info(f"Running: {' '.join(cmd_gen)}")
            subprocess.run(cmd_gen, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"Neper failed: {e.stderr.decode()}")
            return None
        except FileNotFoundError:
            logger.error("Neper executable not found. Is it installed and in your PATH?")
            return None

        # 2. EXTRACT DATA
        # -print cell_centers,cell_euler
        cmd_extract = [
            "neper", "-T",
            "-load", file_prefix + ".tess",
            "-print", "cell_centers,cell_euler" 
        ]

        try:
            result = subprocess.run(cmd_extract, check=True, stdout=subprocess.PIPE, text=True)
            
            # Parse output
            lines = result.stdout.strip().split('\n')
            data = []
            for line in lines:
                if line.startswith('**') or not line.strip():
                    continue
                try:
                    data.append([float(x) for x in line.split()])
                except ValueError:
                    continue
                
            data_arr = np.array(data)
            
            # Neper 2D usually outputs: x y z? or x y?
            if data_arr.size == 0:
                 logger.error("Neper returned empty data")
                 return None

            ncols = data_arr.shape[1]
            if ncols >= 5:
                # Assume first 2 are X, Y. Last 3 are Euler.
                if ncols == 5:
                    centers = data_arr[:, 0:2]
                    orientations = data_arr[:, 2:5]
                elif ncols == 6:
                     # x y z phi1 Phi phi2
                     centers = data_arr[:, 0:2]
                     orientations = data_arr[:, 3:6]
                else:
                    centers = data_arr[:, 0:2]
                    orientations = data_arr[:, -3:]

                logger.info(f"Loaded {len(centers)} seeds from Neper.")
                return {'centers': centers, 'orientations': orientations}
            else:
                logger.error(f"Unexpected Neper output shape: {data_arr.shape}")
                return None

        except Exception as e:
            logger.error(f"Failed to parse Neper output: {e}")
            return None

    def _load_seeds_from_file(self, filepath: str):
        """Read seeds using numpy."""
        try:
            logger.info(f"Loading seeds from {filepath}")
            data = np.loadtxt(filepath)
            if data.ndim == 1:
                data = data.reshape(1, -1)
                
            self.seeds = []
            for row in data:
                # row: x y phi1 Phi phi2
                if row.size >= 2:
                    pos = row[0:2]
                    ori = row[2:5] if row.size >= 5 else np.zeros(3)
                    self.seeds.append(Seed(pos, ori))
                
            if self.seeds:
                logger.info(f"Loaded {len(self.seeds)} seeds. First: {self.seeds[0].position}")

        except Exception as e:
            logger.error(f"Failed to load seeds: {e}")

    def create_tree(self, seeds: Any) -> Any:
        """
        Placeholder for building the AABB or KD-Tree from the list of seeds.
        
        Args:
            seeds: List of Seed objects
            
        Returns:
            The constructed tree object (None for now).
        """
        # TODO: Implement AABB / KD-Tree construction
        logger.info("Building Microstructure Tree... (Placeholder)")
        return None

    def update(self, t: float, dt: float, temperature_field: Any) -> Dict[str, Any]:
        """
        Step the microstructure evolution.
        """
        if not self.active:
            return {}

        # TODO: interaction interaction with thermal field
        # 1. Get T_melt isotherm
        # 2. Query Tree -> find melted seeds
        # 3. Update active seeds
        
        metrics = {
            "n_grains": len(self.seeds) if self.seeds else 0
        }
        return metrics

    def finalize(self):
        if self.active:
            logger.info("Finalizing Microstructure Solver...")
