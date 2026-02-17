import logging
import numpy as np
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from core.parameters import SimulationContext
from interfaces.microstructure import MicrostructureSolver
from scipy.spatial import cKDTree
from implementations.physics.spectral_cpu_kernels import IDCT_II

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
            # Generate seeds using Neper and save to output_dir/seeds/tessellation_3d.tess
            seeds_dir = os.path.join(self.output_dir, "seeds")
            # We know _run_neper produces tessellation_3d.tess
            seeds_file = os.path.join(seeds_dir, "tessellation_3d.tess")
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
        
        # Use simulation bounds
        Lx = self.context.geom.Lx
        Ly = self.context.geom.Ly
        domain_size = (Lx, Ly)
        
        n_grains_log = gen_params.get('n_grains', 'custom')
        logger.info(f"Generating Neper microstructure: n={n_grains_log}, Domain=[{Lx}x{Ly}]")

        self._run_neper(domain_size, work_dir)

    def _run_neper(self, domain_size: Tuple[float, float], output_dir: str):
        """
        Runs Neper to generate seeds.
        """
        os.makedirs(output_dir, exist_ok=True)
        # Use tessellation_3d prefix as in the snippet
        file_prefix = os.path.join(output_dir, "tessellation_3d")

        # 1. CONSTRUCT NEPER COMMAND
        Lx, Ly = domain_size
        
        if hasattr(self.context.geom, 'Lz'):
            Lz = self.context.geom.Lz
            dim = "3"
            domain = f"cube({Lx},{Ly},{Lz})"
        else:
            # Fallback or assume 2D if no Z
            Lz = 1.0 # Dummy
            dim = "3" 
            domain = f"cube({Lx},{Ly},{Lz})"

        gen_params = self.params.generation_params
        
        # Base Command
        cmd_base = ["neper", "-T"]
        
        # Determine Arguments
        if 'command' in gen_params and gen_params['command']:
            # USER PROVIDED COMMAND STRINGS
            # e.g. "-n 100 -morpho gg"
            custom_args = gen_params['command'].split()
            # Remove "neper" or "-T" if user included them by mistake
            cleaned_args = []
            skip_next = False
            for i, arg in enumerate(custom_args):
                if arg in ["neper", "-T"]: continue
                # We also want to strip -dim or -domain if user provided them, to avoid conflicts/double entry
                # Simpler: just use user command as args. But we must ensure domain matches simulation.
                # Neper allows multiple flags, last one usually wins? Or error.
                # To be safe, we append our critical flags (domain, format) at the end.
                cleaned_args.append(arg)

            cmd_args = cleaned_args
            logger.info(f"Using custom Neper arguments: {cleaned_args}")
        else:
            # DEFAULT / SIMPLE MODE
            n_grains = gen_params.get('n_grains', 100)
            morpho = gen_params.get('shape', 'voronoi')
            if morpho == 'circle': morpho = 'voronoi' # compat
            
            cmd_args = [
                "-n", str(n_grains),
                "-morpho", morpho,
                "-ori", "uniform"
            ]
        
        # formatting options
        cmd_final = cmd_base + cmd_args + [
            "-dim", dim,
            "-domain", domain,
            "-format", "tess,tesr", # Request both regular tessellation and raster
            "-tesrformat", "ascii",
            "-o", file_prefix
        ]

        try:
            logger.info(f"Running: {' '.join(cmd_final)}")
            subprocess.run(cmd_final, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 2. VISUALIZATION (Optional if povray is installed)
            tess_file = f"{file_prefix}.tess"
            if os.path.exists(tess_file):
                viz_output = os.path.join(output_dir, "visualization") # Neper adds extension automatically
                cmd_viz = [
                    "neper", "-V", tess_file,
                    "-datacellcol", "ori",
                    "-print", viz_output
                ]
                logger.info(f"Generating visualization: {' '.join(cmd_viz)}")
                # Allow failure for visualization (e.g. missing povray) without crashing simulation
                try:
                    subprocess.run(cmd_viz, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Neper visualization failed (is metrics/povray installed?): {e.stderr.decode()}")
                except FileNotFoundError:
                    logger.warning("Neper executable not found for visualization.")

        except subprocess.CalledProcessError as e:
            logger.error(f"Neper failed: {e.stderr.decode()}")
            return None
        except FileNotFoundError:
            logger.error("Neper executable not found. Is it installed and in your PATH?")
            return None
        

    def _load_seeds_from_file(self, filepath: str):
        """Read seeds from file. Supports .txt and .tess formats."""
        if not os.path.exists(filepath):
            logger.error(f"Seed file not found: {filepath}")
            return

        ext = os.path.splitext(filepath)[1]
        
        if ext == '.tess':
            self._load_neper_tess(filepath)
        else:
            # Fallback to simple txt
            try:
                data = np.loadtxt(filepath, skiprows=1)  # Skip header
                # TODO: Convert to Seed objects
            except Exception as e:
                logger.error(f"Failed to load seeds: {e}")

    def _load_neper_tess(self, filepath: str):
        """
        Parses a Neper .tess file to extract seed positions and properties.
        """
        logger.info(f"Loading seeds from Neper tessellation: {filepath}")
        
        seeds_positions = []
        seeds_orientations = []
        n_grains = 0
        
        try:
            with open(filepath, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                if line == "**cell":
                    i += 1
                    if i < len(lines):
                        try:
                            n_grains = int(lines[i])
                            logger.info(f"Expecting {n_grains} grains based on **cell header.")
                        except ValueError:
                            logger.warning(f"Could not parse element count after **cell: {lines[i]}")
                
                elif line == "*seed":
                    i += 1 # Move to data
                    # Read n_grains lines
                    # Format: id x y z w
                    count = 0
                    while count < n_grains and i < len(lines):
                        parts = lines[i].split()
                        if len(parts) >= 4:
                            # 1:4 are x, y, z
                            pos = np.array([float(p) for p in parts[1:4]])
                            seeds_positions.append(pos)
                        count += 1
                        i += 1
                    continue # Skip the increments at bottom

                elif line == "*ori":
                    i += 1
                    if i < len(lines):
                        descriptor = lines[i] # e.g. "rodrigues:active"
                        logger.info(f"Orientation descriptor: {descriptor}")
                        i += 1
                    
                    # Read n_grains lines
                    count = 0
                    while count < n_grains and i < len(lines):
                        parts = lines[i].split()
                        if len(parts) > 0:
                            ori = np.array([float(p) for p in parts])
                            seeds_orientations.append(ori)
                        count += 1
                        i += 1
                    continue

                i += 1

            # Create Seed objects
            self.seeds = []
            if len(seeds_positions) == n_grains:
                # If orientations are missing, provide identity/zeros
                if len(seeds_orientations) != n_grains:
                    logger.warning(f"Orientation count ({len(seeds_orientations)}) != Seed count ({n_grains}). Using defaults.")
                    seeds_orientations = [np.zeros(3) for _ in range(n_grains)]
                
                for pos, ori in zip(seeds_positions, seeds_orientations):
                    self.seeds.append(Seed(position=pos, orientation=ori))
                
                logger.info(f"Successfully loaded {len(self.seeds)} seeds.")
            else:
                logger.error(f"Seed parsing mismatch: Found {len(seeds_positions)} positions, expected {n_grains}.")

        except Exception as e:
            logger.error(f"Error parsing .tess file: {e}")

    def create_tree(self, seeds: Any) -> Any:
        """
        Builds a KD-Tree from the seed positions for fast spatial queries.
        Note: cKDTree is static. If seeds change (nucleation/melting), 
        this must be called again to rebuild the index.
        
        Args:
            seeds: List of Seed objects
            
        Returns:
            scipy.spatial.cKDTree: The constructed spatial index.
        """
        if not seeds:
            logger.warning("No seeds provided to create_tree.")
            return None

        # Extract positions into (N, 3) array
        # Assuming points are 3D. If 2D, they will be (N, 2)
        try:
            points = np.array([s.position for s in seeds])
            
            logger.info(f"Building cKDTree for {len(seeds)} seeds...")
            tree = cKDTree(points)
            return tree
        except Exception as e:
            logger.error(f"Failed to build search tree: {e}")
            return None

    def update(self, t: float, dt: float, temperature_field: Any) -> Dict[str, Any]:
        """
        Step the microstructure evolution.
        Args:
            temperature_field: Spectral modes 'a' of the temperature solution. 
                               Needs IDCT to get physical T.
        """
        if not self.active or self.tree is None:
            return {}

        T_S = self.context.mat.T_solidus
        rho = 2e-4  # Search radius (default 200um), could be a parameter

        # 1. Reconstruct Temperature Field (IDCT)
        # Note: This is an expensive operation (O(N log N)) 
        # performed at every micro step.
        try:
            T_real = IDCT_II(temperature_field)
        except Exception as e:
            logger.error(f"Failed to reconstruct T field: {e}")
            return {}

        # 2. Identify Melting / Isotherm
        # Create a mask of the "Melt Pool" (T > T_solidus)
        # We want to find seeds close to the boundary (T ~ T_S) 
        # But specifically, we must mark seeds inside the melt pool as "liquid" (phase change).
        
        # Grid Coordinates (We need to map grid indices to physical coordinates)
        # Assuming uniform grid for now, based on geom and shape of T_real
        nz, ny, nx = T_real.shape
        Lx, Ly, Lz = self.context.geom.Lx, self.context.geom.Ly, getattr(self.context.geom, 'Lz', 0.0)
        
        # Coordinate arrays (1D)
        # Check alignment: typically z is first dim in spectral solver here
        z_coords = np.linspace(0, -Lz, nz) # top is 0, bottom is -Lz
        y_coords = np.linspace(-Ly/2, Ly/2, ny) # centered? Need to check geom def.
        x_coords = np.linspace(0, Lx, nx)
        
        # Create meshgrid of coordinates corresponding to the T_real grid
        # Indexing 'ij' gives (nz, ny, nx) if we pass (z, y, x)
        # Z, Y, X = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        
        # OPTIMIZATION: Instead of full meshgrid, query the Tree with the specific points
        # that are "Near Solidus".
        
        # Find indices where T > T_S (Melted)
        melted_indices = np.where(T_real > T_S)
        
        # If nothing is melted, return
        if len(melted_indices[0]) == 0:
            return {"n_grains": len(self.seeds)}

        # Convert indices to physical coordinates [x, y, z]
        # melted_indices is tuple (z_idx, y_idx, x_idx)
        z_melt = z_coords[melted_indices[0]]
        y_melt = y_coords[melted_indices[1]]
        x_melt = x_coords[melted_indices[2]]
        
        melt_points = np.column_stack((x_melt, y_melt, z_melt)) # Shape (N_melt, 3)

        # 3. Query Tree -> Find seeds within the Melt Pool
        # Actually, if a seed is IN the melt pool, it melts (re-initializes or dies).
        # We query for all seeds within 'rho' of ANY melt point? 
        # Or just checking which seeds are inside the region?
        
        # The prompt asks for "closest seeds to this isotherm in a radius rho"
        # Isotherm points are those at the boundary.
        # But functionally, we usually want to know which grains are consumed.
        
        # Let's perform a query for seeds close to the melt volume 
        # (effectively union of balls around melted grid points)
        # This covers the "isotherm + inward" region.
        
        indices_near_melt = self.tree.query_ball_point(melt_points, r=rho)
        
        # Flatten and unique
        affected_seeds_indices = np.unique([idx for sublist in indices_near_melt for idx in sublist])
        
        # TODO: Apply physics (e.g. change phase to liquid, re-orient if valid)
        # For now, we just log the count.
        
        metrics = {
            "n_grains": len(self.seeds),
            "n_melted_seeds": len(affected_seeds_indices)
        }
        return metrics

    def finalize(self):
        if self.active:
            logger.info("Finalizing Microstructure Solver...")
