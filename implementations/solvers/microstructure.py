import logging
import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple
from core.parameters import SimulationContext
from interfaces.microstructure import MicrostructureSolver

logger = logging.getLogger(__name__)

# Try importing MicroStructPy
try:
    import microstructpy as msp
    HAS_MICROSTRUCTPY = True
except ImportError:
    HAS_MICROSTRUCTPY = False
    logger.warning("MicroStructPy not found. Microstructure simulation will be limited.")

class TreeMicroSolver(MicrostructureSolver):
    """
    Concrete implementation of MicrostructureSolver using a Tree-based spatial index (AABB/KD-Tree)
    to manage grain seeds and their evolution.
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
           output_dir (str): Directory to save intermediate files (seeds.txt).
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
            # Generate seeds and save to output_dir/seeds/seeds.txt
            seeds_file = os.path.join(self.output_dir, "seeds", "seeds.txt")
            self._generate_and_save_seeds(seeds_file)
        
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
            logger.info(f"Microstructure initialized with {len(self.seeds)} seeds.")
        
        return self.seeds

    def _generate_and_save_seeds(self, filepath: str):
        """
        Generate synthetic seeds using MicroStructPy API and write to file.
        Based on minimal example: Phase (shape, size) + Domain.
        """
        if not HAS_MICROSTRUCTPY:
            logger.error("MicroStructPy not installed.")
            return
            
        gen_params = self.params.generation_params
        
        try:
            # 1. Define Phase 
            # Default to circle/0.15mm if not provided (minimal example)
            shape = gen_params.get('shape')
            size  = gen_params.get('size') # default somewhat adjusted to SI if needed
            
            phase = {'shape': shape, 'size': size, 'orientation': 'random'} # orientation can be randomized later if needed
            
            # 2. Define Domain
            # Use simulation bounds
            Lx = self.context.geom.Lx
            Ly = self.context.geom.Ly
            # MicroStructPy 2D Geometry
            # Domain is [0, Lx] x [0, Ly].
            
            # Use Rectangle from corners to match simulation domain
            domain = msp.geometry.Rectangle(
                corner=(0, 0),
                side_lengths=(Lx, Ly)
            )
            
            logger.info(f"Generating seeds: Phase={phase}, Domain=[{Lx}x{Ly}]")

            # 3. Create Unpositioned Seeds
            # Ensure domain area is positive
            if domain.area <= 0:
                 logger.error("Domain area is zero or negative.")
                 return

            seeds = msp.seeding.SeedList.from_info(phase, domain.area)
            # Info number of seeds created 
            logger.info(f"Created {len(seeds)} unpositioned seeds.")
            # 4. Position Seeds
            seeds.position(domain, verbose = True)
            logger.info(f"Finished positionning {len(seeds)} seeds.")
            # Plot the positioned seeds and save to a PNG.
            # Use a non-interactive backend so this works headless.
            # --- VORONOI PLOTTING START ---
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import matplotlib as mpl

                # 1. Create Voronoi Mesh from positioned seeds
                pmesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)

                # 2. Calculate Colors based on grain area
                n = len(seeds)
                areas = pmesh.volumes
                std_area = domain.area / n
                min_area, max_area = min(areas), max(areas)

                cell_colors = np.zeros((n, 3))
                for i in range(n):
                    if areas[i] < std_area:
                        # Blue to White scale
                        f = (areas[i] - min_area) / (max_area - min_area) # Normalizing across full range for simplicity
                        cell_colors[i] = (f, f, 1.0)
                    else:
                        # White to Red scale
                        f = (max_area - areas[i]) / (max_area - min_area)
                        cell_colors[i] = (1.0, f, f)

                # 3. Setup Plot
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Plot the Voronoi cells
                pmesh.plot(edgecolors='k', facecolors=cell_colors, linewidth=0.5)
                
                # Optional: Overlay the seed points (transparent with black edge)
                seeds.plot(edgecolors='k', facecolors='none', alpha=0.3)

                plt.axis('square')
                plt.xlim(domain.limits[0])
                plt.ylim(domain.limits[1])

                # 4. Add Colorbar
                colors = [(0, (0, 0, 1)), (0.5, (1, 1, 1)), (1, (1, 0, 0))]
                cmap = mpl.colors.LinearSegmentedColormap.from_list('area_cmap', colors)
                norm = mpl.colors.Normalize(vmin=min_area, vmax=max_area)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                cb = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08)
                cb.set_label('Cell Area ($mm^2$)')

                # 5. Save
                fig_dir = os.path.join(self.output_dir or '.', 'seeds')
                os.makedirs(fig_dir, exist_ok=True)
                fig_path = os.path.join(fig_dir, 'voronoi_diagram.png')
                
                plt.savefig(fig_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                logger.info(f"Saved Voronoi plot to {fig_path}")

            except Exception as e:
                logger.error(f"Voronoi plotting failed: {e}")
            
            # 5. Write to File using SeedList.write()
            # MicroStructPy's SeedList.write expects a filename (str), not a file object.
            # Verify directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Pass the filepath string directly as required by the library
            seeds.write(filepath)
            logger.info(f"Saved seeds to {filepath}")

        except Exception as e:
            logger.error(f"MicroStructPy generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _load_seeds_from_file(self, filepath: str):
        """Read seeds using MicroStructPy SeedList.from_file."""
        if not HAS_MICROSTRUCTPY:
             return

        try:
            logger.info(f"Loading seeds from {filepath}")
            # MicroStructPy loader
            self.seeds = msp.seeding.SeedList.from_file(filepath)
            
            # Check content
            if len(self.seeds) > 0:
                logger.info(f"Loaded {len(self.seeds)} seeds. First at: {self.seeds[0].position}")
        except Exception as e:
            logger.error(f"Failed to load seeds: {e}")

    def create_tree(self, seeds: Any) -> Any:
        """
        Placeholder for building the AABB or KD-Tree from the list of seeds.
        
        Args:
            seeds: List of seed objects (MicroStructPy objects or tuples)
            
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
