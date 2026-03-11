from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, TYPE_CHECKING
import numpy as np
import os

if TYPE_CHECKING:
    from interfaces.laser import LaserPath

@dataclass
class NumParams:
    """Numerical parameters for the simulation."""
    dt: float
    nx: int
    ny: int
    nz: int
    t_end: float = 0.0
    update_interval: float = 1e-3  # Time interval for logging updates
    save_all: bool = False

@dataclass
class MaterialParams:
    """Material properties."""
    name: str = "Material"
    rho: float = 1.0
    k: float = 1.0
    Cp: float = 1.0
    L_f: float = 0.0
    T_solidus: float = 0.0
    T_liquidus: float = 0.0
    Pa: float = 0
    R_v: float = 0
    T_boil: float = 0
    DeltaH_LV: float = 0
    T0: float = 0 # Reference temperature
    h_conv: float = 0.0  # Convective heat transfer coefficient (W/(m²·K))
    # Add more fields as needed from your YAML/config

    @property
    def diff(self) -> float:
        """Thermal diffusivity."""
        return self.k / (self.rho * self.Cp)

@dataclass
class GeomParams:
    """Geometric parameters and grid generation."""
    Lx: float
    Ly: float
    Lz: float
    nx: int
    ny: int
    nz: int
    
    # Grid arrays (initialized in __post_init__ or property)
    x: np.ndarray = field(init=False, default=None)
    y: np.ndarray = field(init=False, default=None)
    z: np.ndarray = field(init=False, default=None)
    dx: float = field(init=False)
    dy: float = field(init=False)
    dz: float = field(init=False)

    def __post_init__(self):
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz
        
        self.x = np.linspace(0.0, self.Lx, self.nx, endpoint=False).astype(np.float32)
        self.y = np.linspace(0.0, self.Ly, self.ny, endpoint=False).astype(np.float32)
        # Check if z endpoint should be included or not. Usually for spectral in Z we might want specific BCs.
        self.z = np.linspace(0.0, self.Lz, self.nz).astype(np.float32)

@dataclass
class LaserParams:
    """Laser source parameters."""
    radius: float
    absorptivity: float
    power: float = 0.0 # Base power if constant, or max power

@dataclass
class SimulationContext:
    """
    Aggregate context holding all simulation parameters.
    io: flat dictionary from YAML config with keys:
        - interval: float, output interval for time-stepped outputs
        - outputs: list[str], outputs to save at each interval
        - at_end: list[str], outputs to save at the end
        - profiles_locations: list[list[float]], locations for profiles
        - cut_views_planes: list[str], planes for cut views
    """
    num: 'NumParams'
    mat: 'MaterialParams'
    geom: 'GeomParams'
    laser: 'LaserParams'
    laser_path: 'LaserPath' # Use forward reference
    
    # Existing fields
    io: Dict[str, Any]  # Flat dict with new keys
    # Execution configuration
    method: str = "spectral"  # "spectral" or "fem"
    backend: str = "cpu"      # "cpu" or "gpu"

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> 'SimulationContext':
        real_t = np.float32
        sim_cfg = cfg.get('simulation', {})
        domain_cfg = cfg.get('domain', {})
        sim_method = sim_cfg.get('method', 'spectral').lower()
        sim_backend = sim_cfg.get('backend', 'cpu').lower()
        Lx, Ly, Lz = domain_cfg['size']
        nx, ny, nz = domain_cfg['mesh']
        t_end = sim_cfg.get('duration', 0.01)
        dt = real_t(sim_cfg['dt'])
        
        num_params = NumParams(
            dt=float(dt),
            nx=int(nx),
            ny=int(ny),
            nz=int(nz),
            t_end=float(t_end),
            update_interval=float(sim_cfg.get('update_interval', 1e-3))
        )

        geom_params = GeomParams(
            Lx=float(Lx), Ly=float(Ly), Lz=float(Lz),
            nx=int(nx), ny=int(ny), nz=int(nz)
        )

        mat_cfg = cfg.get('material', {})
        mat_params = MaterialParams(
            name=mat_cfg.get('name', 'Material'),
            rho=real_t(mat_cfg['rho']),
            k=real_t(mat_cfg['k']),
            Cp=real_t(mat_cfg['Cp']),
            L_f=real_t(mat_cfg.get('L_f', 0.0)),
            T_solidus=real_t(mat_cfg.get('T_solidus', 0.0)),
            T_liquidus=real_t(mat_cfg.get('T_liquidus', 0.0)),
            Pa=real_t(mat_cfg.get('Pa', 0.0)),
            R_v=real_t(mat_cfg.get('R_v', 0.0)),
            T_boil=real_t(mat_cfg.get('T_boil', 0.0)),
            DeltaH_LV=real_t(mat_cfg.get('DeltaH_LV', 0.0)),
            T0=real_t(mat_cfg.get('T0', 0.0)),
            h_conv=real_t(mat_cfg.get('h_conv', 0.0))
        )

        laser_cfg = cfg.get('laser', {})
        laser_params = LaserParams(
            radius=real_t(laser_cfg['radius']),
            absorptivity=real_t(laser_cfg['absorptivity']),
            power=real_t(laser_cfg.get('power_nominal'))
        )
        
        # Laser Path
        path_cfg = laser_cfg.get('path', {})
        laser_path = None
        if path_cfg.get('type', '').lower() == 'gcode':
            from utils.gcode_path import GCodeLaserPath
            gcode_file = path_cfg.get('file', None)
            initial_position = tuple(path_cfg.get('initial_position', [0.0, 0.0]))
            if gcode_file is not None:
                if not os.path.isabs(gcode_file):
                     # parameters.py is in core/, so we need dirname(dirname(__file__)) to reach root
                     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                     gcode_file = os.path.join(base_dir, 'config', 'paths', gcode_file)
                laser_path = GCodeLaserPath(gcode_file, initial_position=initial_position)
                
        io_cfg = cfg.get('io', {})
        return cls(num=num_params, mat=mat_params, geom=geom_params, laser=laser_params, laser_path=laser_path, io=io_cfg, method=sim_method, backend=sim_backend)
