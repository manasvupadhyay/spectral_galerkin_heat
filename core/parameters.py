from dataclasses import dataclass, field
from typing import List, Optional, Any
import numpy as np

@dataclass
class NumParams:
    """Numerical parameters for the simulation."""
    dt: float
    nx: int
    ny: int
    nz: int
    t_end: float = 0.0
    output_interval: float = 1.0e-3
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
        # The good file for now is test_speed_spectral.py
        # self.z = np.linspace(0.0, self.Lz, self.nz).astype(np.float32)
        self.z = np.linspace(0.0, self.Lz, self.nz).astype(np.float32)

@dataclass
class LaserParams:
    """Laser source parameters."""
    radius: float
    absorptivity: float
    power: float = 0.0 # Base power if constant, or max power
    
@dataclass
class SimulationContext:
    """Unified context object holding all simulation configurations."""
    num: NumParams
    mat: MaterialParams
    geom: GeomParams
    laser: LaserParams
    # laser_path will be injected separately or added here if it's a strongly typed object
    laser_path: Any = None 
