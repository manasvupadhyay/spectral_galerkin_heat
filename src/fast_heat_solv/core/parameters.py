from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, TYPE_CHECKING
import numpy as np
import os

if TYPE_CHECKING:
    from fast_heat_solv.core.laser import LaserPath

@dataclass
class NumParams:
    """
    Numerical parameters for the simulation.

    Attributes
    ----------
    dt : float
        Time step size in seconds.
    nx : int
        Number of grid points in the x direction.
    ny : int
        Number of grid points in the y direction.
    nz : int
        Number of grid points in the z direction.
    t_end : float, optional
        End time of the simulation in seconds, by default 0.0.
    update_interval : float, optional
        Time interval for logging updates in seconds, by default 1e-3.
    save_all : bool, optional
        Flag to indicate if all data should be saved, by default False.
    """
    dt: float
    nx: int
    ny: int
    nz: int
    t_end: float = 0.0
    update_interval: float = 1e-3  # Time interval for logging updates
    save_all: bool = False

@dataclass
class MaterialParams:
    """
    Material properties for the simulation.

    Attributes
    ----------
    name : str, optional
        Name of the material, by default "Material".
    rho : float, optional
        Density of the material in kg/m^3, by default 1.0.
    k : float, optional
        Thermal conductivity in W/(m·K), by default 1.0.
    Cp : float, optional
        Specific heat capacity in J/(kg·K), by default 1.0.
    L_f : float, optional
        Latent heat of fusion in J/kg, by default 0.0.
    T_solidus : float, optional
        Solidus temperature in Kelvin, by default 0.0.
    T_liquidus : float, optional
        Liquidus temperature in Kelvin, by default 0.0.
    Pa : float, optional
        Ambient pressure in Pa, by default 0.
    R_v : float, optional
        Specific gas constant of the vapor in J/(kg·K), by default 0.
    T_boil : float, optional
        Boiling temperature in Kelvin, by default 0.
    DeltaH_LV : float, optional
        Latent heat of vaporization in J/kg, by default 0.
    T0 : float, optional
        Reference or initial temperature in Kelvin, by default 0.
    h_conv : float, optional
        Convective heat transfer coefficient in W/(m^2·K), by default 0.0.
    """
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
        """
        Calculates the thermal diffusivity of the material.

        Returns
        -------
        float
            Thermal diffusivity computed as `k / (rho * Cp)`.
        """
        return self.k / (self.rho * self.Cp)

@dataclass
class GeomParams:
    """
    Geometric parameters and grid generation properties.

    Attributes
    ----------
    Lx : float
        Domain length in the x direction in meters.
    Ly : float
        Domain length in the y direction in meters.
    Lz : float
        Domain length in the z direction in meters.
    nx : int
        Number of grid points in the x direction.
    ny : int
        Number of grid points in the y direction.
    nz : int
        Number of grid points in the z direction.
    x : np.ndarray
        1D array of x coordinates.
    y : np.ndarray
        1D array of y coordinates.
    z : np.ndarray
        1D array of z coordinates.
    dx : float
        Grid spacing in the x direction.
    dy : float
        Grid spacing in the y direction.
    dz : float
        Grid spacing in the z direction.
    """
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
    """
    Laser source parameters.

    Attributes
    ----------
    radius : float
        Radius of the laser beam.
    absorptivity : float
        Absorptivity coefficient of the material for the given laser.
    power : float, optional
        Base power if constant, or maximum power of the laser, by default 0.0.
    """
    radius: float
    absorptivity: float
    power: float = 0.0 # Base power if constant, or max power

@dataclass
class SimulationContext:
    """
    Aggregate context holding all simulation parameters.

    Attributes
    ----------
    num : NumParams
        Numerical computation parameters.
    mat : MaterialParams
        Material properties parameters.
    geom : GeomParams
        Geometric and domain parameters.
    laser : LaserParams
        Laser configuration parameters.
    laser_path : LaserPath
        Object determining the laser trajectory and state over time.
    io : dict
        Flat dictionary defining input/output options (e.g., intervals, planes).
    method : str, optional
        Simulation calculation method (e.g., 'spectral' or 'fem'), by default 'spectral'.
    backend : str, optional
        Backend target for computations (e.g., 'cpu' or 'gpu'), by default 'cpu'.
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
    def from_dict(cls, cfg: Dict[str, Any], config_dir: Optional[str] = None) -> 'SimulationContext':
        """
        Parses a nested dictionary and instantiates a full `SimulationContext`.

        Parameters
        ----------
        cfg : dict
            Parsed dictionary typically loaded from a YAML configuration file.
            Should contain keys like 'simulation', 'domain', 'material',
            'laser', and 'io'.
        config_dir : str, optional
            Path to the directory containing the configuration file. Used to
            resolve relative paths for external assets like G-code files,
            by default None.

        Returns
        -------
        SimulationContext
            A populated simulation context ready to initialize solver factories.
        """
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
            from fast_heat_solv.io_utils.gcode_path import GCodeLaserPath
            gcode_file = path_cfg.get('file', None)
            initial_position = tuple(path_cfg.get('initial_position', [0.0, 0.0]))
            if gcode_file is not None:
                if not os.path.isabs(gcode_file):
                    if config_dir is not None:
                        # Ensure config_dir is absolute
                        abs_config_dir = os.path.abspath(config_dir)
                        attempt1 = os.path.join(abs_config_dir, gcode_file)
                        attempt2 = os.path.join(abs_config_dir, 'paths', gcode_file)
                        if os.path.exists(attempt1):
                            gcode_file = attempt1
                        elif os.path.exists(attempt2):
                            gcode_file = attempt2
                        else:
                            # fallback
                            gcode_file = os.path.join(abs_config_dir, 'paths', gcode_file)
                    else:
                        cwd = os.getcwd()
                        attempt1 = os.path.join(cwd, gcode_file)
                        attempt2 = os.path.join(cwd, 'config', 'paths', gcode_file)
                        attempt3 = os.path.join(cwd, 'simulations', 'config', 'paths', gcode_file)
                        
                        if os.path.exists(attempt1):
                            gcode_file = attempt1
                        elif os.path.exists(attempt2):
                            gcode_file = attempt2
                        elif os.path.exists(attempt3):
                            gcode_file = attempt3
                        else:
                            # Final fallback assuming project root is one level above src
                            gcode_file = os.path.join(cwd, 'simulations', 'config', 'paths', gcode_file)
                laser_path = GCodeLaserPath(gcode_file, initial_position=initial_position)
                
        io_cfg = cfg.get('io', {})
        return cls(num=num_params, mat=mat_params, geom=geom_params, laser=laser_params, laser_path=laser_path, io=io_cfg, method=sim_method, backend=sim_backend)
