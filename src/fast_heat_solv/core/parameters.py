"""Core parameters and SimulationContext definition."""

# Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique
#
# Author: Théo Andrieux
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


__author__ = "Théo Andrieux"
__copyright__ = "Copyright 2026, LMS, École Polytechnique"

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, TYPE_CHECKING
import numpy as np
import os

from fast_heat_solv.core.vector import Vec3

if TYPE_CHECKING:
    from fast_heat_solv.core.laser import LaserPath

def _get_value(v):
    """Accept a plain scalar or a {value: ..., unit: ...} mapping."""
    if isinstance(v, dict):
        return v['value']
    return v


@dataclass
class NumParams:
    """
    Numerical parameters for the simulation.

    These parameters control time-stepping accuracy and spatial discretization.
    The spectral method's accuracy depends critically on adequate resolution.

    Attributes
    ----------
    dt : float
        Time step size in seconds. 
    nx : int
        Number of spectral modes in the x direction.
    ny : int
        Number of spectral modes in the y direction.
    nz : int
        Number of spectral modes in the z direction.
    t_end : float, optional
        Total simulation duration in seconds, by default 0.0.
        The solver runs from t=0 to t=t_end with time step dt.
    update_interval : float, optional
        Wall-clock or simulation-time interval between logging/output updates in seconds,
        by default 1e-3.
    save_all : bool, optional
        If True, save the full temperature field at every time step;  default False.
    """
    dt: float
    nx: int
    ny: int
    nz: int
    t_end: float = 0.0
    n_steps: int = 0
    dt_nominal: float = 0.0
    update_interval: float = 1e-3
    save_all: bool = False

@dataclass
class MaterialParams:
    """
    Material properties for the simulation.

    These properties define the thermal and thermodynamic behavior of the workpiece.
    All temperatures must be in Kelvin; all energies in joules per unit mass.

    Attributes
    ----------
    name : str, optional
        Material identifier (e.g., "316L", "Aluminum"). Used for logging and output,
        by default "Material".
    rho : float, optional
        Density in kg/m³. Determines heat capacity and latent heat effects. Typical
        range: 2700 (Al) to 8960 (Cu) kg/m³, by default 1.0.
    k : float, optional
        Thermal conductivity in W/(m·K). Controls heat diffusion rate and cooling speed.
        Critical for predicting melt pool shape and solidification. Typical range:
        15–429 W/(m·K), by default 1.0.
    Cp : float, optional
        Specific heat capacity in J/(kg·K). Energy required to raise material
        temperature by 1 K. Typical range: 385–4180 J/(kg·K), by default 1.0.
    L_f : float, optional
        Latent heat of fusion in J/kg. Energy released/absorbed during solid↔liquid
        phase transition (around melting point). Set to 0.0 for isothermal models,
        by default 0.0.
    T_solidus : float, optional
        Solidus temperature in Kelvin. Below this, material is fully solid.
        Must be < T_liquidus. For single-phase analysis, set both to the melting point,
        by default 0.0.
    T_liquidus : float, optional
        Liquidus temperature in Kelvin. Above this, material is fully liquid.
        by default 0.0.
    Pa : float, optional
        Ambient (atmospheric) pressure in Pa. Used for evaporation calculations.
        Standard: 101325 Pa, by default 0.
    R_v : float, optional
        Specific gas constant of the vapor in J/(kg·K). For Ar or vapor phase.
        by default 0.
    T_boil : float, optional
        Boiling temperature in Kelvin. Above this, material evaporates.
        For 316L steel: ~3090 K, by default 0.
    DeltaH_LV : float, optional
        Latent heat of vaporization in J/kg. Energy released during liquid→vapor
        transition. Typically 1–10 MJ/kg depending on material, by default 0.
    T0 : float, optional
        Initial/ambient temperature in Kelvin. All temperatures computed relative
        to T0 as reference. Typical: 293 K (room temperature), by default 0.
    h_conv : float, optional
        Convective heat transfer coefficient in W/(m²·K) at the domain boundary.
        Controls boundary cooling (e.g., bottom surface). Typical: 50–5000 W/(m²·K),
        by default 0.0.
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
    T0: float = 0
    h_conv: float = 0.0
    # Add more fields as needed from your YAML/config

    @property
    def diff(self) -> float:
        """
        Thermal diffusivity in m²/s.

        Computed as k / (rho * Cp), this dimensionless group governs the rate of
        heat diffusion. Higher values → faster heat propagation. Controls the
        characteristic time scale for thermal evolution independent of domain size.

        Returns
        -------
        float
            Thermal diffusivity in m²/s.
        """
        return self.k / (self.rho * self.Cp)

@dataclass
class GeomParams:
    """Rectangular domain ``[0, Lx] × [0, Ly] × [0, Lz]`` with a uniform spectral grid.

    The geometry is stored as grouped ``(x, y, z)`` triples (:class:`Vec3`)
    rather than loose scalars, so e.g. the spacing is ``geom.d.x`` /
    ``geom.d`` (the whole triple) instead of ``geom.dx``.

    Attributes
    ----------
    size : Vec3
        Domain extent ``(Lx, Ly, Lz)`` in metres.
    n : Vec3
        Mesh counts ``(nx, ny, nz)``.
    d : Vec3
        Grid spacing ``(dx, dy, dz) = size / n`` (computed in ``__post_init__``).
    """
    size: Vec3
    n: Vec3
    d: Vec3 = field(init=False)

    def __post_init__(self):
        self.d = self.size / self.n

@dataclass
class LaserParams:
    """
    Laser source parameters.

    Attributes
    ----------
    radius : float
        Beam radius (meters). Typically 30–100 μm for additive manufacturing.
    absorptivity : float
        Absorptivity coefficient (0–1, dimensionless). Fraction of incident power
        absorbed by material; rest is reflected.
    power : float, optional
        Nominal/maximum laser power (watts), by default 0.0. Actual power may vary
        via :class:`LaserPath.get_state`.
    """
    radius: float
    absorptivity: float
    power: float = 0.0

@dataclass
class SimulationContext:
    """
    Complete simulation configuration: numerics, material, domain, laser, and I/O.

    Attributes
    ----------
    num : NumParams
        Numerical parameters (time step, grid resolution, duration).
    mat : MaterialParams
        Material thermal and thermodynamic properties.
    geom : GeomParams
        Domain geometry and grid definition.
    laser : LaserParams
        Laser beam parameters (radius, absorptivity, nominal power).
    laser_path : LaserPath
        Laser trajectory and power evolution over time.
    io : dict
        I/O configuration (output intervals, visualization planes, etc.).
    method : str, optional
        Solver method ('spectral' or 'fem'), by default 'spectral'.
    backend : str, optional
        Compute backend ('cpu' or 'gpu'), by default 'cpu'.
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
        t_end = float(_get_value(sim_cfg.get('duration', 0.01)))
        dt_nominal = float(real_t(_get_value(sim_cfg['dt'])))
        n_steps = round(t_end / dt_nominal)
        dt = t_end / n_steps  # corrected: n_steps * dt == t_end exactly

        num_params = NumParams(
            dt=dt,
            nx=int(nx),
            ny=int(ny),
            nz=int(nz),
            t_end=t_end,
            n_steps=n_steps,
            dt_nominal=dt_nominal,
            update_interval=float(sim_cfg.get('update_interval', 1e-3))
        )

        geom_params = GeomParams(
            size=Vec3(float(Lx), float(Ly), float(Lz)),
            n=Vec3(int(nx), int(ny), int(nz)),
        )

        mat_cfg = cfg.get('material', {})
        mat_params = MaterialParams(
            name=mat_cfg.get('name', 'Material'),
            rho=real_t(_get_value(mat_cfg['rho'])),
            k=real_t(_get_value(mat_cfg['k'])),
            Cp=real_t(_get_value(mat_cfg['Cp'])),
            L_f=real_t(_get_value(mat_cfg.get('L_f', 0.0))),
            T_solidus=real_t(_get_value(mat_cfg.get('T_solidus', 0.0))),
            T_liquidus=real_t(_get_value(mat_cfg.get('T_liquidus', 0.0))),
            Pa=real_t(_get_value(mat_cfg.get('Pa', 0.0))),
            R_v=real_t(_get_value(mat_cfg.get('R_v', 0.0))),
            T_boil=real_t(_get_value(mat_cfg.get('T_boil', 0.0))),
            DeltaH_LV=real_t(_get_value(mat_cfg.get('DeltaH_LV', 0.0))),
            T0=real_t(_get_value(mat_cfg.get('T0', 0.0))),
            h_conv=real_t(_get_value(mat_cfg.get('h_conv', 0.0)))
        )

        laser_cfg = cfg.get('laser', {})
        laser_params = LaserParams(
            radius=real_t(_get_value(laser_cfg['radius'])),
            absorptivity=real_t(_get_value(laser_cfg['absorptivity'])),
            power=real_t(_get_value(laser_cfg.get('power_nominal')))
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
                        # Standardize on config_dir / paths / gcode_file
                        gcode_file = os.path.abspath(os.path.join(config_dir, 'paths', gcode_file))
                    else:
                        gcode_file = os.path.abspath(os.path.join(os.getcwd(), 'simulations', 'config', 'paths', gcode_file))
                
                # Write absolute path back to config so runner.py can access it
                cfg['laser']['path']['file'] = gcode_file
                laser_path = GCodeLaserPath(gcode_file, initial_position=initial_position)
                
        io_cfg = cfg.get('io', {})
        return cls(num=num_params, mat=mat_params, geom=geom_params, laser=laser_params, laser_path=laser_path, io=io_cfg, method=sim_method, backend=sim_backend)
