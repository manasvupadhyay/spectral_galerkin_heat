"""Core parameters and SimulationContext definition."""

# Copyright 2026 Laboratoire de Mécanique des Solides (LMS), 
# École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris, 
# Route de Saclay, Palaiseau, 91128, France.
#
# Author: Théo Andrieux, Jules Dichamp, Manas V. Upadhyay
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


__author__ = "Théo Andrieux, Jules Dichamp, Manas V. Upadhyay"
__copyright__ = "Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris"

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, TYPE_CHECKING
import numpy as np
import os

from fast_heat_solv.core.vector import Vec3
from fast_heat_solv.core.properties import MaterialModel

if TYPE_CHECKING:
    from fast_heat_solv.core.laser import LaserPath

# ``cfg`` throughout is the parsed-YAML configuration dict consumed
# by ``SimulationContext.from_dict``.
# cfg stands for config 
# Scalar values may be a plain number or a {value, unit} mapping (see _get_value).


def _get_value(v):
    """Accept a plain scalar or a {value: ..., unit: ...} mapping."""
    if isinstance(v, dict):
        return v['value']
    return v


# Floating-point precisions the solver supports, keyed by config name.
_SUPPORTED_DTYPES = {"float32": np.float32, "float64": np.float64}


def _resolve_dtype(name):
    """Resolve a precision name (e.g. ``"float32"``) to a NumPy scalar type.

    Parameters
    ----------
    name : str
        Precision name from the ``simulation.dtype`` config key.

    Returns
    -------
    type
        ``numpy.float32`` or ``numpy.float64``.

    Raises
    ------
    ValueError
        If *name* is not a supported precision.
    """
    key = str(name).lower()
    try:
        return _SUPPORTED_DTYPES[key]
    except KeyError:
        choices = ", ".join(sorted(_SUPPORTED_DTYPES))
        raise ValueError(
            f"Unknown dtype: {name!r}. Choose one of: {choices}."
        ) from None


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
    dtype : type, optional
        Floating-point precision for the solver arrays (``numpy.float32`` or
        ``numpy.float64``), by default ``numpy.float32``. Set via the
        ``simulation.dtype`` config key.
    max_picard_iter : int, optional
        Cap on the spectral solver's fixed-point (Picard) iterations per step.
        ``None`` (default) lets the solver use its built-in default. Set via the
        ``simulation.max_picard_iter`` config key.
    picard_tol : float, optional
        Relative convergence tolerance for the Picard loop. ``None`` (default)
        uses the solver default. Set via ``simulation.picard_tol``.
    picard_omega : float, optional
        Under-relaxation (mixing) factor for the Picard loop. ``None`` (default)
        uses the solver default. Set via ``simulation.picard_omega``.
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
    # Optional Picard fixed-point controls (None -> use the solver defaults).
    # The temperature-dependent property correction needs a higher cap than the
    # base 30; expose them so the config can raise it.
    max_picard_iter: Optional[int] = None
    picard_tol: Optional[float] = None
    picard_omega: Optional[float] = None
    dtype: Any = np.float32

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
        by default 0. **Must be > 0 for any run that melts** — the evaporation
        kernel divides by it (see :func:`compute_evaporation_flux`); 0 is only
        safe when the surface never reaches ``T_liquidus``.
    T_boil : float, optional
        Boiling temperature in Kelvin. Above this, material evaporates.
        For 316L steel: ~3090 K, by default 0. **Must be > 0 for any melting
        run** (same evaporation-kernel division as ``R_v``).
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
    model : MaterialModel, optional
        Temperature-dependent property model (``k(T)``, ``rho(T)``, ``c(T)``
        polynomial branches). ``None`` for a constant-property material, in which
        case the solver uses the scalar ``rho``/``k``/``Cp`` directly. When
        present, the scalar ``rho``/``k``/``Cp`` fields hold the user-supplied
        **reference constants** (each property's in-block ``reference:`` key) that
        are baked into the ETD1 propagators; the property-correction path
        reinstates the fluctuations about them.
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
    model: Optional['MaterialModel'] = None
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
    profile : str, optional
        Beam-profile name (``"gaussian"`` / ``"flat_top"`` / ``"super_gaussian"``)
        resolved to a :class:`fast_heat_solv.core.laser.LaserProfile`, by default
        ``"gaussian"``. Set via the ``laser.profile`` config key.
    r_x, r_y : float, optional
        Beam radii along x and y (metres). Default to ``radius`` (circular beam);
        set both for an elliptical beam. Set via ``laser.r_x`` / ``laser.r_y``.
    super_gaussian_order : float, optional
        Super-Gaussian order ``n`` used when ``profile == "super_gaussian"``
        (``2`` = Gaussian, large = flat-top), by default 2.0. Set via
        ``laser.super_gaussian_order``.
    cell_integrated : bool, optional
        When True, the Gaussian flux is integrated analytically over each grid
        cell (exact power deposition even for a beam narrower than the grid)
        instead of being point-sampled at the cell centre. Gaussian only.
        Default False. Set via ``laser.cell_integrated``.
    """
    radius: float
    absorptivity: float
    power: float = 0.0
    profile: str = "gaussian"
    r_x: float = 0.0
    r_y: float = 0.0
    super_gaussian_order: float = 2.0
    cell_integrated: bool = False

    def __post_init__(self):
        # Default to a circular beam (r_x = r_y = radius) when axes are unset.
        if not self.r_x:
            self.r_x = self.radius
        if not self.r_y:
            self.r_y = self.radius

    @property
    def ref_area(self) -> float:
        """Reference area ``r_x · r_y`` (``r_b²`` for a circular beam).

        Used by :meth:`LaserProfile.peak_intensity` to enforce
        ``∬ q dA = A·P``.
        """
        return self.r_x * self.r_y

@dataclass
class FineMeshParams:
    """Moving fine-mesh parameters — a refined, laser-following sub-box of the domain — for latent-heat / nonlinear terms.

    The solver reconstructs temperature on a refined box that tracks the laser,
    to resolve the sharp mushy-zone gradients the coarse spectral grid cannot.

    Attributes
    ----------
    refinement : int
        Per-axis cell-refinement factor of the fine mesh relative to the global
        spectral grid (fine spacing ``= geom.d / refinement``), by default 4.
    box_size : Vec3
        Extent of the refined, laser-following sub-box in metres ``(Lx_box, Ly_box, Lz_box)``: the
        x/y extents span the laser footprint, the z extent is the near-surface
        depth. By default ``Vec3(0.9e-3, 0.9e-3, 0.04e-3)``.
    """
    refinement: int = 4
    box_size: Vec3 = Vec3(0.9e-3, 0.9e-3, 0.04e-3)

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
    backend : str, optional
        Compute backend ('cpu', 'gpu', or 'cpu_linear'), by default 'cpu'.
    fine : FineMeshParams, optional
        Moving fine-mesh parameters — the refined, laser-following sub-box
        (refinement, box extents). Defaults to
        :class:`FineMeshParams` defaults.
    """
    num: 'NumParams'
    mat: 'MaterialParams'
    geom: 'GeomParams'
    laser: 'LaserParams'
    laser_path: 'LaserPath' # Use forward reference

    # Existing fields
    io: Dict[str, Any]  # Flat dict with new keys
    # Execution configuration
    backend: str = "cpu"      # "cpu", "gpu", or "cpu_linear"
    fine: 'FineMeshParams' = field(default_factory=FineMeshParams)

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
            A populated simulation context ready to build and initialize a solver.
        """
        real_t = _resolve_dtype(cfg.get('simulation', {}).get('dtype', 'float32'))
        sim_cfg = cfg.get('simulation', {})
        domain_cfg = cfg.get('domain', {})
        sim_backend = sim_cfg.get('backend', 'cpu').lower()
        Lx, Ly, Lz = _get_value(domain_cfg['size'])
        nx, ny, nz = _get_value(domain_cfg['mesh'])
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
            update_interval=float(sim_cfg.get('update_interval', 1e-3)),
            max_picard_iter=(int(sim_cfg['max_picard_iter'])
                             if sim_cfg.get('max_picard_iter') is not None else None),
            picard_tol=(float(sim_cfg['picard_tol'])
                        if sim_cfg.get('picard_tol') is not None else None),
            picard_omega=(float(sim_cfg['picard_omega'])
                          if sim_cfg.get('picard_omega') is not None else None),
            dtype=real_t,
        )

        geom_params = GeomParams(
            size=Vec3(float(Lx), float(Ly), float(Lz)),
            n=Vec3(int(nx), int(ny), int(nz)),
        )

        # Fine mesh — the refined, laser-following sub-box (optional). When the
        # ``fine_mesh`` section is absent, ``fine`` is None and the latent heat is
        # evaluated on the main (coarse) grid: lower memory and faster for small
        # runs. Providing the section switches on the localized fine-box path
        # (needed to resolve the mushy zone on large production grids).
        fine_cfg = cfg.get('fine_mesh', None)
        if fine_cfg is None:
            fine_params = None
        else:
            default_box = (0.9e-3, 0.9e-3, 0.04e-3)
            box = fine_cfg.get('box_size', default_box)
            fine_params = FineMeshParams(
                refinement=int(fine_cfg.get('refinement', 4)),
                box_size=Vec3(float(box[0]), float(box[1]), float(box[2])),
            )

        mat_cfg = cfg.get('material', {})
        T_solidus = real_t(_get_value(mat_cfg.get('T_solidus', 0.0)))
        T_liquidus = real_t(_get_value(mat_cfg.get('T_liquidus', 0.0)))
        T0 = real_t(_get_value(mat_cfg.get('T0', 0.0)))
        T_boil = real_t(_get_value(mat_cfg.get('T_boil', 0.0)))

        # Temperature-dependent properties: build a MaterialModel when any of
        # k/rho/Cp is given as polynomial branches.
        material_model = MaterialModel.from_config(
            mat_cfg, float(T_solidus), float(T_liquidus)
        )
        if material_model is None:
            # Fully scalar material — the scalars are the reference constants.
            rho_ref = real_t(_get_value(mat_cfg['rho']))
            k_ref = real_t(_get_value(mat_cfg['k']))
            cp_ref = real_t(_get_value(mat_cfg['Cp']))
        else:
            # The reference constants k̄, ρ̄, C̄p matter a lot for
            # convergence, which is why we force an explicit, deliberate choice.
            # A scalar or constant-branch property is its own reference.
            refs, missing = {}, []
            for cfg_key, tprop in (('k', material_model.k),
                                   ('rho', material_model.rho),
                                   ('Cp', material_model.c)):
                spec = mat_cfg.get(cfg_key)
                is_branch = (isinstance(spec, dict)
                             and ('solid' in spec or 'liquid' in spec))
                if is_branch and not tprop.is_constant:
                    if 'reference' not in spec:
                        missing.append(cfg_key)
                        continue
                    refs[cfg_key] = real_t(_get_value(spec['reference']))
                else:
                    refs[cfg_key] = real_t(float(tprop(float(T0))))
            if missing:
                k_rec, rho_rec, cp_rec = material_model.recommended_references(
                    float(T0), float(T_boil))
                rec = {'k': k_rec, 'rho': rho_rec, 'Cp': cp_rec}
                rec_str = ', '.join(f"{key}.reference={rec[key]:g}"
                                    for key in missing)
                raise ValueError(
                    f"Temperature-dependent material property {missing} needs an "
                    f"explicit 'reference:' key inside its block. It fixes the "
                    f"implicit/explicit split of the property correction; centering "
                    f"it in the middle of the working range matters for better "
                    f"convergence. Recommended — the average of each property's two "
                    f"extrema over [T0, T_boil] "
                    f"K — are: {rec_str}."
                )
            k_ref, rho_ref, cp_ref = refs['k'], refs['rho'], refs['Cp']

        mat_params = MaterialParams(
            name=mat_cfg.get('name', 'Material'),
            rho=rho_ref,
            k=k_ref,
            Cp=cp_ref,
            L_f=real_t(_get_value(mat_cfg.get('L_f', 0.0))),
            T_solidus=T_solidus,
            T_liquidus=T_liquidus,
            Pa=real_t(_get_value(mat_cfg.get('Pa', 0.0))),
            R_v=real_t(_get_value(mat_cfg.get('R_v', 0.0))),
            T_boil=T_boil,
            DeltaH_LV=real_t(_get_value(mat_cfg.get('DeltaH_LV', 0.0))),
            T0=T0,
            h_conv=real_t(_get_value(mat_cfg.get('h_conv', 0.0))),
            model=material_model,
        )

        laser_cfg = cfg.get('laser', {})
        laser_params = LaserParams(
            radius=real_t(_get_value(laser_cfg['radius'])),
            absorptivity=real_t(_get_value(laser_cfg['absorptivity'])),
            power=real_t(_get_value(laser_cfg['power_nominal'])),
            profile=str(laser_cfg.get('profile', 'gaussian')),
            r_x=real_t(_get_value(laser_cfg.get('r_x', 0.0))),
            r_y=real_t(_get_value(laser_cfg.get('r_y', 0.0))),
            super_gaussian_order=float(_get_value(laser_cfg.get('super_gaussian_order', 2.0))),
            cell_integrated=bool(laser_cfg.get('cell_integrated', False)),
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
                        gcode_file = os.path.abspath(os.path.join(os.getcwd(), 'simulations', 'examples', 'paths', gcode_file))
                
                # Write absolute path back to config so runner.py can access it
                cfg['laser']['path']['file'] = gcode_file
                laser_path = GCodeLaserPath(gcode_file, initial_position=initial_position)
                
        io_cfg = cfg.get('io', {})
        return cls(num=num_params, mat=mat_params, geom=geom_params, laser=laser_params, laser_path=laser_path, io=io_cfg, backend=sim_backend, fine=fine_params)
