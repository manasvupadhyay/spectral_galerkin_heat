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

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from fast_heat_solv.core.properties import MaterialModel
from fast_heat_solv.core.vector import Vec3

if TYPE_CHECKING:
    from fast_heat_solv.core.laser import LaserPath

# ``cfg`` throughout is the parsed-YAML configuration dict consumed
# by ``SimulationContext.from_dict``.
# cfg stands for config 
# Scalar values may be a plain number or a {value, unit} mapping (see _get_value).


import pint

# Create a single unit registry instance to be reused.
_ureg = pint.UnitRegistry()

# Configs write exponents in the compact engineering form (``m2``, ``m3``),
# which pint reads as undefined unit names rather than powers. ``.`` (multiply)
# and ``^`` are already understood, so only the digit suffix needs rewriting.
# No SI unit used here carries a digit in its name.
_UNIT_EXPONENT_RE = re.compile(r"(?<=[A-Za-z])(\d+)")


def _normalize_unit(unit: str) -> str:
    """Rewrite compact exponents (``W/(m2.K)``) into pint syntax (``W/(m**2.K)``)."""
    return _UNIT_EXPONENT_RE.sub(r"**\1", unit)


def _get_value(raw_val: Any, target_unit: str | None = None) -> float:
    """
    Extracts a numeric value from a config field, with unit conversion.

    If raw_val is a dict with a "unit" and a target_unit is provided, it
    converts the value to the target unit. Otherwise, it returns the value
    as-is, assuming it's already in the correct units.

    Args:
        raw_val: The raw value from the configuration dictionary. Can be a
            scalar or a dict like {"value": 1.0, "unit": "m"}.
        target_unit: The desired pint-compatible unit string (e.g., "kg / m**3").

    Returns:
        The numeric value, converted to the target unit if applicable.

    Raises:
        pint.errors.DimensionalityError: If the provided unit is incompatible
            with the target unit.
        KeyError: If the dict is malformed (e.g., missing "value").
    """
    if isinstance(raw_val, dict):
        # This handles both simple {value, unit} dicts and the more complex
        # T-dependent property dicts which also contain a "value" for reference.
        value = raw_val["value"]
        unit = raw_val.get("unit")

        if unit and target_unit:
            quantity = _ureg.Quantity(value, _normalize_unit(unit))
            return float(quantity.to(_normalize_unit(target_unit)).magnitude)

        return float(value)

    if isinstance(raw_val, (int, float)):
        return float(raw_val)

    # Raise for other types that are not directly convertible to float.
    raise TypeError(f"Cannot extract a value from type {type(raw_val)}")


def _get_vector(raw_val: Any, target_unit: str | None = None) -> list:
    """Extract a list of numbers from a config field, with unit conversion.

    Vector fields (``domain.size``, ``domain.mesh``) carry the sequence under
    the ``value`` key of a ``{value, unit}`` mapping, so the mapping must be
    unwrapped before the elements are converted; a bare sequence is also
    accepted. Returns floats -- integer fields are cast by the caller.
    """
    if isinstance(raw_val, dict):
        values = raw_val["value"]
        unit = raw_val.get("unit")
        if unit and target_unit:
            src, dst = _normalize_unit(unit), _normalize_unit(target_unit)
            return [float(_ureg.Quantity(v, src).to(dst).magnitude)
                    for v in values]
        return [float(v) for v in values]

    if isinstance(raw_val, (list, tuple)):
        return [float(_get_value(v, target_unit)) for v in raw_val]

    raise TypeError(f"Cannot extract a vector from type {type(raw_val)}")


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
    Numerical parameters for the simulation: time stepping and grid resolution.

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
    max_picard_iter: int | None = None
    picard_tol: float | None = None
    picard_omega: float | None = None
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
    h_conv_top : float, optional
        Convective heat transfer coefficient in W/(m²·K) at the top surface
        (z = Lz, the laser-facing face -- same face as the laser flux and
        evaporation). 0.0 disables it. Typical: 50-5000 W/(m²·K), by default 0.0.
    h_conv_bottom : float, optional
        Convective heat transfer coefficient in W/(m²·K) at the bottom surface
        (z = 0). 0.0 disables it. Typical: 50-5000 W/(m²·K), by default 0.0.
    model : MaterialModel, optional
        Temperature-dependent property model (``k(T)``, ``rho(T)``, ``c(T)``
        solid/liquid branch expressions). ``None`` for a constant-property material, in which
        case the solver uses the scalar ``rho``/``k``/``Cp`` directly. When
        present, the scalar ``rho``/``k``/``Cp`` fields hold the user-supplied
        **reference constants** (each property's in-block ``reference:`` key) used
        by the ETD1 propagators; the property-correction path reinstates the
        fluctuations about them.
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
    h_conv_top: float = 0.0
    h_conv_bottom: float = 0.0
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
    """Refined, laser-following sub-box of the domain, for the latent-heat and
    nonlinear terms.

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
    # Vec3 is frozen and slotted, so one shared default instance is safe.
    box_size: Vec3 = Vec3(0.9e-3, 0.9e-3, 0.04e-3)  # noqa: RUF009

# ---------------------------------------------------------------------------
# Config schema. Every key the parser reads, by section. Anything else in a
# config is a mistake.
# ---------------------------------------------------------------------------

_SECTION_KEYS = {
    "simulation": {
        "name", "backend", "dtype", "dt", "duration", "update_interval",
        "max_picard_iter", "picard_tol", "picard_omega",
    },
    "domain": {"size", "mesh"},
    "material": {
        "name", "rho", "k", "Cp", "L_f", "T_solidus", "T_liquidus", "T0",
        "Pa", "R_v", "T_boil", "DeltaH_LV", "h_conv_top", "h_conv_bottom",
    },
    "laser": {
        "radius", "absorptivity", "power_nominal", "profile", "r_x", "r_y",
        "super_gaussian_order", "cell_integrated", "path",
    },
    "fine_mesh": {"refinement", "box_size"},
}
# Sections this parser does not read.
_PASSTHROUGH_SECTIONS = {"io", "post_processing"}
_LASER_PATH_KEYS = {"type", "file", "initial_position"}


def _reject_unknown(section: str, keys, allowed) -> None:
    """Raise on any key not in *allowed*, listing the keys the section accepts."""
    for key in keys:
        if key in allowed:
            continue
        raise ValueError(
            f"Unknown key {key!r} in the {section!r} config section. "
            f"Valid keys: {', '.join(sorted(allowed))}."
        )


def _validate_config(cfg: dict) -> None:
    """Check section names and their keys before any of them is read.

    Runs first so an unrecognised key is reported as such, instead of
    surfacing later as a default that was silently left in place.
    """
    known_sections = set(_SECTION_KEYS) | _PASSTHROUGH_SECTIONS
    for section in cfg:
        if section in known_sections:
            continue
        raise ValueError(
            f"Unknown config section {section!r}. "
            f"Valid sections: {', '.join(sorted(known_sections))}."
        )

    for section, allowed in _SECTION_KEYS.items():
        body = cfg.get(section)
        if not isinstance(body, dict):
            continue
        _reject_unknown(section, body, allowed)

    path_cfg = cfg.get("laser", {}).get("path") if isinstance(cfg.get("laser"), dict) else None
    if isinstance(path_cfg, dict):
        _reject_unknown("laser.path", path_cfg, _LASER_PATH_KEYS)


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
        Moving fine-mesh parameters: the refined, laser-following sub-box
        (refinement, box extents). Defaults to
        :class:`FineMeshParams` defaults.
    """
    num: 'NumParams'
    mat: 'MaterialParams'
    geom: 'GeomParams'
    laser: 'LaserParams'
    laser_path: 'LaserPath' # Use forward reference

    # Existing fields
    io: dict[str, Any]  # Flat dict with new keys
    # Execution configuration
    backend: str = "cpu"      # "cpu", "gpu", or "cpu_linear"
    fine: 'FineMeshParams' = field(default_factory=FineMeshParams)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], config_dir: str | None = None) -> 'SimulationContext':
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
        _validate_config(cfg)

        real_t = _resolve_dtype(cfg.get('simulation', {}).get('dtype', 'float32'))
        sim_cfg = cfg.get('simulation', {})
        domain_cfg = cfg.get('domain', {})
        sim_backend = sim_cfg.get('backend', 'cpu').lower()
        
        Lx, Ly, Lz = _get_vector(domain_cfg['size'], "m")

        nx, ny, nz = _get_vector(domain_cfg['mesh'])

        t_end = float(_get_value(sim_cfg.get('duration', 0.01), "s"))
        dt_nominal = float(real_t(_get_value(sim_cfg['dt'], "s")))
        
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
            update_interval=float(_get_value(sim_cfg.get('update_interval', 1e-3), "s")),
            max_picard_iter=(int(sim_cfg['max_picard_iter'])
                             if sim_cfg.get('max_picard_iter') is not None else None),
            picard_tol=(float(sim_cfg['picard_tol'])
                        if sim_cfg.get('picard_tol') is not None else None),
            picard_omega=(float(sim_cfg['picard_omega'])
                          if sim_cfg.get('picard_omega') is not None else None),
            dtype=real_t,
        )

        geom_params = GeomParams(
            size=Vec3(Lx, Ly, Lz),
            n=Vec3(int(nx), int(ny), int(nz)),
        )

        fine_cfg = cfg.get('fine_mesh', None)
        if fine_cfg is None:
            fine_params = None
        else:
            default_box = (0.9e-3, 0.9e-3, 0.04e-3)
            box_val = fine_cfg.get('box_size', default_box)
            box_size_m = [float(_get_value(v, "m")) for v in box_val]
            
            fine_params = FineMeshParams(
                refinement=int(_get_value(fine_cfg.get('refinement', 4))),
                box_size=Vec3(box_size_m[0], box_size_m[1], box_size_m[2]),
            )

        mat_cfg = cfg.get('material', {})
        T_solidus = real_t(_get_value(mat_cfg.get('T_solidus', 0.0), "K"))
        T_liquidus = real_t(_get_value(mat_cfg.get('T_liquidus', 0.0), "K"))
        T0 = real_t(_get_value(mat_cfg.get('T0', 0.0), "K"))
        T_boil = real_t(_get_value(mat_cfg.get('T_boil', 0.0), "K"))

        material_model = MaterialModel.from_config(
            mat_cfg, float(T_solidus), float(T_liquidus)
        )
        if material_model is None:
            rho_ref = real_t(_get_value(mat_cfg['rho'], "kg / m**3"))
            k_ref = real_t(_get_value(mat_cfg['k'], "W / (m * K)"))
            cp_ref = real_t(_get_value(mat_cfg['Cp'], "J / (kg * K)"))
        else:
            refs, missing = {}, []
            for cfg_key, tprop, unit in (('k', material_model.k, "W / (m * K)"),
                                         ('rho', material_model.rho, "kg / m**3"),
                                         ('Cp', material_model.c, "J / (kg * K)")):
                spec = mat_cfg.get(cfg_key)
                is_branch = (isinstance(spec, dict)
                             and ('solid' in spec or 'liquid' in spec))
                if is_branch and not tprop.is_constant:
                    if 'reference' not in spec:
                        missing.append(cfg_key)
                        continue
                    refs[cfg_key] = real_t(_get_value(spec['reference'], unit))
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
                    f"convergence. The recommended values, each the average of "
                    f"that property's two extrema over [T0, T_boil] K, "
                    f"are: {rec_str}."
                )
            k_ref, rho_ref, cp_ref = refs['k'], refs['rho'], refs['Cp']

        mat_params = MaterialParams(
            name=mat_cfg.get('name', 'Material'),
            rho=rho_ref,
            k=k_ref,
            Cp=cp_ref,
            L_f=real_t(_get_value(mat_cfg.get('L_f', 0.0), "J / kg")),
            T_solidus=T_solidus,
            T_liquidus=T_liquidus,
            Pa=real_t(_get_value(mat_cfg.get('Pa', 0.0), "Pa")),
            R_v=real_t(_get_value(mat_cfg.get('R_v', 0.0), "J / (kg * K)")),
            T_boil=T_boil,
            DeltaH_LV=real_t(_get_value(mat_cfg.get('DeltaH_LV', 0.0), "J / kg")),
            T0=T0,
            h_conv_top=real_t(_get_value(mat_cfg.get('h_conv_top', 0.0), "W / (m**2 * K)")),
            h_conv_bottom=real_t(_get_value(mat_cfg.get('h_conv_bottom', 0.0), "W / (m**2 * K)")),
            model=material_model,
        )

        laser_cfg = cfg.get('laser', {})
        laser_params = LaserParams(
            radius=real_t(_get_value(laser_cfg['radius'], "m")),
            absorptivity=real_t(_get_value(laser_cfg['absorptivity'])),
            power=real_t(_get_value(laser_cfg['power_nominal'], "W")),
            profile=str(laser_cfg.get('profile', 'gaussian')),
            r_x=real_t(_get_value(laser_cfg.get('r_x', 0.0), "m")),
            r_y=real_t(_get_value(laser_cfg.get('r_y', 0.0), "m")),
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
