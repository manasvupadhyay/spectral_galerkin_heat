import logging
import os
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from spectral_galerkin_heat.backends.base import to_host

logger = logging.getLogger(__name__)

# Fallback in-plane extents [m] when a slice_planes entry omits them.
# width -> in-plane horizontal (h) axis, height -> in-plane vertical (v) axis.
DEFAULT_SLICE_WIDTH = 0.6e-3
DEFAULT_SLICE_HEIGHT = 0.2e-3

# A geometric cut plane for a 2-D slice image. ``normal`` is a signed axis

SlicePlane = namedtuple("SlicePlane", "name normal width height offset_h offset_v isotherms")


def _parse_slice_plane(entry: Any) -> SlicePlane:
    """Parse one ``slice_planes`` YAML entry (a mapping) into a :class:`SlicePlane`.

    Expected keys:
        normal    signed axis 'x','-x','y','-y','z','-z' (required)
        name      free-form label used for the image filename (optional)
        width     in-plane horizontal (h) extent [m] (optional, default)
        height    in-plane vertical (v) extent [m] (optional, default)
        offset_h  shift of the cut centre along h [m] (optional, default 0)
        offset_v  shift of the cut centre along v [m] (optional, default 0)
        isotherms list of [name, temperature_K] contours (optional; none drawn
                  if omitted)
    """
    if not isinstance(entry, dict):
        raise ValueError(
            f"slice_planes entry must be a mapping with a 'normal' key, got {entry!r}."
        )

    raw_normal = entry.get('normal')
    if raw_normal is None or str(raw_normal).strip() == "":
        raise ValueError("slice_planes entry is missing the required 'normal' key.")
    normal = str(raw_normal).strip().lower()
    axis = normal.lstrip('+-')
    if axis not in ('x', 'y', 'z'):
        raise ValueError(
            f"Invalid normal '{raw_normal}': expected one of x,-x,y,-y,z,-z."
        )

    name = entry.get('name')
    if not name:
        # Filename-safe default, e.g. '-y' -> 'neg_y'.
        name = f"neg_{axis}" if normal.startswith('-') else axis

    def _num(key: str, default: float) -> float:
        val = entry.get(key)
        return default if val is None else float(val)

    isotherms = [(str(iso_name), float(iso_val)) for iso_name, iso_val in (entry.get('isotherms') or [])]

    return SlicePlane(
        name=str(name),
        normal=normal,
        width=_num('width', DEFAULT_SLICE_WIDTH),
        height=_num('height', DEFAULT_SLICE_HEIGHT),
        offset_h=_num('offset_h', 0.0),
        offset_v=_num('offset_v', 0.0),
        isotherms=isotherms,
    )



class LocalFSIOManager:
    """Simulation I/O against the local filesystem.

    Organizes output in ``output_root/run_id/{subdir}``. Lifecycle:
    :meth:`initialize` once before stepping, :meth:`process_step` after each
    step (writes at the configured interval), :meth:`process_end` for ``at_end``
    outputs, and :meth:`finalize` to flush/close.
    """

    def __init__(self):
        self.output_root: str = "out"
        self.run_id: str | None = None
        self.base_dir: str | None = None
        self.format_version: str = "1.0"
        self._saved_xmf_steps = []
        
        # Subdirectories map
        self.dirs = {
            'fields': 'fields',
            'profiles': 'profiles',
            'logs': 'logs',
            'diagnostics': 'diagnostics'
        }

    def initialize(self, context: Any) -> None:
        """
        Setup directory structure: root/run_id/{subdirs}
        Also parses IO scheduling config (output_interval, outputs, at_end, etc.).
        """
        self.context = context
        # context.io is always a dict (SimulationContext sets it); the `or {}`
        # only guards a directly-constructed context that passes io=None.
        io_cfg = context.io or {}
        self.output_root = io_cfg.get('output_root', 'out')
        run_tag = io_cfg.get('run_tag', 'sim')

        # --- IO scheduling state ----
        self._interval = io_cfg.get('output_interval')
        self._outputs = io_cfg.get('outputs') or []
        self._at_end = io_cfg.get('at_end') or []
        self._profiles_locations = io_cfg.get('profiles_locations') or []
        # Raw slice_planes entries; parsed per-plane in _save_slices. Each entry
        # is a plane name or [name, width, height, shift_h, shift_v] (see
        # _parse_slice_plane).
        self._slice_planes = io_cfg.get('slice_planes') or []

        if self._interval is None:
            logger.info("io.output_interval is None: periodic outputs disabled; only 'at_end' outputs will be saved.")
            self._next_output_step: float = float('inf')
        else:
            try:
                self._next_output_step = int(self._interval)
            except Exception:
                logger.warning(f"Invalid io.output_interval '{self._interval}' - disabling periodic outputs.")
                self._next_output_step = float('inf')
        
        # 2. Generate Run ID
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{timestamp}_{run_tag}"
        self.base_dir = os.path.join(self.output_root, self.run_id)
        
        # 3. Create Directories
        try:
            os.makedirs(self.base_dir, exist_ok=False)
            logger.info(f"Initialized output directory: {self.base_dir}")
            
            for key, subdir in self.dirs.items():
                path = os.path.join(self.base_dir, subdir)
                os.makedirs(path, exist_ok=True)
                
        except OSError as e:
            logger.error(f"Failed to create output directories at {self.base_dir}: {e}")
            raise

    def get_output_path(self, filename: str, subdir: str | None = None) -> str:
        """Construct full path."""
        if self.base_dir is None:
            raise RuntimeError("IOManager not initialized. Call initialize() first.")
            
        if subdir and subdir in self.dirs:
            return os.path.join(self.base_dir, self.dirs[subdir], filename)
        elif subdir:
            # Custom subdir case
            return os.path.join(self.base_dir, subdir, filename)
        else:
            return os.path.join(self.base_dir, filename)

    def save_step(self, time: float, step: int, state: Any, laser_path: Any, **kwargs) -> None:
        """
        Save the current simulation state for the requested ``output_type``.

        Dispatches to a per-output helper (``full_volume``, ``modes``,
        ``profiles`` or ``slices``, as defined in the YAML config). Time and
        step are recorded in filenames / XMF metadata.
        """
        output_type = kwargs.get('output_type', 'full_volume')
        try:
            match output_type:
                case 'full_volume':
                    self._save_full_volume(time, step, state)
                case 'modes':
                    self._save_modes(time, step, state)
                case 'profiles':
                    self._save_profiles(time, step, state, laser_path,
                                        kwargs.get('profiles_locations', []))
                case 'slices':
                    self._save_slices(time, step, state, laser_path,
                                         kwargs.get('slice_planes', []))
                case _:
                    logger.warning(f"Unknown output_type '{output_type}' in save_step. Skipping.")
        except Exception as e:
            logger.error(f"Failed to save {output_type} for step {step}: {e}")
            raise

    def _save_full_volume(self, time: float, step: int, state: Any) -> None:
        """Reconstruct the full temperature volume and write HDF5 + XDMF."""
        from spectral_galerkin_heat.physics.spectral_helpers import reconstruct_temperature_DCT

        grid = state.grid

        grid.prepare_full_reconstruction(self.context.geom)
        field = reconstruct_temperature_DCT(state.a, state).transpose(2, 1, 0)  # Ensure (z,y,x) ordering
        grid_coords = tuple(grid.coords_rec) if grid.coords_rec is not None else (None, None, None)

        filename_base = self.get_output_path(f"field_step{step:06d}", subdir='fields')
        field = to_host(field)
        step_info = _save_field_to_hdf5(filename_base, field, grid_coords, value_name="temperature", t=time, step=step)
        self._saved_xmf_steps.append(step_info)
        self._write_timeseries_xmf()
        logger.info(f"Saved field for step {step} to {filename_base}.h5/.xmf")

    def _save_modes(self, time: float, step: int, state: Any) -> None:
        """Append this step's spectral modes to a single resizable HDF5 file."""
        modes_file = self.get_output_path("modes.h5", subdir='fields')

        modes_data = to_host(state.a)

        # Open file in append mode (or create)
        with h5py.File(modes_file, 'a') as f:
            ds_name = "modes"
            if ds_name not in f:
                # Create resizable datasets (time, nz, ny, nx)
                # We use maxshape=(None, ...) to allow resizing along the first dimension
                shape = (0, *modes_data.shape)
                maxshape = (None, *modes_data.shape)
                # Enable compression for efficiency
                f.create_dataset(ds_name, shape=shape, maxshape=maxshape, dtype=modes_data.dtype, chunks=True, compression="gzip")
                f.create_dataset("time", shape=(0,), maxshape=(None,), dtype='f8', chunks=True)
                f.create_dataset("step", shape=(0,), maxshape=(None,), dtype='i8', chunks=True)

            dset = f[ds_name]
            d_time = f["time"]
            d_step = f["step"]

            # Resize
            new_size = dset.shape[0] + 1
            dset.resize(new_size, axis=0)
            d_time.resize(new_size, axis=0)
            d_step.resize(new_size, axis=0)

            # Write
            dset[-1] = modes_data
            d_time[-1] = time
            d_step[-1] = step

        logger.info(f"Appended modes for step {step} to {modes_file}")

    def _save_profiles(self, time: float, step: int, state: Any, laser_path: Any,
                       profiles_locations: Any) -> None:
        """Compute 1-D temperature profiles and write them to the profiles/ subfolder."""
        # Compute 1D profiles using the helper (no I/O in helper)
        from spectral_galerkin_heat.physics.spectral_helpers import save_temp_profiles

        # Determine center and laser position.
        laser_position = None
        center = 'hotspot'
        # Case: profiles_locations provided as a plain string 'laser'
        if isinstance(profiles_locations, str) and profiles_locations.lower() == 'laser':
            if laser_path is not None:
                laser_state = laser_path.get_state(time, 0.0)
                laser_position = (float(laser_state.x), float(laser_state.y))
                center = 'laser'
            else:
                logger.warning("profiles_locations='laser' requested but no laser_path available; using 'hotspot'.")

        # Case: profiles_locations is a list/tuple
        elif isinstance(profiles_locations, (list, tuple)) and len(profiles_locations) > 0:
            first = profiles_locations[0]
            if isinstance(first, str) and first.lower() == 'laser':
                if laser_path is not None:
                    laser_state = laser_path.get_state(time, 0.0)
                    laser_position = (float(laser_state.x), float(laser_state.y))
                    center = 'laser'
                    # replace sentinel with actual position for logging
                    profiles_locations = [laser_position]
                else:
                    logger.warning("profiles_locations contains 'laser' but no laser_path available; using 'hotspot'.")
            elif isinstance(first, (list, tuple)) and len(first) >= 2:
                # Explicit numeric coordinates provided
                laser_position = (float(first[0]), float(first[1]))
                center = laser_position

        # Call helper with a well-formed laser_position (None or tuple(x,y)) and center
        profiles = save_temp_profiles(state.a, self.context.num, self.context.geom,
            state, laser_position, center=center
        )

        # Write each profile to the profiles/ subfolder
        profiles_dir = self.get_output_path('', subdir='profiles')
        for direction, (coords, temps) in profiles.items():
            fname = os.path.join(profiles_dir, f"{direction}_step{step:06d}.txt")
            # Format: Coord [m] | Temp [K]
            np.savetxt(fname, np.vstack([coords, temps]).T, header=f'{direction}(m) T(K)', fmt='% .6e')
        logger.info(f"Saved 1D profiles (center={center}, {laser_position}) for step {step} to {profiles_dir}")

    def _save_slices(self, time: float, step: int, state: Any, laser_path: Any,
                        slice_planes: Any) -> None:
        """Ensure the step's XDMF exists, then render a cut-plane image per plane."""
        from spectral_galerkin_heat.physics.spectral_helpers import reconstruct_temperature_DCT

        # 1. Ensure XDMF exists for this step. load_xdmf expects a Path, so build
        #    one here rather than passing a bare string down the slice pipeline.
        filename_base = self.get_output_path(f"field_step{step:06d}", subdir='fields')
        xdmf_path = Path(f"{filename_base}.xmf")

        # Check if file exists; if not, we must reconstruct and save it
        if not os.path.exists(xdmf_path):
            state.grid.prepare_full_reconstruction(self.context.geom)
            field = reconstruct_temperature_DCT(state.a, state).transpose(2, 1, 0)
            grid_coords = tuple(state.grid.coords_rec) if state.grid.coords_rec is not None else (None, None, None)

            field = to_host(field)
            step_info = _save_field_to_hdf5(filename_base, field, grid_coords, value_name="temperature", t=time, step=step)
            self._saved_xmf_steps.append(step_info)
            self._write_timeseries_xmf()
            logger.info(f"Generated XDMF for slices at {xdmf_path}")

        # 2. Generate slices for each plane
        from spectral_galerkin_heat.io_utils.slices import generate_plots
        slices_dir = self.get_output_path('', subdir='slices')
        surf_z = self.context.geom.size.z
        laser_state = laser_path.get_state(time, 0.0)
        lx, ly = float(laser_state.x), float(laser_state.y)
        for entry in slice_planes:
            plane = _parse_slice_plane(entry)

            # Resolve the absolute cut centre here (this is the only place that
            # knows the laser position); slices.py stays oblivious to the laser.
            # The centre is the laser spot plus the in-plane offsets. The sign of
            # the normal is a viewing choice only, so strip it to pick the
            # horizontal / vertical in-plane axes:
            #   axis 'z' (xy): h=x, v=y, cut taken at the surface (z = size.z)
            #   axis 'y' (xz): h=x, v=z, top of the cut aligned with surface
            #   axis 'x' (yz): h=y, v=z, top of the cut aligned with surface
            axis = plane.normal.lstrip('+-')
            if axis == 'z':
                center = (lx + plane.offset_h, ly + plane.offset_v, surf_z)
            elif axis == 'y':
                center = (lx + plane.offset_h, ly,
                          surf_z - plane.height / 2 + plane.offset_v)
            else:  # axis == 'x'
                center = (lx, ly + plane.offset_h,
                          surf_z - plane.height / 2 + plane.offset_v)

            output_file = os.path.join(slices_dir, f"slice_{plane.name}_step{step:06d}.png")
            generate_plots(
                xdmf_path=xdmf_path,
                output_dir=slices_dir,
                show_ui=False,
                save_images=True,
                normal=plane.normal,  # signed; slices.py handles the mirror
                center=center,
                width=plane.width,
                height=plane.height,
                isotherms=plane.isotherms,
                specific_output_filename=output_file
            )
            logger.info(f"Saved slice {plane.name} for step {step} to {output_file}")

    def load_step(self, step: int | str = 'latest') -> dict[str, Any] | None:
        """
        Load back a full field h5 file.
        """
        if self.base_dir is None:
            # Try to resolve run_id or raise error? 
            # For this simple impl, assume initialize or manual setup was done.
            logger.warning("IOManager not initialized with a base_dir for loading.")
            return None

        fields_dir = self.get_output_path("", "fields")
        if not os.path.exists(fields_dir):
            return None

        target_file = None
        
        # Find file
        if step == 'latest':
            files = sorted([f for f in os.listdir(fields_dir) if f.startswith("field_step") and f.endswith(".h5")])
            if files:
                target_file = files[-1]
        elif isinstance(step, int):
            target_file = f"field_step{step:06d}.h5"
        
        if not target_file:
            return None
            
        # Shared reader (see io_utils.loader) keeps load_step / get_field in sync.
        from spectral_galerkin_heat.io_utils.loader import read_field_h5
        return read_field_h5(os.path.join(fields_dir, target_file))

    # ------------------------------------------------------------------
    #  Step-level & end-of-simulation IO orchestration
    # ------------------------------------------------------------------
    def process_step(self, t: float, step: int, state: Any, laser_path: Any) -> None:
        """
        Called every time step.  Internally checks whether it is time to
        write periodic outputs based on the configured interval.
        """
        if step < self._next_output_step:
            return

        for output_type in self._outputs:
            self.save_step(
                t, step, state, laser_path,
                output_type=output_type,
                profiles_locations=self._profiles_locations,
                slice_planes=self._slice_planes,
            )
        logger.info(f"Step {step} | t={t:.6e}s | Output(s) saved: {self._outputs}")

        if self._interval is not None:
            try:
                self._next_output_step += int(self._interval)
            except Exception as exc:
                # Falling through to inf disables every later output for the
                # rest of the run, so say so rather than going quiet.
                logger.warning(
                    "Invalid output interval %r (%s); no further steps will be "
                    "written this run.", self._interval, exc)
                self._next_output_step = float('inf')
        else:
            self._next_output_step = float('inf')

    def process_end(self, t: float, step: int, state: Any, laser_path: Any) -> None:
        """
        Called once after the time-loop finishes to save 'at_end' outputs.
        """
        for output_type in self._at_end:
            self.save_step(
                t, step, state, laser_path,
                output_type=output_type,
                profiles_locations=self._profiles_locations,
                slice_planes=self._slice_planes,
            )
        if self._at_end:
            logger.info(f"Final output(s) saved at end: {self._at_end}")

    def _write_timeseries_xmf(self) -> None:
        if not self._saved_xmf_steps:
            return
            
        out_path = self.get_output_path("temperature_series.xmf", subdir="fields")
        import xml.etree.ElementTree as ET

        from .xdmf_io import XdmfBuilder
        
        builder = XdmfBuilder(version="2.0")
        collection = ET.SubElement(builder.domain, "Grid", Name="TimeSeries", GridType="Collection", CollectionType="Temporal")
        
        for info in self._saved_xmf_steps:
            builder.add_structured_grid(
                name="Mesh",
                dims=info['dims'],
                h5_ref=info['h5_ref'],
                attributes={info['value_name']: info['value_name']},
                time=info['time'],
                step=info['step'],
                parent=collection
            )
            
        builder.write(out_path)

    def finalize(self) -> None:
        """
        Nothing specific to close for local FS, just log.
        """
        logger.info(f"Simulation run {self.run_id} finalized. Data in {self.base_dir}")


def _save_field_to_hdf5(filename_base, field, grid_coords, value_name="Field", verbose=False, t=None, step=None):
    """Serialize a 3D scalar field, on a uniform domain, to HDF5 with an accompanying XDMF wrapper. Adds time and step to XMF metadata and filenames."""
    from .xdmf_io import XdmfBuilder

    h5_name = f"{filename_base}.h5"
    xmf_name = f"{filename_base}.xmf"
    h5_ref = os.path.basename(h5_name)


    x_coords, y_coords, z_coords = grid_coords
    x_coords = to_host(x_coords)
    y_coords = to_host(y_coords)
    z_coords = to_host(z_coords)

    nz, ny, nx = field.shape
    if (nx != len(x_coords)) or (ny != len(y_coords)) or (nz != len(z_coords)):
        raise ValueError(
            f"Field shape (z, y, x) = {field.shape} does not match grid coordinates lengths: "
            f"X({len(x_coords)}), Y({len(y_coords)}), Z({len(z_coords)}). "
            f"Expected field.shape = (len(z), len(y), len(x)) = ({len(z_coords)}, {len(y_coords)}, {len(x_coords)})"
        )

    with h5py.File(h5_name, "w") as f:
        f.create_dataset("X", data=x_coords)
        f.create_dataset("Y", data=y_coords)
        f.create_dataset("Z", data=z_coords)
        dset = f.create_dataset(value_name, data=field)
        # Store time and step as attributes in the HDF5 file
        if t is not None:
            dset.attrs['time'] = t
        if step is not None:
            dset.attrs['step'] = step

    # Build XDMF using ElementTree via XdmfBuilder
    builder = XdmfBuilder(version="2.0")
    builder.add_structured_grid(
        name="Mesh",
        dims=(nz, ny, nx),
        h5_ref=h5_ref,
        attributes={value_name: value_name},
        time=t,
        step=step,
    )
    builder.write(xmf_name)

    if verbose:
        print(f"Saved debug files: {xmf_name} (Open this in Paraview)")
        
    return {
        'time': t,
        'step': step,
        'dims': (nz, ny, nx),
        'h5_ref': h5_ref,
        'value_name': value_name
    }
