import os
import logging
import h5py
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional, Union
from fast_heat_solv.core.io import IOManager

logger = logging.getLogger(__name__)



class LocalFSIOManager(IOManager):
    """
    Concrete implementation of IOManager using the local filesystem.
    Organizes output in: output_root/run_id/{subdir}
    """

    def __init__(self):
        self.output_root: str = "out"
        self.run_id: Optional[str] = None
        self.base_dir: Optional[str] = None
        self.save_full_fields: bool = False
        self.format_version: str = "1.0"
        
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
        Also parses IO scheduling config (interval, outputs, at_end, etc.).
        """
        self.context = context  # Store context for later use
        # 1. Extract config
        io_config = getattr(context, 'io', None)
        self.output_root = getattr(io_config, 'output_root', 'out')
        run_tag = getattr(io_config, 'run_tag', 'sim')
        self.save_full_fields = getattr(io_config, 'save_full_fields', False)

        # --- IO scheduling state ----
        io_cfg = context.io if hasattr(context, 'io') else {}
        self._interval = io_cfg.get('interval')
        self._outputs = io_cfg.get('outputs', [])
        self._at_end = io_cfg.get('at_end', [])
        self._profiles_locations = io_cfg.get('profiles_locations', [])
        self._cut_views_planes = io_cfg.get('cut_views_planes', [])

        if self._interval is None:
            logger.info("io.interval is None: periodic outputs disabled; only 'at_end' outputs will be saved.")
            self._next_output_step: float = float('inf')
        else:
            try:
                self._next_output_step = int(self._interval)
            except Exception:
                logger.warning(f"Invalid io.interval '{self._interval}' - disabling periodic outputs.")
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

    def get_output_path(self, filename: str, subdir: Optional[str] = None) -> str:
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
        Save the current simulation state to HDF5/XDMF, with time and step in filenames and XMF metadata.
        Handles output types as defined in the YAML config (full_volume, profiles, cut_views, modes).
        """
        output_type = kwargs.get('output_type', 'full_volume')
        profiles_locations = kwargs.get('profiles_locations', [])
        cut_views_planes = kwargs.get('cut_views_planes', [])

        try:
            # Helper for reconstruction (lazy loaded)
            from fast_heat_solv.physics.spectral_helpers import reconstruct_temperature_DCT

            if output_type == 'full_volume':
                # Use grid from state (refactoring support)
                grid = state.grid if hasattr(state, 'grid') else state
                
                grid.prepare_full_reconstruction(self.context.geom)
                field = reconstruct_temperature_DCT(state.a, state).transpose(2,1,0)  # Ensure (z,y,x) ordering
                grid_coords = (getattr(grid, 'x_rec', None), getattr(grid, 'y_rec', None), getattr(grid, 'z_rec', None))
                
                filename_base = self.get_output_path(f"field_step{step:06d}", subdir='fields')
                if hasattr(field, "get"):
                    field = field.get()
                _save_field_to_hdf5(filename_base, field, grid_coords, value_name="temperature", t=time, step=step)
                logger.info(f"Saved field for step {step} to {filename_base}.h5/.xmf")

            elif output_type == 'modes':
                # Save spectral modes (coefficients) to a single HDF5 file (append mode)
                modes_file = self.get_output_path("modes.h5", subdir='fields')
                
                # Ensure data is on CPU
                modes_data = state.a
                if hasattr(modes_data, 'get'):
                    modes_data = modes_data.get()
                
                # Open file in append mode (or create)
                with h5py.File(modes_file, 'a') as f:
                    ds_name = "modes"
                    if ds_name not in f:
                        # Create resizable datasets (time, nz, ny, nx)
                        # We use maxshape=(None, ...) to allow resizing along the first dimension
                        shape = (0,) + modes_data.shape
                        maxshape = (None,) + modes_data.shape
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

            elif output_type == 'profiles':
                # Compute 1D profiles using the helper (no I/O in helper)
                from fast_heat_solv.physics.spectral_helpers import save_temp_profiles

                # Determine center and laser_position robustly.
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
                    fname = os.path.join(profiles_dir, f"{direction}_spectral_latent_heat.txt")
                    # Format: Coord [m] | Temp [K]
                    np.savetxt(fname, np.vstack([coords, temps]).T, header=f'{direction}(m) T(K)', fmt='% .6e')
                logger.info(f"Saved 1D profiles (center={center}, {laser_position}) for step {step} to {profiles_dir}")
            
            elif output_type == 'cut_views':
                # 1. Ensure XDMF exists for this step
                filename_base = self.get_output_path(f"field_step{step:06d}", subdir='fields')
                xdmf_path = f"{filename_base}.xmf"
                
                # Check if file exists; if not, we must reconstruct and save it
                if not os.path.exists(xdmf_path):
                    state.prepare_full_reconstruction(self.context.geom)
                    field = reconstruct_temperature_DCT(state.a, state).transpose(2,1,0)
                    grid_coords = (getattr(state, 'x_rec', None), getattr(state, 'y_rec', None), getattr(state, 'z_rec', None))
                    
                    if hasattr(field, "get"):
                        field = field.get()
                    _save_field_to_hdf5(filename_base, field, grid_coords, value_name="temperature", t=time, step=step)
                    logger.info(f"Generated XDMF for cut views at {xdmf_path}")

                # 2. Generate cut views for each plane
                from research.cut_views import generate_plots
                cut_views_dir = self.get_output_path('', subdir='cut_views')
                for plane in cut_views_planes:
                    logger.warning("You may want to set parameters for center, width, height, etc.")
                    output_file = os.path.join(cut_views_dir, f"cut_{plane}_step{step:06d}.png")
                    # Determine center based on laser position
                    laser_state = laser_path.get_state(time, 0.0)
                    height=0.0002
                    center = (float(laser_state.x), float(laser_state.y), self.context.geom.Lz-height/2)
                    generate_plots(
                        xdmf_path=xdmf_path,
                        output_dir=cut_views_dir,
                        show_ui=False,
                        save_images=True,
                        normal=plane[0],  # e.g., 'x', 'y', or 'z'
                        center=center,
                        width=0.0006,  # 0.6 mm
                        height=height,  # 0.2 mm
                        specific_output_filename=output_file
                    )
                    logger.info(f"Saved cut view {plane} for step {step} to {output_file}")
            else:
                logger.warning(f"Unknown output_type '{output_type}' in save_step. Skipping.")
        except Exception as e:
            logger.error(f"Failed to save {output_type} for step {step}: {e}")
            raise

    def load_step(self, step: Union[int, str] = 'latest') -> Optional[Dict[str, Any]]:
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
            
        full_path = os.path.join(fields_dir, target_file)
        if not os.path.exists(full_path):
            return None
            
        try:
            with h5py.File(full_path, 'r') as f:
                data = {
                    'temperature': f['temperature'][:],
                    'time': f['temperature'].attrs.get('time', 0.0),
                    'step': f['temperature'].attrs.get('step', -1)
                }
                return data
        except Exception as e:
            logger.error(f"Failed to load step {step}: {e}")
            return None

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
                cut_views_planes=self._cut_views_planes,
            )
        logger.info(f"Step {step} | t={t:.6e}s | Output(s) saved: {self._outputs}")

        if self._interval is not None:
            try:
                self._next_output_step += int(self._interval)
            except Exception:
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
                cut_views_planes=self._cut_views_planes,
            )
        if self._at_end:
            logger.info(f"Final output(s) saved at end: {self._at_end}")

    def finalize(self) -> None:
        """
        Nothing specific to close for local FS, just log.
        """
        logger.info(f"Simulation run {self.run_id} finalized. Data in {self.base_dir}")


def _save_field_to_hdf5(filename_base, field, grid_coords, value_name="Field", verbose=False, t=None, step=None):
    """Serialize a 3D scalar field, on a uniform domain, to HDF5 with an accompanying XDMF wrapper. Adds time and step to XMF metadata and filenames."""
    h5_name = f"{filename_base}.h5"
    xmf_name = f"{filename_base}.xmf"
    h5_ref = os.path.basename(h5_name)
    

    x_coords, y_coords, z_coords = grid_coords
    # Ensure all coordinate arrays are NumPy arrays (not CuPy)
    # TO DO handle that better upstream
    if hasattr(x_coords, "get"):
        x_coords = x_coords.get()
    if hasattr(y_coords, "get"):
        y_coords = y_coords.get()
    if hasattr(z_coords, "get"):
        z_coords = z_coords.get()

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

    # Add time and step as XML attributes in the XMF file
    time_str = f' Time="{t}"' if t is not None else ''
    step_str = f' Step="{step}"' if step is not None else ''
    xmf_content = f'''<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
     <Grid Name="Mesh" GridType="Uniform"{time_str}{step_str}>
         <Topology TopologyType="3DRectMesh" Dimensions="{nz} {ny} {nx}"/>
         <Geometry GeometryType="VXVYVZ">
             <DataItem Dimensions="{nx}" NumberType="Float" Precision="4" Format="HDF">
                {h5_ref}:/X
             </DataItem>
             <DataItem Dimensions="{ny}" NumberType="Float" Precision="4" Format="HDF">
                {h5_ref}:/Y
             </DataItem>
             <DataItem Dimensions="{nz}" NumberType="Float" Precision="4" Format="HDF">
                {h5_ref}:/Z
             </DataItem>
         </Geometry>
         <Attribute Name="{value_name}" AttributeType="Scalar" Center="Node">
             <DataItem Dimensions="{nz} {ny} {nx}" NumberType="Float" Precision="4" Format="HDF">
                {h5_ref}:/{value_name}
             </DataItem>
         </Attribute>
     </Grid>
 </Domain>
</Xdmf>
'''
    with open(xmf_name, "w") as f:
        f.write(xmf_content)

    if verbose:
        print(f"Saved debug files: {xmf_name} (Open this in Paraview)")
