import os
import logging
import h5py
from datetime import datetime
from typing import Any, Dict, Optional, Union
from core.io import IOManager

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
        """
        # 1. Extract config
        # Assuming context has an 'io' attribute or we fall back to defaults
        # We handle context dynamically since types might vary
        io_config = getattr(context, 'io', None)
        self.output_root = getattr(io_config, 'output_root', 'out')
        run_tag = getattr(io_config, 'run_tag', 'sim')
        self.save_full_fields = getattr(io_config, 'save_full_fields', False)
        
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

    def save_step(self, time: float, step: int, state: Any, **kwargs) -> None:
        """
        Save the current simulation state to HDF5/XDMF, with time and step in filenames and XMF metadata.
        """
        try:
            from utils.spectral_helpers import reconstruct_temperature_volume
            field = reconstruct_temperature_volume(state.a, state)
            grid_coords = (state.x, state.y, state.z)
            filename_base = self.get_output_path(f"field_step{step:06d}", subdir='fields')
            save_field_to_hdf5(filename_base, field, grid_coords, value_name="temperature", t=time, step=step)
            logger.info(f"Saved field for step {step} to {filename_base}.h5/.xmf")
        except Exception as e:
            logger.error(f"Failed to save field for step {step}: {e}")

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

    def finalize(self) -> None:
        """
        Nothing specific to close for local FS, just log.
        """
        logger.info(f"Simulation run {self.run_id} finalized. Data in {self.base_dir}")


# Utility function to save field



def save_field_to_hdf5(filename_base, field, grid_coords, value_name="Field", geom=None, verbose=False, t=None, step=None):
    """Serialize a 3D scalar field, on a uniform domain, to HDF5 with an accompanying XDMF wrapper. Adds time and step to XMF metadata and filenames."""
    h5_name = f"{filename_base}.h5"
    xmf_name = f"{filename_base}.xmf"
    h5_ref = os.path.basename(h5_name)

    x_coords, y_coords, z_coords = grid_coords
    nz, ny, nx = field.shape

    if verbose and geom is not None:
        print(f"Exporting HDF5/XDMF. Domain Size: {geom.Lx:.2e} x {geom.Ly:.2e} x {geom.Lz:.2e}")

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
