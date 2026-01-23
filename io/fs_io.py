import os
import shutil
import logging
import time
import numpy as np
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

    def save_step(self, time_val: float, step: int, field: Any, **kwargs) -> None:
        """
        Save outputs for current step.
        - Always saves 1D profiles (if 'profiles' data is passed in kwargs)
        - Conditionally saves full 3D fields
        """
        if self.base_dir is None:
            return

        # 1. Save Full Field (if configured)
        if self.save_full_fields and field is not None:
            filename = f"field_step{step:06d}.h5"
            path = self.get_output_path(filename, 'fields')
            temp_path = path + ".tmp"
            
            try:
                # Atomic write: write to .tmp then rename
                with h5py.File(temp_path, 'w') as f:
                    dset = f.create_dataset("temperature", data=field, compression="gzip", compression_opts=4)
                    dset.attrs['time'] = time_val
                    dset.attrs['step'] = step
                    
                    # Store extra scalars (like laser power if present)
                    for k, v in kwargs.items():
                        if isinstance(v, (int, float, str, bool)):
                            dset.attrs[k] = v
                            
                os.replace(temp_path, path)
                logger.debug(f"Saved full field to {path}")
            except Exception as e:
                logger.warning(f"Failed to save field step {step}: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # 2. Save 1D Profiles (if provided in kwargs)
        # Expecting kwargs['profiles'] = {'x': (coords, vals), 'y': ...}
        if 'profiles' in kwargs:
            for direction, (coords, vals) in kwargs['profiles'].items():
                fname = f"{direction}_step{step:06d}.txt"
                fpath = self.get_output_path(fname, 'profiles')
                
                # Simple atomic text write
                tmp_txt = fpath + ".tmp"
                try:
                    header = f"direction={direction} time={time_val:.6e} step={step}"
                    np.savetxt(tmp_txt, np.column_stack((coords, vals)), header=header, fmt='%.6e')
                    os.replace(tmp_txt, fpath)
                except Exception as e:
                    logger.warning(f"Failed to save profile {direction}: {e}")

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