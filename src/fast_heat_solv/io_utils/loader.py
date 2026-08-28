import logging
import os
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def read_field_h5(path: str) -> dict[str, Any] | None:
    """Read a saved field HDF5 file into a dict.

    Returns ``{'temperature', 'time', 'step', **attrs}`` or ``None`` if the file
    is missing/unreadable. Shared by :meth:`SimulationResult.get_field` and
    :meth:`LocalFSIOManager.load_step` so the two stay in sync.
    """
    if not os.path.exists(path):
        return None
    try:
        with h5py.File(path, 'r') as f:
            t = f['temperature']
            data = {
                'temperature': t[:],
                'time': t.attrs.get('time', 0.0),
                'step': t.attrs.get('step', -1),
            }
            for k, v in t.attrs.items():
                data[k] = v
            return data
    except Exception as e:
        logger.error(f"Failed to read field {path}: {e}")
        return None


class SimulationResult:
    """
    Interface for loading simulation results from a specific run directory.
    
    Assumes standard LocalFSIO structure:
    
      root/run_id/
        fields/
        profiles/
        logs/
    """
    def __init__(self, run_root: str):
        self.run_root = run_root
        self.fields_dir = os.path.join(run_root, 'fields')
        self.profiles_dir = os.path.join(run_root, 'profiles')
        self.logs_dir = os.path.join(run_root, 'logs')
        
        if not os.path.exists(self.run_root):
            raise FileNotFoundError(f"Run directory not found: {self.run_root}")

    def list_steps(self) -> list[int]:
        """Return a sorted list of available step indices based on profiles."""
        steps = set()
        if os.path.exists(self.profiles_dir):
            for f in os.listdir(self.profiles_dir):
                if f.endswith(".txt") and "step" in f:
                    # Extract step ID assumes format: direction_stepXXXXXX.txt
                    try:
                        part = f.split("step")[1].split(".txt")[0]
                        steps.add(int(part))
                    except (IndexError, ValueError):
                        pass
        return sorted(steps)

    def get_profile(self, step: int, direction: str) -> tuple | None:
        """
        Load 1D profile for a given step and direction.
        Returns (coords, values) tuple, or None if not found.
        """
        fname = f"{direction}_step{step:06d}.txt"
        path = os.path.join(self.profiles_dir, fname)
        
        if not os.path.exists(path):
            logger.warning(f"Profile not found: {path}")
            return None
            
        try:
            data = np.loadtxt(path)
            # Assuming 2 columns: coord, value
            return data[:, 0], data[:, 1]
        except Exception as e:
            logger.error(f"Error loading profile {path}: {e}")
            return None

    def get_field(self, step: int) -> dict[str, Any] | None:
        """
        Load full 3D field for a given step if available (HDF5).
        Returns dict containing 'temperature', 'time', etc.
        """
        fname = f"field_step{step:06d}.h5"
        return read_field_h5(os.path.join(self.fields_dir, fname))


def list_runs(output_root: str = "out") -> list[str]:
    """List all available run IDs in the output root directory."""
    if not os.path.exists(output_root):
        return []
    
    # Filter only directories that look like runs (optional logic)
    runs = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]
    return sorted(runs, reverse=True) # Newest first


def load_run(run_id: str, output_root: str = "out") -> SimulationResult:
    """Factory function to load a simulation result."""
    path = os.path.join(output_root, run_id)
    return SimulationResult(path)
