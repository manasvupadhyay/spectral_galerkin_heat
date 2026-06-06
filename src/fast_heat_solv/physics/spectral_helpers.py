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

import numpy as np
import os
import scipy.fft

try:
    import cupy as _cp
except ImportError:
    _cp = None

# Default to CPU kernels for module-level access, but dispatch properly in functions
import fast_heat_solv.physics.spectral_cpu_kernels as kernels


def _get_array_module(arr):
    if _cp is not None and hasattr(arr, 'device'): # Check if it's a cupy array
        return _cp
    return np

def _get_kernels(arr):
    xp = _get_array_module(arr)
    if xp is _cp:
         import fast_heat_solv.physics.spectral_gpu_kernels as gpu_kernels
         return gpu_kernels
    return kernels


def _C_coef(N, L, xp=np):
    """
    Compute normalization coefficients for DCT-II.

    Parameters
    ----------
    N : int
        Number of modes.
    L : float
        Domain length.
    xp : module, optional
        Array module to use (e.g., numpy or cupy), by default np.

    Returns
    -------
    ndarray
        Normalization coefficients of size N.
    """
    C = xp.sqrt(2.0 / L) * xp.ones(N)
    C[0] = xp.sqrt(1.0 / L)
    return C


def _calculate_subgrid_indices(pos, dx, n_total_fine, n_box):
    """Center a box of ``n_box`` cells around a physical position.

    Backend-agnostic: operates on plain Python scalars, so it is shared by both
    the CPU and GPU kernel modules.

    Parameters
    ----------
    pos : float
        Physical position (laser center).
    dx : float
        Grid spacing.
    n_total_fine : int
        Total number of points in the fine grid.
    n_box : int
        Number of points in the active box.

    Returns
    -------
    tuple of int
        ``(idx_start, idx_end, idx_relative)`` — box start/end indices and the
        position's index relative to the box start.
    """
    # Nearest global index for the center position, clamped to the fine grid.
    idx_global = int(round(max(0.0, min(pos / dx, n_total_fine - 1))))

    # Desired start index to center the box, clamped to [0, max_start].
    idx_start = idx_global - n_box // 2
    max_start = max(0, n_total_fine - n_box)
    idx_start = max(0, min(idx_start, max_start))
    idx_end = idx_start + n_box

    # Position index relative to the box start, clamped to the box.
    idx_relative = idx_global - idx_start
    idx_relative = max(0, min(idx_relative, n_box - 1))

    return idx_start, idx_end, idx_relative

def _cosine_basis_along_axis(n_modes, length, coords):
    """
    Compute cosine basis values cos(k*pi*x/L) for given coordinates.

    Parameters
    ----------
    n_modes : int
        Number of modes to compute.
    length : float
        Domain length along the axis.
    coords : ndarray
        1D array of coordinates where the basis is evaluated.

    Returns
    -------
    ndarray
        2D array of shape (n_modes, len(coords)) with basis values.
    """
    indices = np.arange(n_modes, dtype=np.float64)
    return np.cos(np.pi * indices[:, None] * coords[None, :] / length)

def reconstruct_temperature_volume(a, SsState):
    """
    Reconstruct the temperature field on the full simulation grid.
    
    note : the reconstruction bases (Bx, By, Bz) must be precomputed before.
    The reconstruction grid is node centered in x, y, z.
    
    This function is now legacy, can be still used for exact reconstruction 
    but DCT version is faster. (DCT can be tested against this for correctness).

    Parameters
    ----------
    a : ndarray
        Spectral coefficients.
    SsState : fast_heat_solv.physics.spectral_cpu_kernels.SpectralSolverState
        The solver state containing grid reconstruction bases.

    Returns
    -------
    ndarray
        Full volumetric temperature field array.
    """

    if hasattr(a, 'get'):
        a = a.get()  # Move to CPU if it's a CuPy array
    
    grid = SsState.grid

    if grid.B_recon is None:
        raise RuntimeError("Reconstruction bases not initialized on SsState. Call prepare_full_reconstruction() first.")
    Bx, By, Bz = grid.B_recon  # (modes_axis, n_points+1) for each axis

    T_step1 = np.tensordot(a, Bx, axes=(2, 0))  # (N_z, N_y, N_x)
    T_step2 = np.tensordot(T_step1, By, axes=(1, 0))  # (nz, nx, ny)
    T_full = np.tensordot(T_step2, Bz, axes=(0, 0))  # (N_x, N_y, N_z)

    return T_full.astype(np.float32)

def reconstruct_temperature_DCT(a, SsState):
    """Reconstruct node-centered temperature field using DCT type I.

    The modal expansion is::

        T(x,y,z) = sum_{m,n,p} a[p,n,m] * Cm*cos(m*pi*x/Lx)
                                          * Cn*cos(n*pi*y/Ly)
                                          * Cp*cos(p*pi*z/Lz)

    evaluated on the **node-centered** grid  x_j = j*dx  (j = 0..nx),
    and likewise for y and z.  Output shape is **(N_x+1, N_y+1, N_z+1)**,
    identical to :func:`reconstruct_temperature_volume`.

    The reconstruction grid is integer-spaced (node-centred), which maps
    to DCT type I.  By padding N modes to length N+1 and halving interior
    modes, the DCT-I output equals the desired cosine series at node points.
    Complexity is O(N^3 log N) vs O(N^4) for explicit tensor products.

    Parameters
    ----------
    a : ndarray, shape (N_z, N_y, N_x)
        Spectral coefficients (CuPy arrays are moved to CPU automatically).
    SsState : fast_heat_solv.physics.spectral_cpu_kernels.SpectralSolverState
        Must have ``grid.C`` normalization tuple (C[0]=x, C[1]=y, C[2]=z).

    Returns
    -------
    T : ndarray, shape (N_x+1, N_y+1, N_z+1), dtype float32
        Node-centred temperature field.
    """
    if hasattr(a, 'get'):
        a = a.get()

    grid = SsState.grid
    nz, ny, nx = a.shape

    # ── Fused 1-D weight vectors: normalization × DCT-I halving ──────
    # Combined weight[i] = C[i] * (0.5 if i>0 else 1.0)
    # Precomputed as 1-D float32 vectors (6 elements total).
    # Handle both numpy and cupy arrays (GPU solver uses cupy)
    wx, wy, wz = (np.array(c.get() if hasattr(c, 'get') else c, dtype=np.float32) for c in grid.C)
    wx[1:] *= 0.5; wy[1:] *= 0.5; wz[1:] *= 0.5

    # ── Scale on contiguous memory, then copy once into padded ───────
    # Working on a contiguous copy of `a` is faster than writing
    # through the non-contiguous slice padded[:nz,:ny,:nx].
    b = a.copy()                         # contiguous (nz,ny,nx)
    b *= wz[:, None, None]
    b *= wy[None, :, None]
    b *= wx[None, None, :]

    padded = np.empty((nz + 1, ny + 1, nx + 1), dtype=np.float32)
    padded[:nz, :ny, :nx] = b            # single contiguous-to-strided copy
    padded[nz, :, :] = 0.0               # zero the 3 padding planes
    padded[:, ny, :] = 0.0
    padded[:, :, nx] = 0.0

    # ── 3-D DCT-I (type 1, unnorm) → node-centred values ────────────
    # scipy.fft.dctn (pocketfft) is ~4× faster than pyfftw for these sizes.
    T = scipy.fft.dctn(padded, type=1, norm=None, axes=(0, 1, 2),
                        overwrite_x=True, workers=-1)

    # ── Transpose (nz+1, ny+1, nx+1) → (N_x+1, N_y+1, N_z+1) ─────────
    return np.ascontiguousarray(T.transpose(2, 1, 0), dtype=np.float32)

def reconstruct_temperature_volume_at_points(a, num, geom, SsState, coords):
    """
    Evaluate the temperature field at arbitrary points using modal expansion.

    Parameters
    ----------
    a : ndarray
        Spectral coefficients (CPU or GPU array).
    num : NumParams
        Simulation numerical parameters containing mesh discretizations.
    geom : GeomParams
        Simulation domain geometry.
    SsState : fast_heat_solv.physics.spectral_cpu_kernels.SpectralSolverState
        Current solver state.
    coords : ndarray
        The (N, 3) shaped array containing float coordinates to evaluate at.

    Returns
    -------
    ndarray
        Array of shape N with temperature values at each queried point.
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be of shape (N, 3)")

    x_vals = np.clip(coords[:, 0], 0.0, geom.Lx)
    y_vals = np.clip(coords[:, 1], 0.0, geom.Ly)
    z_vals = np.clip(coords[:, 2], 0.0, geom.Lz)

    # Ensure modal coefficient arrays and spectral coefficients are NumPy arrays
    # This avoids mixed NumPy/CuPy arithmetic when CPU-based helpers are used.
    def _to_numpy(x):
        # If x is a CuPy array with .get(), move to host; otherwise use np.asarray
        if hasattr(x, 'get') and callable(x.get):
            return np.asarray(x.get())
        return np.asarray(x)
    
    # Support both old monolithic state and new decoupled state
    grid = SsState.grid if hasattr(SsState, 'grid') else SsState

    C = [_to_numpy(c) for c in grid.C]
    a_np = _to_numpy(a).astype(np.float32)

    Bx = (C[0][:, None] * _cosine_basis_along_axis(num.nx, geom.Lx, x_vals)).astype(np.float32)
    By = (C[1][:, None] * _cosine_basis_along_axis(num.ny, geom.Ly, y_vals)).astype(np.float32)
    Bz = (C[2][:, None] * _cosine_basis_along_axis(num.nz, geom.Lz, z_vals)).astype(np.float32)
    temps = np.einsum('pnm,pi,ni,mi->i', a_np, Bz, By, Bx, optimize=True)

    return temps.astype(np.float32)


def save_temp_profiles(
    a,
    num,
    geom,
    SsState,
    laser_position,
    center="laser",
    num_points=1000):
    """
    Sample temperature along dense lines using exact modal expansion.
    
    Args:
        a: Spectral coefficients (N_z, N_y, N_x).
        num: Numerical params (N_x, N_y, N_z).
        geom: Geometric params (Lx, Ly, Lz).
        SsState: fast_heat_solv.physics.spectral_cpu_kernels.SpectralSolverState
        laser: Laser object (for centering).
        center: "laser", "hotspot", or tuple (x, y).
        num_points: Number of sampling points along each axis.
        output_dir: Directory to save .txt files.
    """

    # 1. Determine Sample Center (Intersection Point)
    # Use the TOP surface (z = Lz) as reference
    z_top = float(geom.Lz)


    if center == "hotspot":
        # Scan low-res surface to find approximate max
        # This requires reconstructing a 2D slice first
        # For efficiency, we reconstruct T_surf from kernels
        loc_kernels = _get_kernels(a)
        T_surf = loc_kernels.reconstruct_surface_temperature(a, SsState)
        
        # Ensure T_surf is on CPU for coordinate extraction
        if hasattr(T_surf, 'get'):
            T_surf = T_surf.get()
            
        iy_idx, ix_idx = np.unravel_index(np.argmax(T_surf), T_surf.shape)
        
        # Handle decoupled state or monolithic state
        grid = SsState.grid if hasattr(SsState, 'grid') else SsState
        x_center = grid.x[ix_idx] 
        y_center = grid.y[iy_idx]
        
        # Helper to safely scalarize
        def _scalar(val):
            if hasattr(val, 'item'): return val.item()
            return val
            
        x_center = _scalar(x_center)
        y_center = _scalar(y_center)
        
    elif center == "laser":
        if laser_position is None:
            raise ValueError("Laser position required for center='laser'")
        x_center, y_center = float(laser_position[0]), float(laser_position[1])
    else:
        raise ValueError(f"Unknown center method: {center}")

    # Clamp to domain
    x_center = float(np.clip(x_center, 0.0, geom.Lx))
    y_center = float(np.clip(y_center, 0.0, geom.Ly))

    # 2-4. Generate coords, build point clouds, and evaluate temperature for each axis
    # Each entry: (key, domain_length, (fixed_x_or_None, fixed_y_or_None, fixed_z_or_None))
    axes_cfg = [
        ('x', geom.Lx, (None, y_center, z_top)),
        ('y', geom.Ly, (x_center, None, z_top)),
        ('z', geom.Lz, (x_center, y_center, None)),
    ]
    profiles = {}
    for key, L_axis, fixed in axes_cfg:
        coords = np.linspace(0.0, L_axis, num_points, dtype=np.float64)
        pts = np.empty((num_points, 3), dtype=np.float64)
        for col, v in enumerate(fixed):
            pts[:, col] = coords if v is None else v
        profiles[key] = (coords, reconstruct_temperature_volume_at_points(a, num, geom, SsState, pts))
    return profiles
    
    


