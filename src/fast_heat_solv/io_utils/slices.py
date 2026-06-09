"""2D cut view plotting utilities for evaluating simulation exports."""

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

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, griddata
import os
import logging

from .xdmf_io import load_xdmf, StructuredField, UnstructuredField

logger = logging.getLogger(__name__)


# ==========================================
# 1. DATA LOADING (using xdmf_io)
# ==========================================


def _load_data(xdmf_path):
    """
    Loads XDMF data using the consolidated load_xdmf function.

    Returns a dict compatible with the legacy format:
    - For structured: {'x', 'y', 'z', 'T', 'type': 'structured'}
    - For unstructured: {'xyz', 'T', 'type': 'unstructured'}
    """
    field = load_xdmf(xdmf_path)

    if isinstance(field, StructuredField):
        logger.info(f"Detected Format: Structured Grid (3DRectMesh)")
        return {
            "x": field.x,
            "y": field.y,
            "z": field.z,
            "T": field.T,
            "type": "structured",
        }
    else:
        logger.info(f"Detected Format: Unstructured Grid (Tetrahedron)")
        return {
            "xyz": field.xyz,
            "T": field.T,
            "type": "unstructured",
        }

# ==========================================
# 2. INTERPOLATION ENGINE
# ==========================================

def _get_slice(data, normal, center, width, height, reverse_axes=(), resolution=400, method='linear'):
    """
    Interpolates 3D data onto a 2D plane defined by a center point and dimensions.
    normal: 'x', 'y', or 'z' axis normal to the plane
    center: (x, y, z) tuple associated with the center of the slice
    width: dimension of the slice along the horizontal axis of the plot
    height: dimension of the slice along the vertical axis of the plot
    reverse_axes: list of axes ('x', 'y', 'z') to invert sign for data querying
    method: 'linear' or 'nearest' interpolation
    """
    logger.info(f"Interpolating slice Normal={normal} at Center={center}, W={width}, H={height}...")
    
    cx, cy, cz = center

    # 1. Define the 2D grid for the slice in User Coordinates
    if normal == 'z':
        # Plane is X-Y. Z is constant.
        # Width -> X, Height -> Y
        u = np.linspace(cx - width/2, cx + width/2, resolution)
        v = np.linspace(cy - height/2, cy + height/2, resolution)
        U, V = np.meshgrid(u, v)
        W = np.full_like(U, cz)
        
        # User coords: X, Y, Z
        user_points_xyz = np.stack((U.ravel(), V.ravel(), W.ravel()), axis=-1)
        xlabel, ylabel = 'X (m)', 'Y (m)'
        h_axis, v_axis = 'x', 'y'

    elif normal == 'y':
        # Plane is X-Z. Y is constant.
        # Width -> X, Height -> Z
        u = np.linspace(cx - width/2, cx + width/2, resolution)
        v = np.linspace(cz - height/2, cz + height/2, resolution)
        U, V = np.meshgrid(u, v) # U is X, V is Z
        W = np.full_like(U, cy)
        
        # User coords: X, Y, Z
        user_points_xyz = np.stack((U.ravel(), W.ravel(), V.ravel()), axis=-1)
        xlabel, ylabel = 'Scanning direction (m)', 'Build direction (m)'
        h_axis, v_axis = 'x', 'z'

    elif normal == 'x':
        # Plane is Y-Z. X is constant.
        # Width -> Y, Height -> Z
        u = np.linspace(cy - width/2, cy + width/2, resolution)
        v = np.linspace(cz - height/2, cz + height/2, resolution)
        U, V = np.meshgrid(u, v) # U is Y, V is Z
        W = np.full_like(U, cx)
        
        # User coords: X, Y, Z
        user_points_xyz = np.stack((W.ravel(), U.ravel(), V.ravel()), axis=-1)
        xlabel, ylabel = 'Transverse direction (m)', 'Build direction (m)'
        h_axis, v_axis = 'y', 'z'
    
    else:
        raise ValueError("Normal must be x, y, or z")

    # 2. Perform Interpolation
    query_points = user_points_xyz 
    if data['type'] == 'structured':
        # Prepare Interpolator if data is structured
        # RGI expects (z, y, x) axes if T is (z,y,x)
        # Note: We create the RGI on the fly. 
        # Check alignment: Standard XDMF/HDF ordering is Z, Y, X for the 3D array data['T']
        
        # RGI expects query points in (z, y, x) order
        rgi_query = np.column_stack((query_points[:,2], query_points[:,1], query_points[:,0]))
        
        rgi = RegularGridInterpolator((data['z'], data['y'], data['x']), data['T'], 
                                      method=method, bounds_error=False, fill_value=np.nan)
        slice_vals = rgi(rgi_query)

    else: # Unstructured
        points = data['xyz'] # (N, 3)
        values = data['T']   # (N,)
        
        # --- OPTIMIZATION START ---
        # Filter points to only those near the query slice to speed up 'griddata'
        # Calculate bounding box of query slice
        q_min = query_points.min(axis=0)
        q_max = query_points.max(axis=0)
        
        # Add a safety margin to ensure we capture enclosing elements for linear interpolation
        # Using 50% of the view width/height as margin is usually safe and generous enough
        scale = max(width, height)
        margin = scale * 0.5 
        
        box_min = q_min - margin
        box_max = q_max + margin
        
        # Create mask for points roughly inside the volume 
        # (This is fast vectorised numpy comparison)
        mask = (
            (points[:,0] >= box_min[0]) & (points[:,0] <= box_max[0]) &
            (points[:,1] >= box_min[1]) & (points[:,1] <= box_max[1]) &
            (points[:,2] >= box_min[2]) & (points[:,2] <= box_max[2])
        )
        
        p_sub = points[mask]
        v_sub = values[mask]
        
        # Fallback if filtering removes too much (unlikely unless margin is tiny)
        if len(p_sub) < 10: 
            logger.warning("Optimization filter removed too many points. Falling back to full mesh.")
            p_sub = points
            v_sub = values
        else:
            logger.info(f"Optimization: Reduced mesh from {len(points)} to {len(p_sub)} nodes for interpolation.")
            
        # --- OPTIMIZATION END ---
        
        # griddata expects (N, D) points and (M, D) xi.
        # Our interpolation points are (M, 3) in X,Y,Z order.
        slice_vals = griddata(p_sub, v_sub, query_points, method=method)

    slice_data = slice_vals.reshape(U.shape)
    
    # 3. Handle Reversals (Mirroring the plot)
    if h_axis in reverse_axes:
        # Flip horizontal axis (columns)
        slice_data = np.fliplr(slice_data)
        logger.info(f"Reversing horizontal axis ({h_axis})")
        
    if v_axis in reverse_axes:
        # Flip vertical axis (rows)
        slice_data = np.flipud(slice_data)
        logger.info(f"Reversing vertical axis ({v_axis})")
    
    return U, V, slice_data, xlabel, ylabel

# ==========================================
# 3. PLOTTING
# ==========================================

def _plot_meltpool(U, V, T_grid, xlabel, ylabel, liquidus, solidus, title, output_file):
    
    # --- 1. CONFIGURATION FOR ACADEMIC STYLE ---
    # This sets the font to look like LaTeX (Serif/Times)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 1.5,
        "lines.markersize": 7
    })

    # Convert to micrometers
    U = U * 1e6
    V = V * 1e6
    xlabel = xlabel.replace('(m)', '(µm)')
    ylabel = ylabel.replace('(m)', '(µm)')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Filled Contours (Temperature Map)
    # Clip data to avoid holes with extend='neither'
    T_plot = np.clip(T_grid, 500, 2500)
    
    # Plot: Banded colormap (11 discrete colors)
    cmap_plot = plt.get_cmap('jet', 11)
    levels_plot = np.linspace(500, 2500, 12) 
    
    contour_filled = ax.contourf(U, V, T_plot, levels=levels_plot, cmap=cmap_plot, extend='neither')
    
    # Colorbar: Continuous colormap
    cmap_bar = plt.get_cmap('jet')
    norm_bar = plt.Normalize(vmin=500, vmax=2500)
    # Create a separate mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_bar, norm=norm_bar)
    sm.set_array([]) 
    
    # Colorbar placement: top left, outside the plot area
    # [x, y, width, height] in axes coordinates. y > 1 puts it above.
    cax = ax.inset_axes([0.0, 1.05, 0.40, 0.05])
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal', 
                        ticks=[500, 1000, 1500, 2000, 2500])
    
    # Style colorbar (default black text for white background)
    cbar.set_label('Temperature (K)', fontsize=9)
    cbar.ax.xaxis.set_tick_params(labelsize=8)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    
    # 2. Isotherms (Liquidus and Solidus)
    # Solid lines for current melt pool
    iso_levels = [solidus, liquidus]
    colors = ['red', 'red'] # Both red as per your example, or distinct if preferred
    
    contours = ax.contour(U, V, T_grid, levels=iso_levels, colors=colors, linewidths=2)
    
    # Label the lines (optional, but good for debugging)
    # ax.clabel(contours, inline=True, fontsize=10, fmt='%1.0f K')

    # 3. Gradient Vectors (Optional Plus)
    # Calculate gradient
    try:
        # Compute gradient with respect to physical coordinates (in µm)
        # np.gradient expects coordinates for axis 0 (V) then axis 1 (U)
        # We need to construct 1D coordinate arrays for np.gradient
        # U is meshgrid, so U[0, :] gives x-coords
        # V is meshgrid, so V[:, 0] gives y-coords
        dT_dV, dT_dU = np.gradient(T_grid, V[:, 0], U[0, :])
        
        # Downsample for vector plotting so it isn't too crowded
        skip = (slice(None, None, 30), slice(None, None, 15)) # Adjust as needed 
        
        # Calculate magnitude for arrow scaling and normalization
        mag = np.sqrt(dT_dU**2 + dT_dV**2)
        avg_mag = 1.0 #np.mean(mag) if np.mean(mag) > 0 else 1.0
        print(f"Average gradient magnitude: {avg_mag:.2f} K/µm")
        
        # Normalize vectors against the average gradient magnitude
        # We plot negative gradient (heat flow direction)
        dU_norm = dT_dU / (avg_mag *500)
        dV_norm = dT_dV / (avg_mag *500)
        
        ax.quiver(U[skip], V[skip], dU_norm[skip], dV_norm[skip], 
                  color='black', alpha=0.5, scale=20, width=0.002)
    except Exception as e:
        logger.warning("Could not plot gradients: %s", e)
        print("Warning: Gradient plotting failed, skipping this step.")

    # 4. Styling
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    
    plt.tight_layout()
    
    if output_file:
        # The crucial part is bbox_inches='tight' and pad_inches=0
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
        logger.info(f"Plot saved to {output_file}")
    else:
        plt.show() # Blocking show if no output file
        
    # Close figures to avoid memory leaks when running in batches (but keep open for UI if plt.show() blocks)
    # If show() was called, it blocks until closed. If savefig, we should close.
    if output_file:
        plt.close(fig)

# ==========================================
# 4. MAIN FUNCTIONS AND EXECUTION
# ==========================================

def generate_plots(xdmf_path, output_dir=None, show_ui=True, save_images=False,
                   normal='y', center=(0.0, 0.0, 0.0), width=2e-3, height=1e-3,
                   reverse=(), liquidus=1800, solidus=1700, interp='linear',
                   specific_output_filename=None):
    """
    Main function to generate plots from an XDMF file.
    """
    if not os.path.exists(xdmf_path):
        logger.error(f"Error: XDMF file not found: {xdmf_path}")
        return

    # 1. Load Data
    try:
        data = _load_data(xdmf_path)
    except Exception as e:
        logger.exception("Error loading XDMF: %s", e)
        return

    # 2. Interpolate Slice
    try:
        U, V, T_grid, xlabel, ylabel = _get_slice(data, normal, center, width, height, reverse_axes=reverse, method=interp)
    except Exception as e:
        logger.exception("Error extracting slice: %s", e)
        return

    # 3. Plot
    output_file = None
    if save_images:
        if specific_output_filename:
             output_file = specific_output_filename
             # Ensure directory exists for specific file
             os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        else:
            # Determine output directory
            if output_dir is None:
                # Default to same folder as XDMF
                output_dir = os.path.dirname(os.path.abspath(xdmf_path))
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            base_name = os.path.splitext(os.path.basename(xdmf_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}_cut_{normal}.png")

    _plot_meltpool(U, V, T_grid, xlabel, ylabel, liquidus, solidus, 
                  f"Section Normal-{normal.upper()} @ {center}", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Meltpool X-Sections from XDMF.")
    parser.add_argument("xdmf_file", help="Path to input .xmf or .xdmf file")
    parser.add_argument("--normal", default="y", choices=['x', 'y', 'z'], help="Normal of the cut plane")
    
    parser.add_argument("--center", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Center point of the cut plane (x y z)")
    parser.add_argument("--width", type=float, default=2e-3, help="Width of the cut view")
    parser.add_argument("--height", type=float, default=1e-3, help="Height of the cut view")
    parser.add_argument("--reverse", nargs='*', default=[], choices=['x', 'y', 'z'], help="Axes to reverse for querying data (e.g. --reverse x)")
    
    parser.add_argument("--liquidus", type=float, default=1800, help="Liquidus temperature (K)")
    parser.add_argument("--solidus", type=float, default=1700, help="Solidus temperature (K)")
    parser.add_argument("--interp", default="linear", choices=['linear', 'nearest'], help="Interpolation method (linear or nearest)")
    parser.add_argument("--out", default=None, help="Output image filename (overrides auto-generation)")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    
    args = parser.parse_args()

    save_images = args.out is not None
    show_ui = not args.no_show

    # If args.out is provided, it is a specific filename. 
    # We pass it as specific_output_filename to generate_plots.
    # Typical call :python utils/cut_views.py out/0_perfect_sim/fields/field_step002000.xmf --center 0.0095 0.0025 0.002475 --width 0.00035 --height 0.00005 --liquidus 1820 --out cut_spectral.pdf

    generate_plots(
        xdmf_path=Path(args.xdmf_file),
        output_dir=None, # Not used if specific_output_filename is set or save_images is False (mostly)
        show_ui=show_ui,
        save_images=save_images, 
        normal=args.normal,
        center=tuple(args.center),
        width=args.width,
        height=args.height,
        reverse=args.reverse if isinstance(args.reverse, list) else [],
        liquidus=args.liquidus,
        solidus=args.solidus,
        interp=args.interp,
        specific_output_filename=args.out
    )
