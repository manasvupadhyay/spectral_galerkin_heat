"""2D slice plotting utilities for evaluating simulation exports."""

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

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

from .xdmf_io import StructuredField, load_xdmf

logger = logging.getLogger(__name__)


# ==========================================
# 1. DATA LOADING (using xdmf_io)
# ==========================================


def _load_data(xdmf_path):
    """
    Load an XDMF file and flatten it to the dict this module works in.

    :func:`load_xdmf` returns a typed ``StructuredField`` or
    ``UnstructuredField``; the slicing and interpolation code below branches on
    a ``type`` string instead, so this adapts one to the other:

    - structured: ``{'x', 'y', 'z', 'T', 'type': 'structured'}``
    - unstructured: ``{'xyz', 'T', 'type': 'unstructured'}``
    """
    field = load_xdmf(xdmf_path)

    if isinstance(field, StructuredField):
        logger.info("Detected Format: Structured Grid (3DRectMesh)")
        return {
            "x": field.x,
            "y": field.y,
            "z": field.z,
            "T": field.T,
            "type": "structured",
        }
    else:
        logger.info("Detected Format: Unstructured Grid (Tetrahedron)")
        return {
            "xyz": field.xyz,
            "T": field.T,
            "type": "unstructured",
        }

# ==========================================
# 2. INTERPOLATION
# ==========================================

_PLANE_LABELS = {
    "z": ("X (m)", "Y (m)", "x", "y"),
    "y": ("X (m)", "Z (m)", "x", "z"),
    "x": ("Y (m)", "Z (m)", "y", "z"),
}


def _axis_labels(normal):
    """Return ``(xlabel, ylabel, h_axis, v_axis)`` for a slice *normal*."""
    try:
        return _PLANE_LABELS[normal]
    except KeyError:
        raise ValueError("Normal must be x, y, or z") from None


def _parse_normal(normal):
    """Split a possibly-signed *normal* (e.g. ``'-y'``) into ``(axis, flip_h)``.

    A leading ``'-'`` means the plane is viewed from the negative side of the
    axis, which mirrors the in-plane horizontal perspective. Returns the
    unsigned axis (``'x'``/``'y'``/``'z'``) that the interpolation geometry
    uses, and a boolean for whether the horizontal in-plane axis should be
    flipped.
    """
    s = str(normal).strip().lower()
    flip_h = s.startswith('-')
    axis = s.lstrip('+-')
    if axis not in _PLANE_LABELS:
        raise ValueError("Normal must be one of x, -x, y, -y, z, -z")
    return axis, flip_h


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
    xlabel, ylabel, h_axis, v_axis = _axis_labels(normal)  # also validates normal

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

    elif normal == 'y':
        # Plane is X-Z. Y is constant.
        # Width -> X, Height -> Z
        u = np.linspace(cx - width/2, cx + width/2, resolution)
        v = np.linspace(cz - height/2, cz + height/2, resolution)
        U, V = np.meshgrid(u, v) # U is X, V is Z
        W = np.full_like(U, cy)
        
        # User coords: X, Y, Z
        user_points_xyz = np.stack((U.ravel(), W.ravel(), V.ravel()), axis=-1)

    elif normal == 'x':
        # Plane is Y-Z. X is constant.
        # Width -> Y, Height -> Z
        u = np.linspace(cy - width/2, cy + width/2, resolution)
        v = np.linspace(cz - height/2, cz + height/2, resolution)
        U, V = np.meshgrid(u, v) # U is Y, V is Z
        W = np.full_like(U, cx)
        
        # User coords: X, Y, Z
        user_points_xyz = np.stack((W.ravel(), U.ravel(), V.ravel()), axis=-1)

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

def _plot_meltpool(U, V, T_grid, xlabel, ylabel, isotherms, title, output_file):
    
    # --- 1. CONFIGURATION FOR ACADEMIC STYLE ---
    # This sets the font to look like LaTeX (Serif/Times)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
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
    
    ax.contourf(U, V, T_plot, levels=levels_plot, cmap=cmap_plot, extend='neither')
    
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
    
    # 2. Isotherms (Parameterized)
    if isotherms:
        from matplotlib.lines import Line2D

        # Extract levels for the contour function
        levels = [val for name, val in isotherms]

        # Plot contours (using distinct line styles if you want, but standard red works well)
        ax.contour(U, V, T_grid, levels=levels, colors=['red'] * len(levels), linewidths=2)

        # Create custom legend handles
        legend_elements = []
        for name, val in isotherms:
            legend_elements.append(
                Line2D([0], [0], color='red', lw=2, label=f"{name} ({val:g} K)")
            )

        # Place legend outside the plot, to the right of the colorbar.
        # Colorbar is anchored at x=0.0, y=1.05, width=0.40.
        # We start the legend at x=0.45 to give it some padding.
        ax.legend(handles=legend_elements, loc='lower left',
                  bbox_to_anchor=(0.45, 1.05), frameon=False,
                  fontsize=9, handlelength=1.5)

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
        
        # Arrow scaling: fixed normalisation, so arrows stay comparable
        # between frames rather than rescaling to each frame's own gradient.
        avg_mag = 1.0
        
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
                   reverse=(), isotherms=None, interp='linear',
                   specific_output_filename=None):
    """
    Main function to generate plots from an XDMF file.
    """
    if not os.path.exists(xdmf_path):
        logger.error(f"Error: XDMF file not found: {xdmf_path}")
        return

    # A signed normal ('-y') is a pure viewing choice: strip the sign so the
    # interpolation geometry sees the bare axis, and mirror the horizontal
    # in-plane axis to reproduce the "looking from the negative side" view.
    axis, flip_h = _parse_normal(normal)
    reverse = list(reverse)
    if flip_h:
        h_axis = _PLANE_LABELS[axis][2]
        if h_axis not in reverse:
            reverse.append(h_axis)

    # 1. Load Data
    try:
        data = _load_data(xdmf_path)
    except Exception as e:
        logger.exception("Error loading XDMF: %s", e)
        return

    # 2. Interpolate Slice
    try:
        U, V, T_grid, xlabel, ylabel = _get_slice(data, axis, center, width, height, reverse_axes=reverse, method=interp)
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
            output_file = os.path.join(output_dir, f"{base_name}_cut_{axis}.png")

    _plot_meltpool(U, V, T_grid, xlabel, ylabel, isotherms,
                  f"Section Normal-{normal.upper()} @ {center}", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and plot a 2-D slice from a 3-D XDMF field. "
                    "The slice is a pure geometric cut defined by a signed "
                    "normal, an absolute centre, and in-plane extents."
    )

    io_group = parser.add_argument_group("input / output")
    io_group.add_argument("xdmf_file", help="Path to input .xmf or .xdmf file")
    io_group.add_argument("--out", default=None,
                          help="Output image filename (overrides auto-generation). "
                               "When omitted the plot is only displayed.")
    io_group.add_argument("--no-show", action="store_true",
                          help="Do not display the plot window")

    geom_group = parser.add_argument_group("slice geometry")
    geom_group.add_argument("--normal", default="y",
                            choices=['x', 'y', 'z', '-x', '-y', '-z'],
                            help="Signed axis normal to the slice plane. The plane "
                                 "spans the two remaining axes; a leading '-' views "
                                 "it from the negative side (mirrors the in-plane "
                                 "horizontal axis). Pass negative values with an "
                                 "equals sign, e.g. --normal=-y.")
    geom_group.add_argument("--center", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                            metavar=('X', 'Y', 'Z'),
                            help="Absolute centre of the slice plane in domain "
                                 "coordinates (x y z) [m].")
    geom_group.add_argument("--width", type=float, required=True,
                            help="In-plane horizontal (h) extent of the slice [m].")
    geom_group.add_argument("--height", type=float, required=True,
                            help="In-plane vertical (v) extent of the slice [m].")
    geom_group.add_argument("--reverse", nargs='*', default=[], choices=['x', 'y', 'z'],
                            help="In-plane axes to additionally mirror when querying data.")

    contour_group = parser.add_argument_group("contour levels")
    contour_group.add_argument("--isotherm", nargs=2, action="append", dest="isotherms",
                               metavar=("NAME", "VALUE"),
                               help="Define an isotherm to plot (e.g., --isotherm T_s 1674.15). "
                                    "Can be used multiple times. If omitted, no isotherms are drawn.")
    contour_group.add_argument("--interp", default="linear", choices=['linear', 'nearest'],
                               help="Interpolation method (linear or nearest)")

    args = parser.parse_args()

    save_images = args.out is not None
    show_ui = not args.no_show

    # Convert isotherm string values to floats
    parsed_isotherms = []
    if args.isotherms:
        for name, val in args.isotherms:
            parsed_isotherms.append((name, float(val)))

    # If args.out is provided, it is a specific filename passed straight through
    # as specific_output_filename.
    # Typical call: python -m spectral_galerkin_heat.io_utils.slices field_step002000.xmf \
    #   --normal=-y --center 0.0095 0.0025 0.002475 --width 3.5e-4 --height 5e-5 \
    #   --isotherm T_s 1674.15 --out slice.png
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
        isotherms=parsed_isotherms,
        interp=args.interp,
        specific_output_filename=args.out
    )
