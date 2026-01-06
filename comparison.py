"""
Comparison of spectral method results with finite element results.

This script loads temperature profiles from out (spectral) and .validation (FE)
directories and compares them.


"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def load_temperature_profiles(out_dir="out", validation_dir=".validation", file_tag="spectral"):
    """Load temperature profiles from spectral and FE methods.
    
    Args:
        out_dir: Directory for simulation results.
        validation_dir: Directory for validation data.
        file_tag: Tag to construct filename (e.g., 'spectral' for *_spectral_latent_heat.txt 
                  or 'fine' for *_fine_latent_heat.txt).

    Returns:
        dict: Dictionary containing loaded profiles with keys:
              'x_spectral', 'y_spectral', 'z_spectral',
              'x_FE', 'y_FE', 'z_FE'
              Each value is a tuple (coords, temperatures)
    """
    profiles = {}
    
    # Load spectral/simulation method results
    for direction in ['x', 'y', 'z']:
        # Construct filename based on tag, e.g., "x_fine_latent_heat.txt"
        filename = f"{direction}_{file_tag}_latent_heat.txt"
        filepath = os.path.join(out_dir, filename)
        
        if os.path.exists(filepath):
            data = np.loadtxt(filepath)
            # Store under '_spectral' key regardless of tag to maintain compatibility with plotting functions
            profiles[f'{direction}_spectral'] = (data[:, 0], data[:, 1])
            print(f"Loaded {filepath}: {len(data)} points")
        else:
            print(f"Warning: {filepath} not found")
            profiles[f'{direction}_spectral'] = (np.array([]), np.array([]))
    
    # Load finite element results
    for direction in ['x', 'y', 'z']:
        filepath = os.path.join(validation_dir, f"{direction}_FE_latent_heat.txt")
        if os.path.exists(filepath):
            data = np.loadtxt(filepath)
            # Filter out NaN values
            mask = ~np.isnan(data[:, 1])
            profiles[f'{direction}_FE'] = (data[mask, 0], data[mask, 1])
            print(f"Loaded {filepath}: {np.sum(mask)} valid points (filtered {np.sum(~mask)} NaN)")
        else:
            print(f"Warning: {filepath} not found")
            profiles[f'{direction}_FE'] = (np.array([]), np.array([]))
    
    return profiles


def plot_comparison(profiles, output_file="temperature_comparison.png", label_sim="Simulation"):
    """Create comparison plots for all three directions with derivatives and error curves.
    
    Args:
        profiles: Dictionary returned by load_temperature_profiles
        output_file: Path to save the comparison figure (will be suffixed with _x, _y, _z)
        label_sim: Label to use for the simulation data in the legend.
    """
    directions = ['x', 'y', 'z']
    labels = ['x (m)', 'y (m)', 'z (m)']
    
    base_name, ext = os.path.splitext(output_file)
    
    for direction, label in zip(directions, labels):
        # Get data
        coords_spec, T_spec = profiles[f'{direction}_spectral']
        coords_FE, T_FE = profiles[f'{direction}_FE']
        
        if len(coords_spec) == 0 or len(coords_FE) == 0:
            print(f"Skipping {direction}: missing data")
            continue
            
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        
        # --- Top Plot: Temperature ---
        # Plot Spectral
        ax1.plot(coords_spec * 1e3, T_spec, 'b-', label=label_sim, linewidth=2)
        # Plot FE
        ax1.plot(coords_FE * 1e3, T_FE, 'r--', label='Finite Element', linewidth=2)
        
        # Compute and plot error (interpolate FE to Spectral)
        # We use the spectral grid as the reference for error calculation
        if len(coords_FE) > 1:
            # Ensure FE coords are sorted for interpolation
            sort_idx = np.argsort(coords_FE)
            coords_FE_sorted = coords_FE[sort_idx]
            T_FE_sorted = T_FE[sort_idx]
            
            T_FE_interp = np.interp(coords_spec, coords_FE_sorted, T_FE_sorted, left=np.nan, right=np.nan)
            error_T = np.abs(T_spec - T_FE_interp)
            
            ax1_err = ax1.twinx()
            ax1_err.plot(coords_spec * 1e3, error_T, 'g-', label='Error', linewidth=1.5, alpha=0.7)
            ax1_err.set_ylabel('Abs. Error (K)', color='g')
            ax1_err.tick_params(axis='y', labelcolor='g')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_err.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax1.legend(loc='upper right')

        ax1.set_ylabel('Temperature (K)')
        ax1.set_title(f'Temperature Profile along {direction.upper()}')
        ax1.grid(True, alpha=0.3)
        

        # --- Bottom Plot: Derivative ---
        # Compute derivatives (dT/dx)
        if len(coords_spec) > 1:
            dT_spec = np.gradient(T_spec, coords_spec)
            ax2.plot(coords_spec * 1e3, dT_spec, 'b-', label=f'{label_sim} Deriv.', linewidth=2)
        
        if len(coords_FE) > 1:
            dT_FE = np.gradient(T_FE, coords_FE)
            ax2.plot(coords_FE * 1e3, dT_FE, 'r--', label='FE Deriv.', linewidth=2)
        
        # Compute and plot derivative error
        if len(coords_spec) > 1 and len(coords_FE) > 1:
            # Interpolate FE derivative to spectral grid
            dT_FE_interp = np.interp(coords_spec, coords_FE_sorted, dT_FE, left=np.nan, right=np.nan)
            error_dT = np.abs(dT_spec - dT_FE_interp)
            
            ax2_err = ax2.twinx()
            ax2_err.plot(coords_spec * 1e3, error_dT, 'g-', label='Deriv. Error', linewidth=1.5, alpha=0.7)
            ax2_err.set_ylabel('Abs. Error (K/m)', color='g')
            ax2_err.tick_params(axis='y', labelcolor='g')
            
            # Combine legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_err.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax2.legend(loc='upper right')
        
        ax2.set_xlabel(f'{label.split()[0]} (mm)')
        ax2.set_ylabel('Temperature Gradient (K/m)')
        ax2.set_title(f'Temperature Gradient along {direction.upper()}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        current_output = f"{base_name}_{direction}{ext}"
        plt.savefig(current_output, dpi=150)
        print(f"Comparison plot saved to: {current_output}")
    
    plt.show()


def compute_metrics(profiles):
    """Compute comparison metrics between spectral and FE methods.
    
    Args:
        profiles: Dictionary returned by load_temperature_profiles
    
    Returns:
        dict: Metrics for each direction
    """
    metrics = {}
    
    for direction in ['x', 'y', 'z']:
        coords_spec, T_spec = profiles[f'{direction}_spectral']
        coords_FE, T_FE = profiles[f'{direction}_FE']
        
        if len(coords_spec) == 0 or len(coords_FE) == 0:
            print(f"Skipping metrics for {direction}: missing data")
            continue
        
        # Interpolate FE data onto spectral grid for comparison
        T_FE_interp = np.interp(coords_spec, coords_FE, T_FE, left=np.nan, right=np.nan)
        
        # Remove NaN values (outside FE domain)
        valid_mask = ~np.isnan(T_FE_interp)
        T_spec_valid = T_spec[valid_mask]
        T_FE_valid = T_FE_interp[valid_mask]
        
        if len(T_spec_valid) == 0:
            print(f"No overlapping points for {direction}")
            continue
        
        # Compute metrics
        max_T_spec = np.max(T_spec_valid)
        max_T_FE = np.max(T_FE_valid)
        mae = np.mean(np.abs(T_spec_valid - T_FE_valid))
        rmse = np.sqrt(np.mean((T_spec_valid - T_FE_valid)**2))
        max_diff = np.max(np.abs(T_spec_valid - T_FE_valid))
        
        metrics[direction] = {
            'max_T_spectral': max_T_spec,
            'max_T_FE': max_T_FE,
            'mae': mae,
            'rmse': rmse,
            'max_diff': max_diff,
            'n_points': len(T_spec_valid)
        }
        
        print(f"\n{direction.upper()}-direction metrics:")
        print(f"  Max T (Spectral): {max_T_spec:.2f} K")
        print(f"  Max T (FE):       {max_T_FE:.2f} K")
        print(f"  MAE:              {mae:.2f} K")
        print(f"  RMSE:             {rmse:.2f} K")
        print(f"  Max difference:   {max_diff:.2f} K")
        print(f"  Comparison points: {len(T_spec_valid)}")
    
    return metrics


def apply_transforms(profiles, align_target='none', invert_target='none', invert_axes=()):
    """Apply coordinate transforms to profiles in-place.

    Args:
        profiles: dict of loaded profiles
        align_target: which dataset to translate so its zero matches the other's zero.
                      Options: 'none', 'FE', 'spectral', 'both'. If 'FE', the FE profile
                      is translated so its minimum coordinate equals the spectral minimum.
        invert_target: 'none', 'FE', 'spectral', or 'both' — which dataset(s) to invert sign for
        invert_axes: iterable of axis names to invert, e.g. ('x','y')
    """
    def sort_pair(coords, vals):
        if len(coords) == 0:
            return coords, vals
        order = np.argsort(coords)
        return coords[order], vals[order]

    for direction in ['x', 'y', 'z']:
        spec_key = f'{direction}_spectral'
        fe_key = f'{direction}_FE'

        coords_spec, T_spec = profiles[spec_key]
        coords_FE, T_FE = profiles[fe_key]

        # Apply inversion if requested (flip sign). We then sort so coords are ascending.
        if direction in invert_axes and invert_target in ('spectral', 'both'):
            coords_spec = -coords_spec
        if direction in invert_axes and invert_target in ('FE', 'both'):
            coords_FE = -coords_FE

        coords_spec, T_spec = sort_pair(coords_spec, T_spec)
        coords_FE, T_FE = sort_pair(coords_FE, T_FE)

        # Align starts by translating only the requested target so its minimum equals the other's
        if align_target != 'none' and len(coords_spec) > 0 and len(coords_FE) > 0:
            if align_target == 'FE':
                # shift FE so its min matches spectral min
                coords_FE = coords_FE - coords_FE.min() + coords_spec.min()
            elif align_target == 'spectral':
                # shift spectral so its min matches FE min
                coords_spec = coords_spec - coords_spec.min() + coords_FE.min()
            elif align_target == 'both':
                # fall back to previous behavior: translate both so global min is zero
                start = min(coords_spec.min(), coords_FE.min())
                coords_spec = coords_spec - start
                coords_FE = coords_FE - start

        profiles[spec_key] = (coords_spec, T_spec)
        profiles[fe_key] = (coords_FE, T_FE)

    return profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare spectral and FE temperature profiles.")
    parser.add_argument('--out-dir', default='out', help='Directory with spectral outputs')
    parser.add_argument('--validation-dir', default='.validation', help='Directory with FE outputs')
    parser.add_argument('--output-figure', default='temperature_comparison.png', help='Output figure file')
    parser.add_argument('--align-start', action='store_true', help='Translate one or both profiles so their starts align')
    parser.add_argument('--align-target', choices=['none', 'FE', 'spectral', 'both'], default='FE',
                        help='Which dataset to translate when --align-start is given (default: FE)')
    parser.add_argument('--invert-target', choices=['none', 'FE', 'spectral', 'both'], default='none',
                        help="Which dataset to invert sign for to match axis direction")
    parser.add_argument('--invert-axes', default='',
                        help='Comma-separated list of axes to invert (e.g. "x,y"). Empty = none')
    parser.add_argument('--file-tag', default='spectral', 
                        help='Tag within filename to switch data source. e.g. "spectral" -> x_spectral_latent_heat.txt, "fine" -> x_fine_latent_heat.txt')

    args = parser.parse_args()

    print(f"Loading temperature profiles (type: {args.file_tag})...")
    profiles = load_temperature_profiles(out_dir=args.out_dir, validation_dir=args.validation_dir, file_tag=args.file_tag)

    # Parse invert axes
    invert_axes = tuple([s.strip().lower() for s in args.invert_axes.split(',') if s.strip()])

    # Decide alignment target: if --align-start was passed, use args.align_target, otherwise 'none'
    align_target = args.align_target if args.align_start else 'none'

    # Apply requested transforms (inversion + alignment)
    profiles = apply_transforms(profiles, align_target=align_target,
                                invert_target=args.invert_target, invert_axes=invert_axes)

    print("\nComputing comparison metrics...")
    metrics = compute_metrics(profiles)

    print("\nGenerating comparison plots...")
    # Capitalize tag for plotting label (e.g., "Spectral" or "Fine")
    label_sim = args.file_tag.capitalize()
    plot_comparison(profiles, output_file=args.output_figure, label_sim=label_sim)
