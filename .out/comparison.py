"""
Comparison of spectral method results with finite element results.

This script loads temperature profiles from .out (spectral) and .validation (FE)
directories and compares them.

python comparison.py --align-start --align-target FE --invert-target FE --invert-axes z 
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def load_temperature_profiles(out_dir=".out", validation_dir=".validation"):
    """Load temperature profiles from spectral, rosenthal and FE methods.
    
    Returns:
        dict: Dictionary containing loaded profiles with keys:
              'x_spectral', 'y_spectral', 'z_spectral',
              'x_rosenthal', 'y_rosenthal', 'z_rosenthal',
              'x_FE', 'y_FE', 'z_FE'
              Each value is a tuple (coords, temperatures)
    """
    profiles = {}
    
    # Load spectral method results
    for direction in ['x', 'y', 'z']:
        filepath = os.path.join(out_dir, f"{direction}_spectral_latent_heat.txt")
        if os.path.exists(filepath):
            data = np.loadtxt(filepath)
            profiles[f'{direction}_spectral'] = (data[:, 0], data[:, 1])
            print(f"Loaded {filepath}: {len(data)} points")
        else:
            print(f"Warning: {filepath} not found")
            profiles[f'{direction}_spectral'] = (np.array([]), np.array([]))

    # Load Rosenthal results (user-provided files in .out)
    for direction in ['x', 'y', 'z']:
        filepath = os.path.join(out_dir, f"{direction}_rosenthal_latent_heat.txt")
        if os.path.exists(filepath):
            data = np.loadtxt(filepath)
            profiles[f'{direction}_rosenthal'] = (data[:, 0], data[:, 1])
            print(f"Loaded {filepath}: {len(data)} points")
        else:
            print(f"Warning: {filepath} not found")
            profiles[f'{direction}_rosenthal'] = (np.array([]), np.array([]))
    
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


def plot_comparison(profiles, output_file="temperature_comparison.png"):
    """Create comparison plots for all three directions.
    
    Args:
        profiles: Dictionary returned by load_temperature_profiles
        output_file: Path to save the comparison figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    directions = ['x', 'y', 'z']
    labels = ['x (m)', 'y (m)', 'z (m)']
    
    for ax, direction, label in zip(axes, directions, labels):
        # Plot spectral method
        coords_spec, T_spec = profiles.get(f'{direction}_spectral', (np.array([]), np.array([])))
        if len(coords_spec) > 0:
            ax.plot(coords_spec * 1e3, T_spec, 'b-', label='Spectral', linewidth=2)
        
        # Plot Rosenthal method
        coords_ros, T_ros = profiles.get(f'{direction}_rosenthal', (np.array([]), np.array([])))
        if len(coords_ros) > 0:
            ax.plot(coords_ros * 1e3, T_ros, 'g-.', label='Rosenthal', linewidth=2)
        
        # Plot FE method
        coords_FE, T_FE = profiles.get(f'{direction}_FE', (np.array([]), np.array([])))
        if len(coords_FE) > 0:
            ax.plot(coords_FE * 1e3, T_FE, 'r--', label='Finite Element', linewidth=2)
        
        ax.set_xlabel(f'{label.split()[0]} (mm)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title(f'Temperature Profile along {direction.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nComparison plot saved to: {output_file}")
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
        ros_key = f'{direction}_rosenthal'

        coords_spec, T_spec = profiles.get(spec_key, (np.array([]), np.array([])))
        coords_FE, T_FE = profiles.get(fe_key, (np.array([]), np.array([])))
        coords_ros, T_ros = profiles.get(ros_key, (np.array([]), np.array([])))

        # Apply inversion if requested (flip sign). We then sort so coords are ascending.
        if direction in invert_axes and invert_target in ('spectral', 'both'):
            coords_spec = -coords_spec
        if direction in invert_axes and invert_target in ('FE', 'both'):
            coords_FE = -coords_FE
        if direction in invert_axes and invert_target in ('both', 'rosenthal'):
            # support explicit 'rosenthal' is not in choices, keep safe no-op unless user uses 'both'
            coords_ros = -coords_ros

        coords_spec, T_spec = sort_pair(coords_spec, T_spec)
        coords_FE, T_FE = sort_pair(coords_FE, T_FE)
        coords_ros, T_ros = sort_pair(coords_ros, T_ros)

        # Align starts by translating only the requested target so its minimum equals the other's
        if align_target != 'none' and len(coords_spec) > 0 and len(coords_FE) > 0:
            if align_target == 'FE':
                # shift FE so its min matches spectral min
                coords_FE = coords_FE - coords_FE.min() + coords_spec.min()
                coords_ros = coords_ros - coords_ros.min() + coords_spec.min() if len(coords_ros) > 0 else coords_ros
            elif align_target == 'spectral':
                # shift spectral so its min matches FE min
                coords_spec = coords_spec - coords_spec.min() + coords_FE.min()
                coords_ros = coords_ros - coords_ros.min() + coords_FE.min() if len(coords_ros) > 0 else coords_ros
            elif align_target == 'both':
                # translate all so global min is zero
                start_vals = [a.min() for a in (coords_spec, coords_FE, coords_ros) if len(a) > 0]
                if start_vals:
                    start = min(start_vals)
                    if len(coords_spec) > 0:
                        coords_spec = coords_spec - start
                    if len(coords_FE) > 0:
                        coords_FE = coords_FE - start
                    if len(coords_ros) > 0:
                        coords_ros = coords_ros - start

        profiles[spec_key] = (coords_spec, T_spec)
        profiles[fe_key] = (coords_FE, T_FE)
        profiles[ros_key] = (coords_ros, T_ros)

    return profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare spectral, Rosenthal and FE temperature profiles.")
    parser.add_argument('--out-dir', default='.out', help='Directory with spectral and rosenthal outputs')
    parser.add_argument('--validation-dir', default='.validation', help='Directory with FE outputs')
    parser.add_argument('--output-figure', default='temperature_comparison.png', help='Output figure file')
    parser.add_argument('--align-start', action='store_true', help='Translate one or both profiles so their starts align')
    parser.add_argument('--align-target', choices=['none', 'FE', 'spectral', 'both'], default='FE',
                        help='Which dataset to translate when --align-start is given (default: FE)')
    parser.add_argument('--invert-target', choices=['none', 'FE', 'spectral', 'both'], default='none',
                        help="Which dataset to invert sign for to match axis direction")
    parser.add_argument('--invert-axes', default='',
                        help='Comma-separated list of axes to invert (e.g. "x,y"). Empty = none')

    args = parser.parse_args()

    print("Loading temperature profiles...")
    profiles = load_temperature_profiles(out_dir=args.out_dir, validation_dir=args.validation_dir)

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
    plot_comparison(profiles, output_file=args.output_figure)
