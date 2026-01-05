import argparse
import os
from types import SimpleNamespace

import h5py
import fastHeatSolv.helpers as hp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 1D temperature profiles from an HDF5 volume saved by the "
            "spectral solver (e.g. out/T_volume_final.h5)."
        )
    )
    parser.add_argument(
        "volume",
        nargs="?",
        default=os.path.join("out", "T_volume_final.h5"),
        help="Path to the HDF5 file containing the temperature volume (default: out/T_volume_final.h5)",
    )
    parser.add_argument(
        "--center",
        choices=["laser", "hotspot"],
        default="laser",
        help="Centering strategy for the profiles (default: laser)",
    )
    parser.add_argument(
        "--laser-x",
        type=float,
        default=None,
        help="Laser x-position in meters (required if center=laser and not stored elsewhere)",
    )
    parser.add_argument(
        "--laser-y",
        type=float,
        default=None,
        help="Laser y-position in meters (required if center=laser and not stored elsewhere)",
    )
    parser.add_argument(
        "--output-dir",
        default="out",
        help="Directory where the profile text files will be written (default: out)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.center == "laser":
        if args.laser_x is None or args.laser_y is None:
            raise SystemExit(
                "laser_x and laser_y must be provided when center='laser'. "
                "Either pass --laser-x/--laser-y or use --center hotspot."
            )
        laser_obj = SimpleNamespace(x=args.laser_x, y=args.laser_y)
    else:
        laser_obj = None

    with h5py.File(args.volume, "r") as f:
        x = f["X"][:]
        y = f["Y"][:]
        z = f["Z"][:]
        temperature = f["Temperature"][:]

    hp.save_temp_profiles_fine(
        laser=laser_obj,
        center=args.center,
        volume=temperature,
        grid_coords=(x, y, z),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
