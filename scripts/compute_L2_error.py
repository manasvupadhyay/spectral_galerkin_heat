#!/usr/bin/env python3
"""CLI entry point for L2 error comparison between two XDMF solution files.

Usage
-----
    python scripts/compute_L2_error.py file_A.xdmf file_B.xdmf [options]

    # Compare spectral output vs FE validation:
    python scripts/compute_L2_error.py \\
        ../out/sim/fields/field_step000200.xmf \\
        validation.xdmf \\
        --attr-a temperature --attr-b Temperature

    # Fast convergence study (skip error output):
    python scripts/compute_L2_error.py spectral.xmf fe.xdmf --no-error-output
"""

import argparse
import logging
import sys
from pathlib import Path

from fast_heat_solv.io_utils.compute_L2_error import compare

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute L2 norm error between two XDMF solution files."
    )
    parser.add_argument("file_a", help="Path to XDMF file A (reference)")
    parser.add_argument("file_b", help="Path to XDMF file B")
    parser.add_argument(
        "--attr-a",
        default=None,
        help="Attribute name in file A (default: first found)",
    )
    parser.add_argument(
        "--attr-b",
        default=None,
        help="Attribute name in file B (default: first found)",
    )
    parser.add_argument(
        "--output",
        default="error",
        help="Base name for error output files (default: 'error')",
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help="Grid resolution for two-unstructured case, e.g. '128,128,128'",
    )
    parser.add_argument(
        "--no-error-output",
        action="store_true",
        help="Skip writing error XDMF/H5 files (faster for convergence studies)",
    )

    args = parser.parse_args()

    res = None
    if args.resolution:
        parts = [int(x) for x in args.resolution.split(",")]
        res = tuple(parts[:3])

    norms = compare(
        path_a=Path(args.file_a),
        path_b=Path(args.file_b),
        attr_a=args.attr_a,
        attr_b=args.attr_b,
        output_base=Path(args.output),
        resolution=res,
        write_error=not args.no_error_output,
    )

    if norms["pct_valid"] < 50.0:
        sys.exit(2)


if __name__ == "__main__":
    main()
