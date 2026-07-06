"""Tolerance study: effect of Picard convergence tolerance on the temperature
field and iteration count.

Sweeps N tolerance levels from tol_max down to tol_min (default 1e-2 → 1e-6),
runs n_steps time steps for each, then saves:
  - temperature_lines.csv  : T(x) along top surface at y=Ly/2 for each tolerance
  - iterations.csv         : per-step and average Picard iteration count per tolerance
  - temperature_lines.png  : superimposed T(x) curves
  - avg_iterations.png     : avg iterations vs tolerance (semilog-x)

Re-plot from saved CSVs:
  python simulations/benchmark_tol_study.py \\
      --from-csv out/tol_study_YYYYMMDD-HHMMSS \\
      --output-dir out/replot
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulations.main import load_config
from fast_heat_solv.solvers import build_solver
from fast_heat_solv.core.laser import LaserPath, LaserState
from fast_heat_solv.core.parameters import SimulationContext
import fast_heat_solv.physics.spectral_cpu_kernels as kernels

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared plot style (matches the project reference)
# ---------------------------------------------------------------------------
_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 1.5,
    "lines.markersize": 7,
    "xtick.direction": "out",
    "ytick.direction": "out",
}
_COLORS = ["blue", "red", "green", "orange", "purple",
           "brown", "deeppink", "gray", "cyan", "olive"]
_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h", "X", "P"]


# ---------------------------------------------------------------------------
# Laser path
# ---------------------------------------------------------------------------
class LaserPathFromX0(LaserPath):
    """Laser moving along +x, starting at a configurable x0 position."""

    def __init__(self, context: SimulationContext, speed: float, x0: float):
        self.context = context
        self.speed = np.float32(speed)
        self.x0 = np.float32(x0)
        self.y0 = np.float32(context.geom.size.y * 0.5)

    def get_state(self, time: float, dt: float) -> LaserState:
        laser = self.context.laser
        x = self.x0 + self.speed * np.float32(time)
        return LaserState(
            x=np.float32(x),
            y=np.float32(self.y0),
            power=np.float32(laser.power),
            is_on=True,
            v=(np.float32(self.speed), np.float32(0.0)),
        )


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------
def build_context(
    config_path: str,
    n_steps: int,
    laser_speed: float,
    laser_x0: float,
) -> SimulationContext:
    config = load_config(config_path)
    dt = float(config.get("simulation", {}).get("dt", 6e-6))

    config.setdefault("domain", {})
    config["domain"]["size"] = [0.5e-3, 0.5e-3, 0.05e-3]
    config["domain"]["mesh"] = [128, 128, 200]

    config.setdefault("simulation", {})
    config["simulation"]["dt"] = dt
    config["simulation"]["duration"] = dt * float(n_steps)

    config.setdefault("laser", {})
    config["laser"]["power_nominal"] = float(config["laser"].get("power_nominal", 200.0))

    config_dir = os.path.dirname(os.path.abspath(config_path))
    context = SimulationContext.from_dict(config, config_dir=config_dir)
    context.backend = "cpu"
    context.method = "spectral"
    context.laser_path = LaserPathFromX0(context, speed=laser_speed, x0=laser_x0)
    return context


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------
def run_tol_case(
    config_path: str,
    tol: float,
    omega: float,
    max_iter: int,
    n_steps: int,
    laser_speed: float,
    laser_x0: float,
) -> tuple:
    """
    Returns
    -------
    x_coords       : 1-D array, x cell-centre positions (m)
    T_line         : 1-D array, T at top surface y=Ly/2 after last step (K)
    avg_iter       : float, average Picard iterations per step
    iter_counts    : list of int, per-step iteration count
    laser_x_final  : float, laser x position at end of simulation (m)
    """
    context = build_context(config_path, n_steps=n_steps,
                            laser_speed=laser_speed, laser_x0=laser_x0)
    solver = build_solver(context)
    solver.initialize(context)

    solver.mixing_omega = np.float32(omega)
    solver.convergence_tol = np.float32(tol)
    solver.max_picard_iter = int(max_iter)
    solver.track_picard_history = False

    dt = float(context.num.dt)
    t = 0.0
    iter_counts = []

    for step_idx in range(1, n_steps + 1):
        state, metrics = solver.step(t, dt)
        n_iter = int(metrics.get("n_evap_iter", 0))
        T_max = float(metrics.get("T_surface_max", float("nan")))
        iter_counts.append(n_iter)
        logger.info(
            "tol=%.1e | step=%02d/%d | n_iter=%3d | T_surface_max=%8.1f K",
            tol, step_idx, n_steps, n_iter, T_max,
        )
        t += dt

    # Top-surface temperature at y = Ly/2, along x (full line)
    T_surface = kernels.reconstruct_surface_temperature(solver.state.a, solver.state)
    # T_surface shape: (ny, nx)
    ny = T_surface.shape[0]
    T_line = np.array(T_surface[ny // 2, :], dtype=float)

    x_coords = np.array(solver.state.grid.x, dtype=float)  # shape (nx,)
    laser_x_final = laser_x0 + laser_speed * n_steps * dt

    avg_iter = float(np.mean(iter_counts))
    return x_coords, T_line, avg_iter, iter_counts, laser_x_final


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
def write_temperature_csv(path: Path, tols, x_coords, T_lines) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_m"] + [f"T_tol_{t:.2e}" for t in tols])
        for ix in range(len(x_coords)):
            writer.writerow(
                [f"{x_coords[ix]:.12e}"] + [f"{T[ix]:.6f}" for T in T_lines]
            )


def write_iterations_csv(path: Path, tols, avg_iters, all_iter_counts) -> None:
    n_steps = len(all_iter_counts[0]) if all_iter_counts else 0
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tol", "avg_iter_per_step"] + [f"step_{i+1}" for i in range(n_steps)])
        for tol, avg, counts in zip(tols, avg_iters, all_iter_counts):
            writer.writerow([f"{tol:.6e}", f"{avg:.4f}"] + [str(c) for c in counts])


def load_temperature_csv(path: Path) -> tuple:
    """Returns (tols, x_coords, T_lines)."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        tol_cols = [c for c in fieldnames if c.startswith("T_tol_")]
        tols = [float(c.replace("T_tol_", "")) for c in tol_cols]
        x_coords, T_lines_T = [], [[] for _ in tols]
        for row in reader:
            x_coords.append(float(row["x_m"]))
            for k, col in enumerate(tol_cols):
                T_lines_T[k].append(float(row[col]))
    return np.array(tols), np.array(x_coords), [np.array(t) for t in T_lines_T]


def load_iterations_csv(path: Path) -> tuple:
    """Returns (tols, avg_iters)."""
    tols, avg_iters = [], []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tols.append(float(row["tol"]))
            avg_iters.append(float(row["avg_iter_per_step"]))
    return np.array(tols), np.array(avg_iters)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_temperature(
    path: Path, x_coords, tols, T_lines, laser_x_final: float
) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update(_RCPARAMS)

    fig, ax = plt.subplots(figsize=(10, 5))
    every = max(1, len(x_coords) // 12)

    for i, (tol, T) in enumerate(zip(tols, T_lines)):
        ax.plot(
            x_coords * 1e6, T,
            color=_COLORS[i % len(_COLORS)],
            linestyle=_LINESTYLES[(i // len(_COLORS)) % len(_LINESTYLES)],
            marker=_MARKERS[i % len(_MARKERS)],
            markevery=every,
            linewidth=1.0, markersize=5,
            label=f"tol={tol:.0e}",
        )

    ax.axvline(laser_x_final * 1e6, color="0.4", linestyle=":",
               linewidth=1.0, label="laser pos. (final)")
    ax.set_xlabel(r"$x$ ($\mu$m)")
    ax.set_ylabel(r"$T$ (K)")
    ax.set_title("Temperature along laser axis – top surface, $y = L_y/2$ (final step)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="both", direction="out")
    ax.legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info("plot: %s", path)


def plot_iterations(path: Path, tols, avg_iters) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update(_RCPARAMS)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogx(
        tols, avg_iters,
        color="blue", marker="o", linestyle="-",
        linewidth=1.0, markersize=6,
    )
    ax.invert_xaxis()  # tighter (smaller) tolerance on the right

    ax.set_xlabel("Tolerance")
    ax.set_ylabel("Avg. Picard iterations per step")
    ax.set_title("Average Picard iterations per time step vs. tolerance")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="both", direction="out")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info("plot: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Picard tolerance study: temperature field and iteration count vs. tolerance."
    )
    parser.add_argument(
        "--config",
        default="simulations/config/standard_test.yaml",
        help="Path to the YAML config.",
    )
    parser.add_argument("--tol-min", type=float, default=1e-6,
                        help="Tightest tolerance (default 1e-6).")
    parser.add_argument("--tol-max", type=float, default=1e-2,
                        help="Loosest tolerance (default 1e-2).")
    parser.add_argument("--n-tols", type=int, default=10,
                        help="Number of tolerance levels (default 10).")
    parser.add_argument("--num-steps", type=int, default=40,
                        help="Time steps per run (default 40).")
    parser.add_argument("--omega", type=float, default=0.2,
                        help="Picard relaxation parameter omega (default 0.2).")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="Max Picard iterations per step (default 500).")
    parser.add_argument("--laser-speed", type=float, default=0.8,
                        help="Laser speed in m/s along +x (default 0.8).")
    parser.add_argument("--laser-x0", type=float, default=0.1e-3,
                        help="Laser start position in m (default 0.1e-3 = 0.1 mm).")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-generated if omitted).")
    parser.add_argument(
        "--from-csv",
        default=None,
        metavar="DIR",
        help="Re-plot from a previously saved output directory (skips simulation).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # -----------------------------------------------------------------------
    # Re-plot path
    # -----------------------------------------------------------------------
    if args.from_csv:
        src_dir = Path(args.from_csv)
        output_dir = Path(args.output_dir) if args.output_dir else src_dir / "replot"
        output_dir.mkdir(parents=True, exist_ok=True)

        tols_t, x_coords, T_lines = load_temperature_csv(src_dir / "temperature_lines.csv")
        tols_i, avg_iters = load_iterations_csv(src_dir / "iterations.csv")

        # Use tols_t as authoritative order; tols_i may differ if partially run
        plot_temperature(output_dir / "temperature_lines.png",
                         x_coords, tols_t, T_lines, laser_x_final=float("nan"))
        plot_iterations(output_dir / "avg_iterations.png", tols_i, avg_iters)
        logger.info("output_dir: %s", output_dir)
        return 0

    # -----------------------------------------------------------------------
    # Simulation sweep
    # -----------------------------------------------------------------------
    tols = np.logspace(np.log10(args.tol_max), np.log10(args.tol_min), args.n_tols)

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path("out") / f"tol_study_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Tolerance study ===")
    logger.info("config        : %s", args.config)
    logger.info("domain        : (0.5, 0.5, 0.05) mm  mesh: (128, 128, 200)")
    logger.info("laser_speed   : %.3f m/s   laser_x0: %.4f mm",
                args.laser_speed, args.laser_x0 * 1e3)
    logger.info("n_steps       : %d   omega: %.3f   max_iter: %d",
                args.num_steps, args.omega, args.max_iter)
    logger.info("tolerances    : %s", [f"{t:.1e}" for t in tols])

    T_lines: list = []
    avg_iters: list = []
    all_iter_counts: list = []
    x_coords = None
    laser_x_final = float("nan")

    for tol in tols:
        logger.info("--- tol=%.1e ---", tol)
        x_c, T_line, avg_iter, iter_counts, lxf = run_tol_case(
            args.config,
            tol=tol,
            omega=args.omega,
            max_iter=args.max_iter,
            n_steps=args.num_steps,
            laser_speed=args.laser_speed,
            laser_x0=args.laser_x0,
        )
        if x_coords is None:
            x_coords = x_c
            laser_x_final = lxf
        T_lines.append(T_line)
        avg_iters.append(avg_iter)
        all_iter_counts.append(iter_counts)
        logger.info("tol=%.1e done | avg_iter=%.2f", tol, avg_iter)

    # -----------------------------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------------------------
    temp_csv = output_dir / "temperature_lines.csv"
    iter_csv = output_dir / "iterations.csv"
    write_temperature_csv(temp_csv, tols, x_coords, T_lines)
    write_iterations_csv(iter_csv, tols, avg_iters, all_iter_counts)
    logger.info("csv: %s", temp_csv)
    logger.info("csv: %s", iter_csv)

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    plot_temperature(output_dir / "temperature_lines.png",
                     x_coords, tols, T_lines, laser_x_final)
    plot_iterations(output_dir / "avg_iterations.png", tols, avg_iters)

    logger.info("output_dir: %s", output_dir)
    return 0


# Re-plot from a saved run:
#   python simulations/benchmark_tol_study.py \
#       --from-csv out/tol_study_YYYYMMDD-HHMMSS \
#       --output-dir out/replot

if __name__ == "__main__":
    raise SystemExit(main())
