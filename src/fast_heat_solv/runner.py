"""
StandaloneHeatRunner – self-contained simulation runner.

This class extracts the time-stepping loop, I/O management, and telemetry
from ``SimulationWorkflow``. It is the recommended entry-point for running
fastHeatSolv in **standalone** mode (i.e. driven by a YAML config file).

Author: Théo Andrieux (@TheoADX)
Copyright: (c) 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique. All rights reserved.
"""

__author__ = "Théo Andrieux"
__copyright__ = "Copyright 2026, LMS, École Polytechnique"

import time
import logging
import os
import sys
import shutil
import datetime
import subprocess
import multiprocessing
import platform
from typing import Optional, Dict, Any

from fast_heat_solv.factories.base import SimulationFactory
from fast_heat_solv.solvers.base import HeatSolver
from fast_heat_solv.io_utils.io_base import IOManager
from fast_heat_solv.core.parameters import SimulationContext

logger = logging.getLogger(__name__)

_SEP = "=" * 70


def initialize_run_logging(config: Dict[str, Any], yaml_path: str, out_dir: str) -> None:
    """
    Prepare the run directory for traceability before the simulation starts.

    Copies the YAML config (and G-code file if present) into *out_dir*, then
    writes a human-readable ASCII header to ``logs/simulation.log``.  The
    header is written directly to the file so it sits above any timestamped
    log lines produced by the Python logging handlers.

    Must be called after the output directory tree has been created by
    ``IOManager.initialize()``.

    Parameters
    ----------
    config    : Raw YAML config dict as returned by ``load_config()``.
    yaml_path : Absolute path to the original YAML configuration file.
    out_dir   : Run output directory (``io_manager.base_dir``).
    """
    # ------------------------------------------------------------------
    # 1. Copy input files into out_dir for reproducibility
    # ------------------------------------------------------------------
    shutil.copy2(yaml_path, os.path.join(out_dir, "config.yaml"))

    gcode_src = config.get("laser", {}).get("path", {}).get("file", None)
    gcode_label = "None"
    if gcode_src:
        gcode_filename = os.path.basename(gcode_src)
        # If gcode_src is not absolute, assume it is relative to the directory of yaml_path / 'paths'
        if not os.path.isabs(gcode_src):
            gcode_src = os.path.abspath(os.path.join(os.path.dirname(yaml_path), "paths", gcode_src))
        
        if os.path.exists(gcode_src):
            shutil.copy2(gcode_src, os.path.join(out_dir, gcode_filename))
            gcode_label = gcode_filename
        else:
            logger.warning("G-code file not found for copying: %s", gcode_src)
            gcode_label = f"{gcode_filename} (NOT FOUND)"

    # ------------------------------------------------------------------
    # 2. Gather run metadata
    # ------------------------------------------------------------------
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(yaml_path),
        ).decode().strip()
    except Exception:
        git_hash = "N/A"

    python_version = sys.version.split()[0]
    os_info = f"{platform.system()} {platform.release()}"
    cpu_count = multiprocessing.cpu_count()

    try:
        if platform.system() == "Linux":
            cpu_model = subprocess.check_output(
                "cat /proc/cpuinfo | grep 'model name' | head -n 1 | awk -F: '{print $2}'",
                shell=True
            ).decode().strip()
        elif platform.system() == "Darwin":
            cpu_model = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode().strip()
        else:
            cpu_model = platform.processor() or "Unknown"
    except Exception:
        cpu_model = platform.processor() or "Unknown"

    try:
        gpu_model = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL
        ).decode().strip().split('\n')[0]
        if not gpu_model:
            gpu_model = "None / N/A"
    except Exception:
        gpu_model = "None / N/A"

    sim = config.get("simulation", {})
    dom = config.get("domain", {})
    mat = config.get("material", {})
    lsr = config.get("laser", {})
    io  = config.get("io", {})

    radius_um = lsr.get("radius", 0)
    try:
        radius_str = f"{float(radius_um) * 1e6:.1f} um"
    except (TypeError, ValueError):
        radius_str = "N/A"

    # ------------------------------------------------------------------
    # 3. Build the ASCII header
    # ------------------------------------------------------------------
    lines = [
        _SEP,
        "                   FAST HEAT SOLV - SIMULATION LOG",
        _SEP,
        "[RUN INFO]",
        f"Start Time       : {start_time}",
        f"Git Commit       : {git_hash}",
        f"Python Version   : {python_version}",
        f"OS               : {os_info}",
        f"CPU Model        : {cpu_model}",
        f"Cores Allocated  : {cpu_count} available",
        f"GPU Model        : {gpu_model}",
        "",
        "[FILES SAVED IN OUTPUT DIR]",
        "Config File      : config.yaml",
        f"Laser Path       : {gcode_label}",
        "",
        "[SIMULATION PARAMETERS]",
        f"Name             : {sim.get('name', 'N/A')}",
        f"Backend & Method : {sim.get('backend', 'N/A')} - {sim.get('method', 'N/A')}",
        f"Duration         : {sim.get('duration', 'N/A')} s",
        f"Time Step (dt)   : {sim.get('dt', 'N/A')} s",
        f"Update Interval  : {sim.get('update_interval', 'N/A')}",
        "",
        "[DOMAIN]",
        f"Size (x, y, z)   : {dom.get('size', 'N/A')} m",
        f"Mesh (nx, ny, nz): {dom.get('mesh', 'N/A')}",
        "",
        f"[MATERIAL: {mat.get('name', 'N/A')}]",
        (
            f"Temperatures (K) : T0={mat.get('T0', 'N/A')} | "
            f"T_sol={mat.get('T_solidus', 'N/A')} | "
            f"T_liq={mat.get('T_liquidus', 'N/A')} | "
            f"T_boil={mat.get('T_boil', 'N/A')}"
        ),
        (
            f"Properties       : rho={mat.get('rho', 'N/A')} | "
            f"k={mat.get('k', 'N/A')} | "
            f"Cp={mat.get('Cp', 'N/A')} | "
            f"L_f={mat.get('L_f', 'N/A')}"
        ),
        "",
        "[LASER]",
        f"Power            : {lsr.get('power_nominal', 'N/A')} W",
        f"Radius           : {radius_str}",
        f"Absorptivity     : {lsr.get('absorptivity', 'N/A')}",
        "",
        "[IO]",
        f"Save Interval    : {io.get('interval', 'N/A')} steps",
        f"Outputs          : {io.get('outputs', [])}",
        _SEP,
        "",
    ]

    # ------------------------------------------------------------------
    # 4. Write header directly to the file (bypasses logging formatter)
    # ------------------------------------------------------------------
    log_path = os.path.join(out_dir, "logs", "simulation.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info("Run header written to %s", log_path)


class StandaloneHeatRunner:
    """
    Orchestrates the full simulation lifecycle in standalone mode.

    Responsibilities
    ----------------
    * Creates solver and I/O manager via the supplied factory.
    * Runs the time-stepping loop (``while t < t_end``).
    * Delegates periodic / end-of-run output to the ``IOManager``.
    * Prints ETA / telemetry to the logger.

    Parameters
    ----------
    context : SimulationContext
        Fully populated simulation parameters.
    factory : SimulationFactory
        Abstract factory that produces backend-specific components.
    """

    def __init__(
        self,
        context: SimulationContext,
        factory: SimulationFactory,
        config: Optional[Dict[str, Any]] = None,
        yaml_path: Optional[str] = None,
    ):
        self.context = context
        self.factory = factory
        self.config = config        # raw YAML dict — used for the run header
        self.yaml_path = yaml_path  # absolute path to the original YAML file

        # Create components via factory
        self.heat_solver: HeatSolver = self.factory.create_heat_solver()
        self.io_manager: IOManager = self.factory.create_io_manager()

    # ------------------------------------------------------------------
    #  Telemetry helpers
    # ------------------------------------------------------------------
    def _init_telemetry(self) -> None:
        """
        Initialises wall-clock tracking and optional memory probe.

        Sets up the internal clock mechanisms required to track the processing 
        time per step and optionally hooks into the `psutil` package to track
        RAM consumption during the loop's execution.
        """
        self._start_wall_time = time.time()
        self._last_eta_log_time = self._start_wall_time
        self._eta_log_interval = self.context.num.update_interval
        self._total_step_time = 0.0
        self._n_steps_timed = 0
        try:
            import psutil
            self._psutil_proc = psutil.Process()
        except Exception:
            self._psutil_proc = None

    def _log_progress(
        self,
        t: float,
        step: int,
        step_elapsed: float,
        metrics: Optional[Dict[str, Any]],
    ) -> None:
        """
        Logs ETA, memory usage, and solver metrics at the configured interval.

        Parameters
        ----------
        t : float
            Current simulation physical time in seconds.
        step : int
            Current time step iteration index.
        step_elapsed : float
            Wall-time duration of the previous simulation step in seconds.
        metrics : dict of str to any, optional
            A dictionary containing any internal solving metrics mapped by string ID.
        """
        self._total_step_time += step_elapsed
        self._n_steps_timed += 1

        t_end = self.context.num.t_end
        now = time.time()
        if (now - self._last_eta_log_time < self._eta_log_interval) and (t < t_end):
            return  # not yet time to log

        elapsed = now - self._start_wall_time
        frac_done = min(t / t_end, 1.0) if t_end > 0 else 0.0

        if frac_done > 0:
            est_remaining = (elapsed / frac_done) - elapsed
            
            hours, remainder = divmod(est_remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours >= 24:
                days, hours = divmod(hours, 24)
                eta_str = f"{int(days)}d {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            else:
                eta_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                
            mem_mb = None
            if self._psutil_proc is not None:
                try:
                    mem_mb = self._psutil_proc.memory_info().rss / (1024.0 ** 2)
                except Exception:
                    pass
            avg_step = self._total_step_time / self._n_steps_timed
            logger.info(
                f"[ETA] Step {step} | t={t:.6e}s | Elapsed: {elapsed:.1f}s | "
                f"Remaining: {eta_str} | step_time={step_elapsed:.3f}s | "
                f"avg_step={avg_step:.3f}s | mem_mb={mem_mb if mem_mb is not None else 'NA'}"
            )
        else:
            logger.info(
                f"[ETA] Step {step} | t={t:.6e}s | Elapsed: {elapsed:.1f}s | Remaining: unknown"
            )

        # Format metrics
        parts = []
        for k, v in (metrics or {}).items():
            try:
                if isinstance(v, int):
                    parts.append(f"{k}={v}")
                else:
                    parts.append(f"{k}={float(v):.3f}")
            except Exception:
                parts.append(f"{k}={v}")
        logger.info(f"[METRICS] {', '.join(parts)}")
        self._last_eta_log_time = now

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Executes the main simulation loop.

        It initializes the logging mechanism and file outputs, builds the initial 
        state via the solver interface, and proceeds to perform the chronological
        advancement loop until `t_end` is reached. Final states are serialized 
        before cleanup.
        """
        logger.info("Initializing standalone simulation runner...")

        # 1. Initialize IO System
        self.io_manager.initialize(self.context)

        # --- LOG FILE SETUP ---
        run_dir = self.io_manager.base_dir
        if run_dir is not None:
            # Write the ASCII header and copy input files first so they appear
            # at the top of the log file, before any timestamped entries.
            if self.config is not None and self.yaml_path is not None:
                initialize_run_logging(self.config, self.yaml_path, run_dir)

            log_file_path = os.path.join(run_dir, "logs", "simulation.log")
            # Remove previous file handlers to avoid duplicates
            for h in logger.handlers[:]:
                if isinstance(h, logging.FileHandler):
                    logger.removeHandler(h)
            file_handler = logging.FileHandler(log_file_path, mode="a")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(file_handler)
            logger.info("File logging enabled: %s", log_file_path)

        # 2. Initialize Solver (pass context for dependency injection)
        state = self.heat_solver.initialize(self.context)

        # 3. Time Loop
        dt = self.context.num.dt
        dt_nominal = self.context.num.dt_nominal
        t_end = self.context.num.t_end
        n_steps = self.context.num.n_steps
        laser_path = getattr(self.context, "laser_path")

        dt_correction = dt - dt_nominal
        if abs(dt_correction) > 1e-6 * dt_nominal:
            logger.info(
                f"dt correction: dt_nominal={dt_nominal:.6e} s -> dt={dt:.6e} s "
                f"(delta={dt_correction:+.3e} s, {dt_correction/dt_nominal*100:+.4f}%) "
                f"so that n_steps={n_steps} * dt = t_end={t_end:.6e} s exactly"
            )
        logger.info(f"Starting time loop: 0 -> {t_end:.4e} s (dt={dt:.4e}, n_steps={n_steps})")
        self._init_telemetry()

        for step in range(n_steps + 1):
            t = step * dt  # exact, no accumulation

            # A. Periodic output (IOManager decides internally)
            self.io_manager.process_step(t, step, state, laser_path)

            # B. Evolve state (pure physics – no I/O inside step)
            step_start = time.time()
            state, metrics = self.heat_solver.step(t, dt)
            step_elapsed = time.time() - step_start

            # D. Telemetry (internally rate-limited)
            self._log_progress(t + dt, step + 1, step_elapsed, metrics)

        # End-of-simulation outputs — state is at exactly t_end
        self.io_manager.process_end(t_end, n_steps, state, laser_path)

        # 4. Finalize
        self.heat_solver.finalize()
        self.io_manager.finalize()

        # Optionally log profiler diagnostics path if present
        try:
            run_dir = getattr(self.io_manager, "base_dir", None)
            if run_dir:
                prof_path = os.path.join(run_dir, "diagnostics", "profiler.txt")
                if os.path.exists(prof_path):
                    logger.info(f"Profiler output written to: {prof_path}")
        except Exception:
            pass

        logger.info("Simulation completed successfully.")
