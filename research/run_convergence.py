#!/usr/bin/env python3
import os
import sys
import glob
import shutil
import subprocess
import csv

# We assume standard yaml package is available (often used via PyYAML)
import yaml

# Add project root to sys.path so the tests package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.compute_L2_error import compare

import math


def log_spaced_ints(start, end, num):
    """Return a list of `num` log-spaced integers from start to end (inclusive).

    Ensures strictly increasing values and forces first==start, last==end.
    """
    if start <= 0 or end <= 0:
        raise ValueError("start and end must be positive for log spacing")
    if num < 2:
        return [int(round(start))]

    ratio = (end / start) ** (1.0 / (num - 1))
    vals = [start * (ratio ** i) for i in range(num)]
    ints = [int(round(v)) for v in vals]
    # enforce strictly increasing sequence
    out = []
    for v in ints:
        if not out:
            out.append(max(1, v))
        else:
            if v > out[-1]:
                out.append(v)
            else:
                out.append(out[-1] + 1)
    out[0] = int(round(start))
    out[-1] = int(round(end))
    return out

def get_latest_dir(base_dir="out"):
    """Finds the most recently created directory in the `out` folder."""
    if not os.path.exists(base_dir):
        return None
    dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def run_simulation_and_get_error(param_name, param_val, nx, ny, nz, template_yaml_path, out_csv_writer):
    print(f"\n[{param_name.upper()} = {param_val}] Running convergence point... Mesh: [{nx}, {ny}, {nz}]")
    
    # 1. Read base yaml
    with open(template_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. Modify mesh dimensions
    config['domain']['mesh'] = [nx, ny, nz]
    
    # 3. Create a temporary yaml config
    tmp_yaml = "simulations/config/tmp_convergence.yaml"
    with open(tmp_yaml, 'w') as f:
        yaml.dump(config, f)
        
    # 4. Run the simulation
    # Using check=True will raise an error if the simulation crashes
    subprocess.run(["python", "simulations/main.py", tmp_yaml], check=True)
    
    # 5. Locate the newly created simulation directory and output xmf
    latest_out = get_latest_dir("out")
    if not latest_out:
        raise RuntimeError("No output directory found after simulation.")
        
    fields_dir = os.path.join(latest_out, "fields")
    xmf_files = glob.glob(os.path.join(fields_dir, "field_step*.xmf"))
    
    if not xmf_files:
        raise RuntimeError(f"No xmf files found in {fields_dir}")
        
    # Sort them and take the last step
    xmf_files.sort()
    latest_xmf = xmf_files[-1]
    
    print(f"  -> Found target field: {latest_xmf}")
    
    # 6. Compute L2 Error against eagar_tsai.xmf
    eagar_tsai_path = "validation_results/validation.xdmf"
    if not os.path.exists(eagar_tsai_path):
        raise FileNotFoundError(f"Reference file {eagar_tsai_path} not found. Please generate it first.")
        
    print("  -> Computing L2 error differences...")
    # Hide stdout while running evaluate if you prefer, passing output arguments to /dev/null
    norms = compare(
        path_a=eagar_tsai_path,
        path_b=latest_xmf,
        attr_a="temperature",
        attr_b="temperature"
    )
    
    # 7. Write to CSV
    out_csv_writer.writerow({
        "Tested_Variable": param_name,
        "Variable_Value": param_val,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "L2_abs": norms["L2_abs"],
        "L2_rel": norms["L2_rel"],
        "Linf": norms["Linf"]
    })
    
    # 8. Clean up simulation files to save space
    print(f"  -> Removing simulation folder {latest_out} to free up space...")
    shutil.rmtree(latest_out)
    print("  -> Cleanup done.\n")

def main():
    base_yaml = "simulations/config/standard_test.yaml"
    csv_file_path = "research/convergence_results_FE.csv"
    
    # Base configuration
    base_nx, base_ny, base_nz = 600, 256, 1700 # Starting mesh size 
    
    # User-requested ranges
    # Generate 10 log-spaced integer mesh sizes (inclusive endpoints)
    nx_range = log_spaced_ints(10, 800, 10)
    ny_range = log_spaced_ints(10, 300, 10)
    nz_range = log_spaced_ints(10, 2000, 10)
    
    print("Starting convergence tests with the following mesh sizes:")
    print(f"  nx: {nx_range}")
    print(f"  ny: {ny_range}")
    print(f"  nz: {nz_range}")
    
    # Write header
    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = ["Tested_Variable", "Variable_Value", "nx", "ny", "nz", "L2_abs", "L2_rel", "Linf"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 1. Vary nx
        for nx in nx_range:
            run_simulation_and_get_error("nx", nx, nx, base_ny, base_nz, base_yaml, writer)
            csvfile.flush() # flush continuously to store results even if killed early
            
        # 2. Vary ny
        for ny in ny_range:
            run_simulation_and_get_error("ny", ny, base_nx, ny, base_nz, base_yaml, writer)
            csvfile.flush()

        # 3. Vary nz
        for nz in nz_range:
            run_simulation_and_get_error("nz", nz, base_nx, base_ny, nz, base_yaml, writer)
            csvfile.flush()
            
    # Clean up temp configuration
    if os.path.exists("config/tmp_convergence.yaml"):
        os.remove("config/tmp_convergence.yaml")
        
    print(f"Done! All convergence results logged to '{csv_file_path}'.")

if __name__ == "__main__":
    main()
