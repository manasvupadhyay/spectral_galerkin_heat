#!/usr/bin/env python3
"""
test_images_correction.py
=========================
Compare the spectral solver (well-converged at [300, 150, 600] modes) against:

  1. Pure Eagar-Tsai  (semi-infinite, no boundary correction)
  2. Corrected Eagar-Tsai  (method of images for x=0, y=0, y=Ly)

Expected result: error_corrected < error_pure  because the spectral solver
enforces zero-flux boundaries, which the corrected Eagar-Tsai accounts for.
"""

import os
import sys
import glob
import shutil
import subprocess
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validation_results.compute_L2_error import compare

# ── Paths ──
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CONFIG_YAML   = os.path.join(BASE_DIR, "config", "standard_test.yaml")
TMP_YAML      = os.path.join(BASE_DIR, "config", "tmp_images_test.yaml")
ET_UNCORR_XMF = os.path.join(BASE_DIR, "validation_results", "eagar_tsai.xmf")
ET_CORR_XMF   = os.path.join(BASE_DIR, "validation_results", "eagar_tsai_corrected.xmf")


def get_latest_dir(base_dir="out"):
    if not os.path.exists(base_dir):
        return None
    dirs = [os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))]
    return max(dirs, key=os.path.getmtime) if dirs else None


def main():
    # ────────────────────────────────────────────────────────────────────
    #  Step 1 – Generate both Eagar-Tsai reference fields
    # ────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Step 1: Generating Eagar-Tsai reference fields")
    print("=" * 70)
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "utils", "eagar_tsai.py")],
                   check=True, cwd=BASE_DIR)
    assert os.path.exists(ET_UNCORR_XMF), f"Missing {ET_UNCORR_XMF}"
    assert os.path.exists(ET_CORR_XMF),   f"Missing {ET_CORR_XMF}"

    # ────────────────────────────────────────────────────────────────────
    #  Step 2 – Run spectral solver at [300, 150, 600]
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Step 2: Running spectral solver at [300, 150, 600] modes")
    print("=" * 70)

    with open(CONFIG_YAML, "r") as f:
        config = yaml.safe_load(f)
    config["domain"]["mesh"] = [300, 150, 600]
    with open(TMP_YAML, "w") as f:
        yaml.dump(config, f)

    subprocess.run([sys.executable, "main.py", TMP_YAML], check=True, cwd=BASE_DIR)

    latest_out = get_latest_dir(os.path.join(BASE_DIR, "out"))
    assert latest_out, "No output directory found after spectral simulation."
    fields_dir = os.path.join(latest_out, "fields")
    xmf_files  = sorted(glob.glob(os.path.join(fields_dir, "field_step*.xmf")))
    assert xmf_files, f"No xmf files found in {fields_dir}"
    spectral_xmf = xmf_files[-1]
    print(f"  -> Spectral output: {spectral_xmf}")

    # ────────────────────────────────────────────────────────────────────
    #  Step 3 – Compute L2 errors
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Step 3: Computing L2 errors")
    print("=" * 70)

    norms_pure = compare(
        path_a=ET_UNCORR_XMF,
        path_b=spectral_xmf,
        attr_a="temperature",
        attr_b="temperature",
    )
    norms_corr = compare(
        path_a=ET_CORR_XMF,
        path_b=spectral_xmf,
        attr_a="temperature",
        attr_b="temperature",
    )

    print(f"\n  Pure Eagar-Tsai vs Spectral:")
    print(f"    L2_abs = {norms_pure['L2_abs']:.6e}")
    print(f"    L2_rel = {norms_pure['L2_rel']:.6e}")
    print(f"    Linf   = {norms_pure['Linf']:.6e}")

    print(f"\n  Corrected Eagar-Tsai (images) vs Spectral:")
    print(f"    L2_abs = {norms_corr['L2_abs']:.6e}")
    print(f"    L2_rel = {norms_corr['L2_rel']:.6e}")
    print(f"    Linf   = {norms_corr['Linf']:.6e}")

    # ────────────────────────────────────────────────────────────────────
    #  Step 4 – Assert improvement
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Step 4: Verification")
    print("=" * 70)
    ok = True
    for metric in ("L2_abs", "L2_rel", "Linf"):
        improved = norms_corr[metric] < norms_pure[metric]
        status   = "PASS" if improved else "FAIL"
        print(f"  [{status}] {metric}: corrected ({norms_corr[metric]:.4e}) "
              f"< pure ({norms_pure[metric]:.4e})")
        if not improved:
            ok = False

    # ── Cleanup ──
    print(f"\n  Cleaning up {latest_out} ...")
    shutil.rmtree(latest_out)
    if os.path.exists(TMP_YAML):
        os.remove(TMP_YAML)

    if ok:
        print("\n  ALL TESTS PASSED – corrected Eagar-Tsai is closer to the spectral solution.\n")
    else:
        print("\n  SOME TESTS FAILED – see above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
