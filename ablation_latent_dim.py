#!/usr/bin/env python3
"""
Ablation utility: run `newMain5.py` multiple times with different latent-space dimensions
and compare performance.

Example:
    python ablation_latent_dim.py --dims 128,256,512,768 --epochs 50 --batchSize 8 --dataset neuro

Any extra flags will be forwarded to `newMain5.py` in each run (except `--graphEmDim`, which
is set automatically).
"""
import argparse
import subprocess
import sys
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent-dimension ablation wrapper for newMain5.py")
    parser.add_argument("--dims", required=True, help="Comma-separated list of latent dims, e.g. 128,256,512")
    # All remaining args will be captured and forwarded unchanged
    parser.add_argument("remaining", nargs=argparse.REMAINDER, help="Additional flags forwarded to newMain5.py")
    return parser.parse_args()


def extract_best_auc(stdout: str) -> float:
    """Try to find the final reported best AUC from newMain5.py output."""
    # Look for the last line that looks like: "--- Pipeline Finished. Best Test AUC with SVM: 0.8421 ---"
    match = None
    for line in reversed(stdout.splitlines()):
        m = re.search(r"Best Test AUC.*?([0-9]+\.[0-9]+)", line)
        if m:
            match = float(m.group(1))
            break
    return match if match is not None else float("nan")


def main():
    args = parse_args()
    dims = [int(d.strip()) for d in args.dims.split(',') if d.strip()]
    if not dims:
        print("No valid dimensions provided.", file=sys.stderr)
        sys.exit(1)

    script_path = Path(__file__).parent / "newMain5.py"
    if not script_path.exists():
        print(f"Cannot locate newMain5.py at {script_path}", file=sys.stderr)
        sys.exit(1)

    summary = {}
    for dim in dims:
        cmd = [sys.executable, str(script_path), f"-graphEmDim", str(dim)] + args.remaining
        print(f"\n================ Latent dim {dim} ================")
        print("Command:", " ".join(cmd))
        completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(completed.stdout)
        auc = extract_best_auc(completed.stdout)
        summary[dim] = auc

    print("\n========== Ablation Summary ==========")
    for dim in dims:
        auc_val = summary.get(dim, float('nan'))
        print(f"Dim {dim:4}: Best Test AUC = {auc_val:.4f}")


if __name__ == "__main__":
    main() 