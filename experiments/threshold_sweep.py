"""
Phase 4 – Controller threshold sweep.

Performs a coarse grid search over the controller's key thresholds using
Scenario A (persistent straggler) as the test bed.  Each combination is
run with a short step budget (total_steps from the base config) so all
sweep points complete quickly.

Metrics collected per sweep point
──────────────────────────────────
  - switch_count   : number of mode switches logged
  - first_switch   : global_step of the first SYNC→ASYNC switch (-1 if none)
  - final_val_acc  : last val_acc_top1 in the PS log
  - wall_time_s    : total wall-clock time for the run

Sweep parameters
────────────────
  straggler_var_high : threshold to trigger SYNC→ASYNC
  grad_var_low       : gradient stability guard for SYNC→ASYNC
  cooldown_steps     : minimum steps between switches

Usage
─────
  python experiments/threshold_sweep.py

  # Specify base config (default: configs/phase4_threshold_sweep_base.yaml):
  python experiments/threshold_sweep.py --base-config configs/phase4_threshold_sweep_base.yaml

  # Print parameter grid and exit without running:
  python experiments/threshold_sweep.py --dry-run

Output
──────
  logs/phase4_sweep_results.csv   — all sweep rows
  A simple best-config summary is printed to stdout.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import subprocess
import sys
import time
from itertools import product
from typing import Dict, List, Optional

import yaml

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
#  Grid definition
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_GRID: Dict[str, List] = {
    "straggler_var_high": [0.12, 0.18, 0.25],
    "grad_var_low":       [0.70, 0.85],
    "cooldown_steps":     [100, 150],
}

SWEEP_RESULT_FIELDS = [
    "run_id",
    "straggler_var_high",
    "grad_var_low",
    "cooldown_steps",
    "switch_count",
    "first_switch_step",
    "final_val_acc",
    "wall_time_s",
    "status",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Log parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_switches(log_dir: str, run_id: str) -> tuple:
    """Return (switch_count, first_switch_step) from the switches CSV."""
    path = os.path.join(log_dir, f"hybrid_resnet18_cifar10_nw2_{run_id}_switches.csv")
    if not os.path.exists(path):
        return 0, -1
    rows = []
    try:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except Exception:
        return 0, -1
    if not rows:
        return 0, -1
    first = int(rows[0].get("global_step", -1))
    return len(rows), first


def _parse_final_val_acc(log_dir: str, run_id: str) -> float:
    """Return the last val_acc_top1 logged in the PS CSV."""
    path = os.path.join(log_dir, f"hybrid_resnet18_cifar10_nw2_{run_id}_ps.csv")
    if not os.path.exists(path):
        return -1.0
    last_acc = -1.0
    try:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    last_acc = float(row.get("val_acc_top1", -1))
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass
    return last_acc


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 threshold sweep.")
    parser.add_argument(
        "--base-config",
        default="configs/phase4_threshold_sweep_base.yaml",
        help="Base YAML config to use (overridden per sweep point).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sweep grid and exit without running.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream subprocess output to stdout.",
    )
    args = parser.parse_args()

    base_config_path = os.path.join(_PROJECT_ROOT, args.base_config)
    with open(base_config_path) as fh:
        base_cfg = yaml.safe_load(fh)

    log_dir = base_cfg.get("log_dir", "./logs/sweep")
    log_dir_abs = os.path.join(_PROJECT_ROOT, log_dir)
    os.makedirs(log_dir_abs, exist_ok=True)

    # Build full Cartesian product
    keys = list(SWEEP_GRID.keys())
    values = list(SWEEP_GRID.values())
    combos = list(product(*values))
    total = len(combos)

    print(f"\nPhase 4 Threshold Sweep")
    print(f"  Base config : {args.base_config}")
    print(f"  Grid params : {keys}")
    print(f"  Total points: {total}")

    if args.dry_run:
        print("\nSweep grid (dry-run):")
        for i, combo in enumerate(combos, 1):
            print(f"  [{i:02d}] " + " | ".join(f"{k}={v}" for k, v in zip(keys, combo)))
        return

    results: List[Dict] = []

    for idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        run_id = (
            f"sweep_{idx:02d}"
            f"_svh{params['straggler_var_high']}"
            f"_gvl{params['grad_var_low']}"
            f"_cd{params['cooldown_steps']}"
        ).replace(".", "")

        print(f"\n[{idx:02d}/{total}] {params} | run_id={run_id}")

        # Patch config
        cfg = copy.deepcopy(base_cfg)
        cfg["run_id"] = run_id
        cfg["seed"] = 42
        ctrl = cfg.get("controller", {})
        for k, v in params.items():
            ctrl[k] = v
        cfg["controller"] = ctrl

        tmp_path = os.path.join(log_dir_abs, f"_tmp_{run_id}.yaml")
        with open(tmp_path, "w") as fh:
            yaml.dump(cfg, fh)

        cmd = [
            sys.executable,
            os.path.join(_PROJECT_ROOT, "src", "hybrid", "launcher.py"),
            "--nworkers", str(base_cfg.get("num_worker_procs", 2)),
            "--config", tmp_path,
        ]

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=_PROJECT_ROOT,
            capture_output=not args.verbose,
            text=True,
        )
        wall = round(time.perf_counter() - t0, 1)

        status = "ok" if proc.returncode == 0 else "failed"
        if status == "failed" and not args.verbose:
            print(f"  FAILED: {proc.stderr[-500:] if proc.stderr else '(no stderr)'}")

        n_switches, first_step = _parse_switches(log_dir_abs, run_id)
        final_acc = _parse_final_val_acc(log_dir_abs, run_id)

        row = {
            "run_id": run_id,
            **params,
            "switch_count": n_switches,
            "first_switch_step": first_step,
            "final_val_acc": final_acc,
            "wall_time_s": wall,
            "status": status,
        }
        results.append(row)
        print(
            f"  → switches={n_switches}, first@step={first_step}, "
            f"acc={final_acc:.2f}%, wall={wall:.0f}s"
        )

        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Write results CSV
    out_csv = os.path.join(_PROJECT_ROOT, "logs", "phase4_sweep_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SWEEP_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    # Print best config (switch_count==1 with earliest first_switch_step)
    candidates = [r for r in results if r["switch_count"] == 1 and r["status"] == "ok"]
    if candidates:
        best = min(candidates, key=lambda r: r["first_switch_step"])
        print(f"\n{'='*60}")
        print(f"  Best config (1 switch, earliest trigger):")
        for k in SWEEP_RESULT_FIELDS:
            print(f"    {k}: {best[k]}")
    else:
        print("\nNo single-switch candidates found.")

    print(f"\n  Sweep results → {out_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
