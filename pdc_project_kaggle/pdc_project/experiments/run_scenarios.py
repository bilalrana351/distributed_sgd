"""
Phase 4 – Scenario experiment orchestrator.

Runs all 9 Phase 4 experiments sequentially (or a chosen subset) and
prints a timing summary at the end.  Each experiment is launched via the
same hybrid launcher used for Phase 3, so no new code paths are introduced.

Usage
─────
  # Run all 9 scenarios (may take several hours on CPU):
  python experiments/run_scenarios.py

  # Run a single scenario group (all 3 modes):
  python experiments/run_scenarios.py --scenario s0

  # Run one specific mode:
  python experiments/run_scenarios.py --scenario sA --mode hybrid

  # Run with a different seed (overrides the config seed):
  python experiments/run_scenarios.py --seed 123

  # Dry-run: print commands without executing:
  python experiments/run_scenarios.py --dry-run

Output
──────
  All logs land in ./logs/ with run_id-based names (p4_<scenario>_<mode>_s<seed>).
  A summary CSV is written to ./logs/phase4_run_summary.csv on completion.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
from typing import Dict, List, Optional

import yaml

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
#  Experiment registry
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS: List[Dict] = [
    # Scenario 0 – Homogeneous
    {"scenario": "s0", "mode": "sync",   "config": "configs/phase4_s0_sync.yaml",   "nworkers": 2},
    {"scenario": "s0", "mode": "async",  "config": "configs/phase4_s0_async.yaml",  "nworkers": 2},
    {"scenario": "s0", "mode": "hybrid", "config": "configs/phase4_s0_hybrid.yaml", "nworkers": 2},
    # Scenario A – Persistent Straggler
    {"scenario": "sA", "mode": "sync",   "config": "configs/phase4_sA_sync.yaml",   "nworkers": 2},
    {"scenario": "sA", "mode": "async",  "config": "configs/phase4_sA_async.yaml",  "nworkers": 2},
    {"scenario": "sA", "mode": "hybrid", "config": "configs/phase4_sA_hybrid.yaml", "nworkers": 2},
    # Scenario B – Transient Straggler
    {"scenario": "sB", "mode": "sync",   "config": "configs/phase4_sB_sync.yaml",   "nworkers": 2},
    {"scenario": "sB", "mode": "async",  "config": "configs/phase4_sB_async.yaml",  "nworkers": 2},
    {"scenario": "sB", "mode": "hybrid", "config": "configs/phase4_sB_hybrid.yaml", "nworkers": 2},
]

SUMMARY_FIELDS = [
    "scenario", "mode", "seed", "config",
    "status", "wall_time_s", "exit_code",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Run one experiment
# ─────────────────────────────────────────────────────────────────────────────

def _patch_config_for_seed(cfg: dict, seed: int, scenario: str, mode: str) -> dict:
    """Return a shallow copy of cfg with seed + run_id updated."""
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    base_rid = f"p4_{scenario}_{mode}_s{seed}"
    cfg["run_id"] = base_rid
    return cfg


def _run_experiment(
    exp: Dict,
    seed: int,
    dry_run: bool,
    verbose: bool,
) -> Dict:
    """Run a single experiment and return a summary row."""
    import subprocess

    config_path = exp["config"]
    nworkers = exp["nworkers"]
    scenario = exp["scenario"]
    mode = exp["mode"]

    # Load and patch config
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    cfg = _patch_config_for_seed(cfg, seed, scenario, mode)

    # Write a temporary patched config so the launcher can read it
    os.makedirs("./logs", exist_ok=True)
    tmp_cfg_path = f"./logs/_tmp_{scenario}_{mode}_s{seed}.yaml"
    with open(tmp_cfg_path, "w") as fh:
        yaml.dump(cfg, fh)

    cmd = [
        sys.executable,
        "src/hybrid/launcher.py",
        "--nworkers", str(nworkers),
        "--config", tmp_cfg_path,
    ]

    label = f"{scenario}/{mode}/seed={seed}"
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Config:  {config_path}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY-RUN] Skipping execution.")
        return {
            "scenario": scenario, "mode": mode, "seed": seed,
            "config": config_path, "status": "dry_run",
            "wall_time_s": 0.0, "exit_code": -1,
        }

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=_PROJECT_ROOT,
        capture_output=not verbose,
        text=True,
    )
    wall = round(time.perf_counter() - t0, 1)

    if result.returncode != 0:
        print(f"  [FAILED] {label} — exit code {result.returncode}")
        if not verbose and result.stderr:
            print(result.stderr[-2000:])
        status = "failed"
    else:
        print(f"  [OK] {label} — {wall:.0f}s")
        status = "ok"

    # Cleanup temp config
    try:
        os.remove(tmp_cfg_path)
    except OSError:
        pass

    return {
        "scenario": scenario, "mode": mode, "seed": seed,
        "config": config_path, "status": status,
        "wall_time_s": wall, "exit_code": result.returncode,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 scenario runner.")
    parser.add_argument(
        "--scenario",
        choices=["s0", "sA", "sB", "all"],
        default="all",
        help="Which scenario group to run (default: all).",
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "async", "hybrid", "all"],
        default="all",
        help="Which mode to run within the scenario (default: all).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds to run (default: 42). Use e.g. --seeds 42 123 777.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream subprocess output to stdout.",
    )
    args = parser.parse_args()

    # Filter experiment list
    exps = EXPERIMENTS
    if args.scenario != "all":
        exps = [e for e in exps if e["scenario"] == args.scenario]
    if args.mode != "all":
        exps = [e for e in exps if e["mode"] == args.mode]

    if not exps:
        print("No experiments match the filter. Exiting.")
        return

    total = len(exps) * len(args.seeds)
    print(f"\nPhase 4 Scenario Runner")
    print(f"  Experiments to run : {len(exps)}")
    print(f"  Seeds              : {args.seeds}")
    print(f"  Total runs         : {total}")
    print(f"  Dry-run            : {args.dry_run}")

    summary_rows = []
    overall_t0 = time.perf_counter()

    for seed in args.seeds:
        for exp in exps:
            row = _run_experiment(exp, seed=seed, dry_run=args.dry_run, verbose=args.verbose)
            summary_rows.append(row)

    # Write summary CSV
    summary_path = os.path.join(_PROJECT_ROOT, "logs", "phase4_run_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    total_wall = round(time.perf_counter() - overall_t0, 1)
    ok_count = sum(1 for r in summary_rows if r["status"] == "ok")
    fail_count = sum(1 for r in summary_rows if r["status"] == "failed")

    print(f"\n{'='*60}")
    print(f"  ALL DONE in {total_wall:.0f}s")
    print(f"  OK: {ok_count}  |  Failed: {fail_count}  |  Total: {total}")
    print(f"  Summary → {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
