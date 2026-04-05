"""
Phase 4/5 – Results analysis and comparison table generator.

Reads the Phase 4 CSV logs, computes key metrics, and outputs:
  1. A per-scenario comparison table (sync / async / hybrid)
  2. Per-run switch event details
  3. Time-to-Accuracy (TTA) for a configurable target accuracy
  4. Aggregated multi-seed stats (mean ± std) when multiple seeds are present

Usage
─────
  # Analyse all Phase 4 logs (default: ./logs/):
  python experiments/analyze_results.py

  # Specify a custom log directory:
  python experiments/analyze_results.py --log-dir ./logs

  # Set TTA target accuracy:
  python experiments/analyze_results.py --tta-target 40.0

  # Save output tables to a markdown file:
  python experiments/analyze_results.py --output docs/phase4_results_tables.md

Metrics computed
────────────────
  final_val_acc     : last val_acc_top1 in PS log
  tta_s             : wall-clock seconds to first reach --tta-target accuracy
  total_wall_s      : elapsed_time of the final PS log row
  avg_throughput    : mean throughput_img_per_sec across all PS log rows
  switch_count      : number of mode switches
  final_staleness   : avg_staleness of the last PS log row
  mode_seg_sync_pct : fraction of global steps spent in sync mode
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
#  CSV readers
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
#  Per-run metrics
# ─────────────────────────────="────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def compute_run_metrics(
    log_dir: str,
    run_tag: str,
    tta_target: float,
) -> Dict:
    """
    Compute all metrics for one run identified by its run_tag.

    run_tag corresponds to the suffix used in log filenames, e.g.:
      hybrid_resnet18_cifar10_nw2_<run_tag>_ps.csv
    """
    ps_rows   = _read_csv(os.path.join(log_dir, f"hybrid_resnet18_cifar10_nw2_{run_tag}_ps.csv"))
    sw_rows   = _read_csv(os.path.join(log_dir, f"hybrid_resnet18_cifar10_nw2_{run_tag}_switches.csv"))
    stale_rows = _read_csv(os.path.join(log_dir, f"hybrid_resnet18_cifar10_nw2_{run_tag}_staleness.csv"))

    if not ps_rows:
        return {}

    # Final accuracy and wall time
    final_row = ps_rows[-1]
    final_acc = _float(final_row.get("val_acc_top1"))
    total_wall = _float(final_row.get("elapsed_time"))
    final_staleness = _float(final_row.get("avg_staleness"))

    # Time-to-accuracy (TTA)
    tta_s = float("inf")
    for row in ps_rows:
        acc = _float(row.get("val_acc_top1"))
        if acc >= tta_target:
            tta_s = _float(row.get("elapsed_time"))
            break

    # Average throughput (only rows that have it)
    tp_vals = [
        _float(r.get("throughput_img_per_sec"))
        for r in ps_rows
        if r.get("throughput_img_per_sec") not in ("", None, "0", "0.0")
    ]
    avg_tp = sum(tp_vals) / len(tp_vals) if tp_vals else 0.0

    # Switch events
    switch_count = len(sw_rows)

    # Mode fraction from staleness log
    total_stale_rows = len(stale_rows)
    sync_rows_count = sum(
        1 for r in stale_rows
        if r.get("current_mode", "").strip().lower() == "sync"
    )
    sync_pct = (sync_rows_count / total_stale_rows * 100) if total_stale_rows else 100.0

    # Switch event details (list of dicts)
    switch_events = []
    for sw in sw_rows:
        switch_events.append({
            "global_step":    _int(sw.get("global_step")),
            "old_mode":       sw.get("old_mode", ""),
            "new_mode":       sw.get("new_mode", ""),
            "trigger_reason": sw.get("trigger_reason", sw.get("reason", "")),
            "worker_time_var": _float(sw.get("worker_time_var")),
            "grad_norm_var":   _float(sw.get("grad_norm_var")),
            "loss_trend":      _float(sw.get("loss_trend")),
        })

    return {
        "run_tag":         run_tag,
        "final_val_acc":   round(final_acc, 2),
        "total_wall_s":    round(total_wall, 1),
        "tta_s":           round(tta_s, 1) if not math.isinf(tta_s) else None,
        "avg_throughput":  round(avg_tp, 1),
        "switch_count":    switch_count,
        "final_staleness": round(final_staleness, 2),
        "sync_pct":        round(sync_pct, 1),
        "switch_events":   switch_events,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-seed aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    n = len(vals)
    mu = sum(vals) / n
    if n == 1:
        return round(mu, 2), 0.0
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return round(mu, 2), round(var ** 0.5, 2)


def aggregate_metrics(runs: List[Dict]) -> Dict:
    """Aggregate a list of same-config runs (different seeds)."""
    if not runs:
        return {}
    scalar_keys = ["final_val_acc", "total_wall_s", "avg_throughput",
                   "switch_count", "final_staleness", "sync_pct"]
    agg = {}
    for k in scalar_keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        mu, sd = _mean_std(vals)
        agg[f"{k}_mean"] = mu
        agg[f"{k}_std"]  = sd

    # TTA: None means did not reach target — exclude from mean
    tta_vals = [r["tta_s"] for r in runs if r.get("tta_s") is not None]
    mu, sd = _mean_std(tta_vals)
    agg["tta_s_mean"] = mu
    agg["tta_s_std"]  = sd
    agg["tta_reached_pct"] = round(len(tta_vals) / len(runs) * 100, 0)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
#  Auto-discover runs from log_dir
# ─────────────────────────────────────────────────────────────────────────────

def discover_run_tags(log_dir: str) -> List[str]:
    """Find all run_tags in log_dir by scanning for *_ps.csv files."""
    prefix = "hybrid_resnet18_cifar10_nw2_"
    suffix = "_ps.csv"
    tags = []
    if not os.path.isdir(log_dir):
        return tags
    for fname in sorted(os.listdir(log_dir)):
        if fname.startswith(prefix) and fname.endswith(suffix):
            tag = fname[len(prefix): -len(suffix)]
            if tag:
                tags.append(tag)
    return tags


# ─────────────────────────────────────────────────────────────────────────────
#  Table formatters
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val, na="N/A") -> str:
    if val is None:
        return na
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def build_comparison_table(
    grouped: Dict,  # {(scenario, mode): List[metrics_dict]}
    tta_target: float,
) -> str:
    """Build a markdown comparison table grouped by scenario."""
    scenarios = ["s0", "sA", "sB"]
    modes = ["sync", "async", "hybrid"]
    scenario_labels = {
        "s0": "Scenario 0 – Homogeneous",
        "sA": "Scenario A – Persistent Straggler",
        "sB": "Scenario B – Transient Straggler",
    }

    lines = []
    lines.append(f"## Phase 4 Results (TTA target = {tta_target}% val accuracy)\n")

    for sc in scenarios:
        lines.append(f"### {scenario_labels.get(sc, sc)}\n")
        lines.append(
            "| Mode   | Final Acc (%) | TTA (s)  | Wall (s) | Throughput (img/s) | "
            "Switches | Staleness | Sync % |"
        )
        lines.append(
            "|--------|:-------------:|:--------:|:--------:|:------------------:|"
            ":--------:|:---------:|:------:|"
        )
        for mode in modes:
            key = (sc, mode)
            runs = grouped.get(key, [])
            if not runs:
                lines.append(f"| {mode:<6} | — | — | — | — | — | — | — |")
                continue
            agg = aggregate_metrics(runs)
            acc_str = f"{agg['final_val_acc_mean']:.1f} ± {agg['final_val_acc_std']:.1f}"
            tta_str = (
                f"{agg['tta_s_mean']:.0f} ± {agg['tta_s_std']:.0f}"
                if agg["tta_reached_pct"] > 0 else "—"
            )
            wall_str  = f"{agg['total_wall_s_mean']:.0f} ± {agg['total_wall_s_std']:.0f}"
            tp_str    = f"{agg['avg_throughput_mean']:.1f}"
            sw_str    = f"{agg['switch_count_mean']:.1f}"
            stale_str = f"{agg['final_staleness_mean']:.1f}"
            sync_str  = f"{agg['sync_pct_mean']:.0f}%"
            lines.append(
                f"| {mode:<6} | {acc_str:^13} | {tta_str:^8} | {wall_str:^8} | "
                f"{tp_str:^18} | {sw_str:^8} | {stale_str:^9} | {sync_str:^6} |"
            )
        lines.append("")

    return "\n".join(lines)


def build_switch_events_table(all_metrics: List[Dict]) -> str:
    """Build a markdown table of all mode switch events across all runs."""
    lines = ["## Phase 4 Mode Switch Events\n"]
    lines.append(
        "| Run Tag | Global Step | Old Mode | New Mode | Trigger Reason | "
        "W-TimeVar | GradNormVar | LossTrend |"
    )
    lines.append(
        "|---------|:-----------:|:--------:|:--------:|:-------------- |"
        ":---------:|:-----------:|:---------:|"
    )

    any_events = False
    for m in all_metrics:
        for ev in m.get("switch_events", []):
            any_events = True
            lines.append(
                f"| {m['run_tag']} | {ev['global_step']} | {ev['old_mode']} | "
                f"{ev['new_mode']} | {ev['trigger_reason']} | "
                f"{ev['worker_time_var']:.4f} | {ev['grad_norm_var']:.4f} | "
                f"{ev['loss_trend']:.4f} |"
            )

    if not any_events:
        lines.append("| (no switch events found) | — | — | — | — | — | — | — |")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 results analyser.")
    parser.add_argument("--log-dir", default="./logs",
                        help="Directory containing Phase 4 CSV logs.")
    parser.add_argument("--tta-target", type=float, default=40.0,
                        help="Validation accuracy target for TTA metric (default 40%%).")
    parser.add_argument("--output", default=None,
                        help="Optional path to write markdown tables.")
    args = parser.parse_args()

    log_dir = os.path.join(_PROJECT_ROOT, args.log_dir)
    tags = discover_run_tags(log_dir)

    if not tags:
        print(f"No *_ps.csv files found in {log_dir}. Run experiments first.")
        return

    print(f"Found {len(tags)} run(s) in {log_dir}:")
    for t in tags:
        print(f"  {t}")

    # Compute metrics per run
    all_metrics = []
    for tag in tags:
        m = compute_run_metrics(log_dir, tag, tta_target=args.tta_target)
        if m:
            all_metrics.append(m)

    # Group by (scenario, mode) using run_tag naming convention:
    # p4_<scenario>_<mode>_s<seed>  OR  p4_<scenario>_<mode>  (legacy single-seed)
    grouped: Dict = defaultdict(list)
    ungrouped = []
    for m in all_metrics:
        tag = m["run_tag"]
        parsed = False
        # Try: p4_s0_sync_s42, p4_sA_hybrid_s123, etc.
        parts = tag.split("_")
        if len(parts) >= 3 and parts[0] == "p4":
            scenario = parts[1]           # s0 / sA / sB
            mode = parts[2]               # sync / async / hybrid
            grouped[(scenario, mode)].append(m)
            parsed = True
        if not parsed:
            ungrouped.append(m)

    # Build tables
    comparison_md  = build_comparison_table(grouped, args.tta_target)
    switch_ev_md   = build_switch_events_table(all_metrics)

    output = "\n\n".join([comparison_md, switch_ev_md])
    print("\n" + "=" * 70)
    print(output)
    print("=" * 70)

    if args.output:
        out_path = os.path.join(_PROJECT_ROOT, args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(output)
        print(f"\nTables written to: {out_path}")

    # Per-run detail
    print("\n## Per-run detail")
    print(f"{'Run Tag':<45} {'FinalAcc':>9} {'TTA(s)':>8} {'Wall(s)':>8} "
          f"{'Switch#':>8} {'Stale':>7} {'Sync%':>6}")
    print("-" * 95)
    for m in sorted(all_metrics, key=lambda x: x["run_tag"]):
        tta = f"{m['tta_s']:.0f}" if m["tta_s"] is not None else "—"
        print(
            f"{m['run_tag']:<45} {m['final_val_acc']:>9.2f} {tta:>8} "
            f"{m['total_wall_s']:>8.0f} {m['switch_count']:>8} "
            f"{m['final_staleness']:>7.1f} {m['sync_pct']:>6.0f}%"
        )


if __name__ == "__main__":
    main()
