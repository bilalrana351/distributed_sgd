"""
Phase 5 – Final report assets.

Consumes the already-merged Phase 4 logs (logs/phase4_merged/) and produces:

  1. logs/phase5_per_seed.csv           -- one row per (scenario, mode, seed) run
  2. docs/phase5_per_seed_table.md      -- same content as a readable markdown table
  3. docs/figures/phase5_tta_bars.png             -- TTA mean±std bars per scenario
  4. docs/figures/phase5_acc_vs_walltime.png      -- accuracy-vs-wall-clock curves
  5. docs/figures/phase5_hybrid_mode_timeline.png -- hybrid mode regions per run
  6. docs/figures/phase5_throughput_staleness.png -- throughput + staleness bars

Run from the repo root (PowerShell):

  .\\.venv\\Scripts\\python.exe experiments\\phase5_report_assets.py

All paths are resolved from the repository root automatically.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.analyze_results import (  # noqa: E402
    _read_csv,
    compute_run_metrics,
    discover_run_tags,
)

PREFIX = "hybrid_resnet18_cifar10_nw2_"
SCENARIOS = ["s0", "sA", "sB"]
MODES = ["sync", "async", "hybrid"]
SCENARIO_LABEL = {
    "s0": "Scenario 0 – Homogeneous",
    "sA": "Scenario A – Persistent straggler",
    "sB": "Scenario B – Transient straggler",
}
MODE_COLOR = {
    "sync": "#1f77b4",
    "async": "#ff7f0e",
    "hybrid": "#2ca02c",
}


def _parse_tag(tag: str) -> Optional[Tuple[str, str, str]]:
    """p4_<scenario>_<mode>_s<seed>  ->  (scenario, mode, seed) or None."""
    parts = tag.split("_")
    if len(parts) != 4 or parts[0] != "p4":
        return None
    scenario, mode, seed_part = parts[1], parts[2], parts[3]
    if not seed_part.startswith("s"):
        return None
    return scenario, mode, seed_part[1:]


def collect_per_seed(log_dir: str, tta_target: float) -> List[Dict]:
    """One dict per run: scenario, mode, seed, run_tag + all scalar metrics."""
    rows: List[Dict] = []
    for tag in discover_run_tags(log_dir):
        parsed = _parse_tag(tag)
        if parsed is None:
            continue
        scenario, mode, seed = parsed
        metrics = compute_run_metrics(log_dir, tag, tta_target)
        if not metrics:
            continue
        rows.append(
            {
                "scenario": scenario,
                "mode": mode,
                "seed": seed,
                "run_tag": tag,
                "final_val_acc": metrics["final_val_acc"],
                "tta_s": metrics["tta_s"],
                "total_wall_s": metrics["total_wall_s"],
                "avg_throughput": metrics["avg_throughput"],
                "switch_count": metrics["switch_count"],
                "final_staleness": metrics["final_staleness"],
                "sync_pct": metrics["sync_pct"],
            }
        )
    rows.sort(key=lambda r: (SCENARIOS.index(r["scenario"]) if r["scenario"] in SCENARIOS else 99,
                             MODES.index(r["mode"]) if r["mode"] in MODES else 99,
                             r["seed"]))
    return rows


def write_per_seed_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "scenario", "mode", "seed", "run_tag",
        "final_val_acc", "tta_s", "total_wall_s",
        "avg_throughput", "switch_count",
        "final_staleness", "sync_pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_per_seed_markdown(rows: List[Dict], path: str, tta_target: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# Phase 5 – Per-seed detail table",
        "",
        f"Every row is a single training run. TTA target = {tta_target}% val accuracy; "
        "`—` in TTA means the run did not reach the target inside its budget.",
        "",
        "| Scenario | Mode | Seed | Final Acc (%) | TTA (s) | Wall (s) | Throughput (img/s) | Switches | Staleness | Sync % |",
        "|----------|------|-----:|--------------:|--------:|---------:|-------------------:|---------:|----------:|-------:|",
    ]
    for r in rows:
        tta = f"{r['tta_s']:.0f}" if r["tta_s"] is not None else "—"
        lines.append(
            f"| {r['scenario']} | {r['mode']} | {r['seed']} | "
            f"{r['final_val_acc']:.2f} | {tta} | {r['total_wall_s']:.0f} | "
            f"{r['avg_throughput']:.1f} | {r['switch_count']} | "
            f"{r['final_staleness']:.1f} | {r['sync_pct']:.0f}% |"
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _group(rows: List[Dict]) -> Dict[Tuple[str, str], List[Dict]]:
    g: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        g[(r["scenario"], r["mode"])].append(r)
    return g


def plot_tta_bars(rows: List[Dict], out_path: str, tta_target: float) -> None:
    import matplotlib.pyplot as plt

    grouped = _group(rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(SCENARIOS))
    width = 0.26
    for i, mode in enumerate(MODES):
        means, stds, reached = [], [], []
        for sc in SCENARIOS:
            runs = grouped.get((sc, mode), [])
            vals = [r["tta_s"] for r in runs if r["tta_s"] is not None]
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                reached.append(f"{len(vals)}/{len(runs)}")
            else:
                means.append(0.0)
                stds.append(0.0)
                reached.append(f"0/{len(runs) if runs else 0}")
        bars = ax.bar(x + (i - 1) * width, means, width,
                      yerr=stds, capsize=4, label=mode, color=MODE_COLOR[mode],
                      edgecolor="black", linewidth=0.5)
        for b, m, r in zip(bars, means, reached):
            if m > 0:
                ax.text(b.get_x() + b.get_width() / 2, m + max(stds) * 0.35 + 15,
                        r, ha="center", va="bottom", fontsize=8, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIOS], rotation=0, fontsize=9)
    ax.set_ylabel(f"Time to reach {tta_target:.0f}% val acc (s)")
    ax.set_title("Phase 5 — Time-to-Accuracy per scenario and mode (mean ± std over seeds)")
    ax.legend(title="Mode")
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0.01, 0.98,
            "numbers above bars = seeds that reached target / total seeds",
            transform=ax.transAxes, fontsize=8, va="top", color="#555")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ps_series(log_dir: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (elapsed_time, val_acc_top1) arrays from a PS csv."""
    rows = _read_csv(os.path.join(log_dir, f"{PREFIX}{tag}_ps.csv"))
    if not rows:
        return np.array([]), np.array([])
    t = np.array([float(r["elapsed_time"]) for r in rows], dtype=float)
    a = np.array([float(r["val_acc_top1"]) for r in rows], dtype=float)
    order = np.argsort(t)
    return t[order], a[order]


def plot_acc_vs_walltime(log_dir: str, rows: List[Dict], out_path: str,
                         tta_target: float) -> None:
    import matplotlib.pyplot as plt

    grouped = _group(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, sc in zip(axes, SCENARIOS):
        for mode in MODES:
            runs = grouped.get((sc, mode), [])
            series = []
            tmax = 0.0
            for r in runs:
                t, a = _ps_series(log_dir, r["run_tag"])
                if t.size >= 2:
                    series.append((t, a))
                    tmax = max(tmax, float(t.max()))
            if not series:
                continue
            grid = np.linspace(0.0, tmax, 80)
            stacked = np.stack([np.interp(grid, t, a) for t, a in series], axis=0)
            mean_acc = stacked.mean(axis=0)
            std_acc = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean_acc)
            col = MODE_COLOR[mode]
            ax.plot(grid, mean_acc, color=col, linewidth=2, label=mode)
            if stacked.shape[0] > 1:
                ax.fill_between(grid, mean_acc - std_acc, mean_acc + std_acc,
                                color=col, alpha=0.18)
        ax.axhline(tta_target, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(SCENARIO_LABEL[sc], fontsize=10)
        ax.set_xlabel("Wall-clock time (s)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Val acc top-1 (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(
        f"Phase 5 — Validation accuracy vs wall-clock time (shaded band = ±1 std across seeds; "
        f"dashed line = {tta_target:.0f}%)",
        y=1.10, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hybrid_mode_timeline(log_dir: str, rows: List[Dict], out_path: str) -> None:
    """For every hybrid run in sA and sB, draw a horizontal track showing mode over global step.

    Sync = blue band, Async = orange band; switch events are annotated with their trigger.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    hybrid_runs = [r for r in rows if r["mode"] == "hybrid" and r["scenario"] in ("sA", "sB")]
    hybrid_runs.sort(key=lambda r: (r["scenario"], r["seed"]))
    if not hybrid_runs:
        return

    fig, ax = plt.subplots(figsize=(11, max(2.5, 0.7 * len(hybrid_runs) + 1.2)))
    yticks, ylabels = [], []

    for i, r in enumerate(hybrid_runs):
        tag = r["run_tag"]
        ps_rows = _read_csv(os.path.join(log_dir, f"{PREFIX}{tag}_ps.csv"))
        sw_rows = _read_csv(os.path.join(log_dir, f"{PREFIX}{tag}_switches.csv"))
        stale_rows = _read_csv(os.path.join(log_dir, f"{PREFIX}{tag}_staleness.csv"))

        total_steps = 0
        if ps_rows:
            try:
                total_steps = max(int(float(p["global_step"])) for p in ps_rows)
            except (TypeError, ValueError):
                total_steps = 0
        if total_steps == 0 and stale_rows:
            try:
                total_steps = max(int(float(s.get("global_step", 0))) for s in stale_rows)
            except (TypeError, ValueError):
                total_steps = 0
        if total_steps == 0:
            continue

        segments: List[Tuple[int, int, str]] = []
        start_mode = "sync"
        if ps_rows:
            first_mode = (ps_rows[0].get("current_mode") or "").strip().lower()
            if first_mode in ("sync", "async"):
                start_mode = first_mode
        prev_step = 0
        prev_mode = start_mode
        for sw in sw_rows:
            try:
                gs = int(float(sw.get("global_step", 0)))
            except (TypeError, ValueError):
                continue
            new_mode = (sw.get("new_mode") or "").strip().lower()
            if new_mode not in ("sync", "async"):
                continue
            if gs > prev_step:
                segments.append((prev_step, gs, prev_mode))
            prev_step = gs
            prev_mode = new_mode
        segments.append((prev_step, total_steps, prev_mode))

        y = i
        for s0, s1, m in segments:
            ax.barh(y, width=s1 - s0, left=s0, height=0.7,
                    color=MODE_COLOR.get(m, "#cccccc"), edgecolor="black", linewidth=0.3)
        for sw in sw_rows:
            try:
                gs = int(float(sw.get("global_step", 0)))
            except (TypeError, ValueError):
                continue
            reason = sw.get("trigger_reason") or sw.get("reason") or ""
            ax.axvline(gs, ymin=(y + 0.18) / len(hybrid_runs),
                       ymax=(y + 0.82) / len(hybrid_runs),
                       color="black", linewidth=0.7, alpha=0.8)
            ax.text(gs, y + 0.45, reason, rotation=90, fontsize=7,
                    ha="right", va="center", color="black")

        yticks.append(y)
        ylabels.append(f"{r['scenario']} seed {r['seed']}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("Global step")
    ax.set_title("Phase 5 — Hybrid mode timeline (blue = sync, orange = async; vertical lines = switch events)")
    legend_handles = [
        mpatches.Patch(color=MODE_COLOR["sync"], label="sync"),
        mpatches.Patch(color=MODE_COLOR["async"], label="async"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_throughput_staleness(rows: List[Dict], out_path: str) -> None:
    import matplotlib.pyplot as plt

    grouped = _group(rows)
    x = np.arange(len(SCENARIOS))
    width = 0.26
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    for i, mode in enumerate(MODES):
        tp_means, tp_stds = [], []
        st_means, st_stds = [], []
        for sc in SCENARIOS:
            runs = grouped.get((sc, mode), [])
            tps = [r["avg_throughput"] for r in runs]
            sts = [r["final_staleness"] for r in runs]
            tp_means.append(float(np.mean(tps)) if tps else 0.0)
            tp_stds.append(float(np.std(tps, ddof=1)) if len(tps) > 1 else 0.0)
            st_means.append(float(np.mean(sts)) if sts else 0.0)
            st_stds.append(float(np.std(sts, ddof=1)) if len(sts) > 1 else 0.0)
        ax1.bar(x + (i - 1) * width, tp_means, width, yerr=tp_stds, capsize=3,
                label=mode, color=MODE_COLOR[mode], edgecolor="black", linewidth=0.4)
        ax2.bar(x + (i - 1) * width, st_means, width, yerr=st_stds, capsize=3,
                label=mode, color=MODE_COLOR[mode], edgecolor="black", linewidth=0.4)

    for ax, title, ylabel in (
        (ax1, "Average throughput", "Images / second"),
        (ax2, "Final staleness (end of run)", "Staleness (steps)"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS, fontsize=10)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(title="Mode", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Phase 5 — Resource utilisation signals per scenario and mode", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 report assets.")
    parser.add_argument("--merged-dir", default="logs/phase4_merged")
    parser.add_argument("--per-seed-csv", default="logs/phase5_per_seed.csv")
    parser.add_argument("--per-seed-md", default="docs/phase5_per_seed_table.md")
    parser.add_argument("--figures-dir", default="docs/figures")
    parser.add_argument("--tta-target", type=float, default=40.0)
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is required. Install with: pip install -r requirements.txt")
        sys.exit(1)

    merged = os.path.join(_PROJECT_ROOT, args.merged_dir)
    per_seed_csv = os.path.join(_PROJECT_ROOT, args.per_seed_csv)
    per_seed_md = os.path.join(_PROJECT_ROOT, args.per_seed_md)
    fig_dir = os.path.join(_PROJECT_ROOT, args.figures_dir)

    if not os.path.isdir(merged):
        print(f"ERROR: merged log directory not found: {merged}")
        sys.exit(1)

    rows = collect_per_seed(merged, args.tta_target)
    if not rows:
        print(f"ERROR: no runs discovered under {merged}")
        sys.exit(1)

    write_per_seed_csv(rows, per_seed_csv)
    write_per_seed_markdown(rows, per_seed_md, args.tta_target)

    plot_tta_bars(rows, os.path.join(fig_dir, "phase5_tta_bars.png"), args.tta_target)
    plot_acc_vs_walltime(merged, rows,
                         os.path.join(fig_dir, "phase5_acc_vs_walltime.png"),
                         args.tta_target)
    plot_hybrid_mode_timeline(merged, rows,
                              os.path.join(fig_dir, "phase5_hybrid_mode_timeline.png"))
    plot_throughput_staleness(rows,
                              os.path.join(fig_dir, "phase5_throughput_staleness.png"))

    print(f"Wrote {per_seed_csv}")
    print(f"Wrote {per_seed_md}")
    print(f"Wrote {os.path.join(fig_dir, 'phase5_tta_bars.png')}")
    print(f"Wrote {os.path.join(fig_dir, 'phase5_acc_vs_walltime.png')}")
    print(f"Wrote {os.path.join(fig_dir, 'phase5_hybrid_mode_timeline.png')}")
    print(f"Wrote {os.path.join(fig_dir, 'phase5_throughput_staleness.png')}")
    print(f"Rows: {len(rows)}  scenarios x modes: {len(_group(rows))}")


if __name__ == "__main__":
    main()
