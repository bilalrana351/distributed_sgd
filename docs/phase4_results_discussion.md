# Phase 4 — Results discussion and sweep

This note accompanies the aggregated tables in [phase4_results_tables.md](phase4_results_tables.md).

## Figures

- **Validation curves** (mean over seeds; linear interpolation on a common step grid when PS rows differ): [`docs/figures/phase4_val_acc_curves.png`](docs/figures/phase4_val_acc_curves.png)
- **Threshold sweep** (exploratory grid on Scenario A): [`docs/figures/phase4_sweep_summary.png`](docs/figures/phase4_sweep_summary.png)

## Main comparison (recap)

- **Sync and async** rows pool the historical exports under `phase_4_logs/zip` (flat merge via `experiments/merge_phase4_compare.py`).
- **Hybrid** for Scenarios A and B uses the post-fix logs from `threshold_folders`.
- **Scenario 0 hybrid** was not re-run for this table; the hybrid cell is intentionally empty (—).
- Metrics in the table follow `experiments/analyze_results.py` (final val acc, TTA at 40%, wall time, mean throughput, switch count, staleness, sync % from the staleness log).

## Threshold sweep (17 completed configs)

The sweep varies controller thresholds on **Scenario A** with a fixed budget in the sweep driver. Each row is one run. Numeric columns are parsed from the run id (`svh` = straggler variance high gate, `gvl` = gradient-variance low gate, `cd` = cooldown steps). **This block is separate from the sA/sB hybrid rows** in the main table (different objective: grid search vs. final protocol).

| run_id | straggler_var_high | grad_var_low | cooldown | switches | first_sw_step | final_val_acc | wall_s |
|--------|-------------------:|-------------:|---------:|---------:|--------------:|--------------:|-------:|
| `sweep_17_svh025_gvl07_cd100` | 0.25 | 0.70 | 100 | 1 | 120 | 37.24 | 660.0 |
| `sweep_11_svh018_gvl085_cd100` | 0.18 | 0.85 | 100 | 1 | 120 | 37.19 | 654.3 |
| `sweep_04_svh012_gvl085_cd150` | 0.12 | 0.85 | 150 | 1 | 160 | 36.82 | 705.5 |
| `sweep_10_svh018_gvl07_cd150` | 0.18 | 0.70 | 150 | 1 | 160 | 36.20 | 717.5 |
| `sweep_03_svh012_gvl085_cd100` | 0.12 | 0.85 | 100 | 1 | 120 | 35.52 | 648.3 |
| `sweep_05_svh012_gvl095_cd100` | 0.12 | 0.95 | 100 | 1 | 120 | 35.42 | 651.0 |
| `sweep_02_svh012_gvl07_cd150` | 0.12 | 0.70 | 150 | 1 | 160 | 35.26 | 703.1 |
| `sweep_15_svh018_gvl10_cd100` | 0.18 | 1.00 | 100 | 1 | 120 | 34.38 | 639.0 |
| `sweep_09_svh018_gvl07_cd100` | 0.18 | 0.70 | 100 | 1 | 120 | 34.22 | 652.2 |
| `sweep_14_svh018_gvl095_cd150` | 0.18 | 0.95 | 150 | 1 | 160 | 33.23 | 712.1 |
| `sweep_01_svh012_gvl07_cd100` | 0.12 | 0.70 | 100 | 1 | 120 | 32.55 | 642.4 |
| `sweep_07_svh012_gvl10_cd100` | 0.12 | 1.00 | 100 | 1 | 120 | 32.08 | 644.9 |
| `sweep_08_svh012_gvl10_cd150` | 0.12 | 1.00 | 150 | 1 | 160 | 31.93 | 696.6 |
| `sweep_12_svh018_gvl085_cd150` | 0.18 | 0.85 | 150 | 1 | 160 | 31.72 | 699.8 |
| `sweep_13_svh018_gvl095_cd100` | 0.18 | 0.95 | 100 | 1 | 120 | 31.04 | 660.0 |
| `sweep_16_svh018_gvl10_cd150` | 0.18 | 1.00 | 150 | 1 | 160 | 30.94 | 699.6 |
| `sweep_06_svh012_gvl095_cd150` | 0.12 | 0.95 | 150 | 1 | 160 | 30.68 | 699.3 |

**How to read the scatter plot:** x-axis is `straggler_var_high`, y-axis is final validation accuracy; colour encodes `grad_var_low`. The right panel ranks the same runs by final accuracy. See switch counts in the table for controller churn.

## Limitations and caveats

1. **Mixed log vintages:** sync/async and hybrid for A/B do not come from the same export batch; only hybrid uses the post-fix switch logging. Mention this when comparing switch counts or hybrid dynamics to pure baselines.
2. **Seeds:** Aggregates are mean ± std over three seeds where available; hybrid on Scenario B shows higher variance and some seeds miss the 40% TTA target—interpret TTA and wall time with the per-run table from the analyzer.
3. **Async on Scenario A** reaches the TTA target quickly but ends below sync/hybrid final accuracy in this setup; that is a trade-off to describe, not a bug in the analyzer.
4. **Sweep incompleteness:** The exported sweep has **17** points; the full Cartesian grid in `experiments/threshold_sweep.py` can be larger. Treat the sweep as exploratory.
5. **Throughput in the markdown table** shows mean of per-row `throughput_img_per_sec` from PS logs (not a calibrated hardware benchmark).

## Reproducibility

```text
python experiments/merge_phase4_compare.py
python experiments/analyze_results.py --log-dir logs/phase4_merged --tta-target 40.0 --output docs/phase4_results_tables.md
python experiments/phase4_report_assets.py
```

Sweep summary CSV: `logs/phase4_sweep_results_from_logs.csv` (generated from files under `threshold_folders`, not from re-running the sweep).
