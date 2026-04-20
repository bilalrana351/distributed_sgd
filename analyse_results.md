## Context

Repo: Phase 4 hybrid training project (`pdc_project`). I finished Kaggle runs with this plan:

- **sA**: hybrid only, **all three seeds** (42, 123, 777) — new logs.
- **sB**: hybrid only, **all three seeds** — new logs.
- **s0**: intentionally **skipped** (no runs).
- **Threshold sweep**: ran `experiments/threshold_sweep.py` (hybrid controller grid on Scenario A); outputs go to `logs/sweep/` and summary CSV `logs/phase4_sweep_results.csv` per repo defaults.

I also have **older logs** in a folder I’ll call `phase_4_logs/` (or similar) from before fixes; those runs had **switch/logging issues**. For **sync and async** modes I want to use those **previous** logs where the filenames/tags are valid, and for **hybrid** I want to use the **new** Kaggle logs so switch behavior is trustworthy.

I have **new** artifacts under something like `threshold_folders/` or extracted zips from Kaggle (hybrid scenario runs + sweep).

## Goal

1. Produce the Phase 4 markdown tables using `experiments/analyze_results.py`.
2. Make the **comparison table** meaningful: **sA/sB** should show **sync + async (old logs) + hybrid (new logs)**; 
3. Treat **threshold sweep** as a **separate analysis** (do not mix sweep `*_ps.csv` into the main scenario comparison unless we explicitly want a third block — default: keep sweep separate).
4. If anything is **ungrouped** (run tags don’t match `p4_<scenario>_<mode>_s<seed>`), help me rename files or document which rows are excluded.
## s0 in the table

- I did **not** re-run s0 on Kaggle for hybrid.
- I **did** complete s0 for **sync** and **async** earlier; those CSVs live in `phase_4_logs` (or the old export).
- When merging logs for `analyze_results.py`, **include all `p4_s0_sync_*` and `p4_s0_async_*` files** from the old run so the Phase 4 table shows s0 sync/async with mean ± std across seeds.
- **Exclude** s0 hybrid unless I add a `p4_s0_hybrid_*` run; expect the **hybrid** cell for s0 to show dashes.

## Technical constraints (from code)

- Analyzer: `python experiments/analyze_results.py --log-dir <DIR> --tta-target 40.0 --output docs/phase4_results_tables.md`
- It only scans **one directory, non-recursively** for files named:
  `hybrid_resnet18_cifar10_nw2_<run_tag>_ps.csv` (plus matching `_switches.csv`, `_staleness.csv`).
- Grouping uses run_tag pattern: `p4_sA_sync_s42`, `p4_sB_hybrid_s123`, etc. (`p4`, scenario `s0|sA|sB`, mode `sync|async|hybrid`, seed suffix).

## What I need you to do

1. **Inventory** what CSVs I have in:
   - old: `phase_4_logs` (sync/async; note legacy switch issues)
   - new: Kaggle exports / `threshold_folders` (hybrid sA/sB all seeds; sweep under `logs/sweep` or equivalent)
2. **Propose a merge layout**, e.g. a single folder `logs/phase4_merged/` where I copy only the files needed for the main table (old sync/async + new hybrid), with **no duplicate run_tags**.
3. **Run** (or give exact commands for) `analyze_results.py` on that merged dir; save to `docs/phase4_results_tables.md`.
4. **Separately**, explain how to interpret or tabulate `logs/phase4_sweep_results.csv` (or run analyzer on `logs/sweep` only if that folder has flat `*_ps.csv` files — check first).
5. **Report narrative**: 1–2 sentences I can use in write-up: hybrid metrics from post-fix runs; sync/async from earlier logs; s0 omitted; sweep is exploratory threshold search on Scenario A.

## Files I can attach

- Tree listing or zip: `phase_4_logs`, new hybrid logs, sweep folder, and optional `logs/phase4_run_summary.csv` from Kaggle.

Please be explicit about **copy vs symlink**, **Windows paths**, and **double-check run_tag naming** so nothing lands in “ungrouped.”