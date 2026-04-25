# Phase 4 End-to-End Record (Simple + Detailed)

## 1) What this phase was about

Phase 4 goal was to **evaluate the hybrid parameter-server system under proposal-style scenarios**, comparing three training modes:

- **Sync** — averaged-gradient rounds (workers align each logical step).  
- **Async** — immediate per-gradient updates (Phase 2 style).  
- **Hybrid** — controller-driven switching between sync and async inside the same run.

Runs were organized by **scenario**:

| Scenario | Meaning (high level) |
|----------|----------------------|
| **s0** | Homogeneous workers (no artificial straggler). |
| **sA** | Persistent straggler (one worker systematically slower). |
| **sB** | Transient straggler / latency spike pattern. |

For each scenario and mode, experiments used **multiple random seeds** (typically 42, 123, 777) so tables can report **mean ± std** where applicable.

Phase 4 also included an **exploratory threshold sweep** on Scenario A: a grid over controller knobs (`straggler_var_high`, `grad_var_low`, `cooldown_steps`) to see how sensitive switching and final accuracy are. That sweep is **documented separately** from the main scenario comparison table.

This phase is the bridge from Phase 3’s **host-side hybrid proof** to **Phase 5** (final evaluation narrative, any extra plots, and write-up polish).

---

## 2) Important setup reality (logs, exports, and merge policy)

### Where runs actually executed

Full Phase 4 scenario runs were completed on **remote compute** (e.g. Kaggle notebooks), not only on the Windows laptop. Artifacts arrived as **folder exports / zips** under:

- `phase_4_logs/zip/...` — older batch (sync/async for all scenarios; hybrid rows may reflect **pre-fix** switch logging in places).  
- `threshold_folders/...` — newer batch (**post-fix** hybrid for Scenarios A and B; sweep CSVs where present).

### Run tag convention (required for the analyzer)

The results tooling expects PS logs named:

`hybrid_resnet18_cifar10_nw2_<run_tag>_ps.csv`

with **`run_tag`** matching:

`p4_<scenario>_<mode>_s<seed>`

Examples: `p4_sA_sync_s42`, `p4_sB_hybrid_s123`, `p4_s0_async_s777`.

The analyzer scans **one flat directory** (non-recursive). Nested export folders must be **merged** before analysis.

### Deliberate merge policy for the “official” comparison table

To make the **main markdown table** trustworthy:

- **Sync and async** cells use the **historical** exports under `phase_4_logs/zip` (all scenarios, all seeds).  
- **Hybrid** cells for **sA and sB** use the **new** hybrid logs from `threshold_folders` (post-fix switch behavior).  
- **Scenario 0 hybrid** was **not** re-run for this table: the hybrid row is intentionally **empty (—)** in the published table.  
- **Threshold sweep** `*_ps.csv` files (`sweep_*` run tags) are **not** mixed into the main scenario merge (they would not group into s0/sA/sB mode rows).

Duplicate basenames can appear when one export folder accidentally contains another scenario’s files (e.g. sB logs inside an s0 seed folder). The merge script **deduplicates by basename** and prefers paths whose folder names match the scenario (`artifacts_sB`, `chunkB`, etc.).

---

## 3) What was implemented in Phase 4 (repo-side)

### Configs (per scenario × mode)

Representative YAML entries under `configs/`:

- `phase4_s0_sync.yaml`, `phase4_s0_async.yaml`, `phase4_s0_hybrid.yaml`  
- `phase4_sA_sync.yaml`, `phase4_sA_async.yaml`, `phase4_sA_hybrid.yaml`, `phase4_sA_hybrid_smoke.yaml`  
- `phase4_sB_sync.yaml`, `phase4_sB_async.yaml`, `phase4_sB_hybrid.yaml`  
- `phase4_threshold_sweep_base.yaml` — base for `experiments/threshold_sweep.py`

### Experiments and documentation scripts

| Script | Purpose |
|--------|---------|
| `experiments/analyze_results.py` | Discovers `*_ps.csv` in a flat dir; computes final acc, TTA @ target, wall time, throughput, switch count, staleness, sync %; emits markdown tables. |
| `experiments/merge_phase4_compare.py` | Builds `logs/phase4_merged/` from old sync/async + new sA/sB hybrid per policy above. |
| `experiments/merge_phase4_logs.py` | Generic “copy all CSVs from artifact children” merge (first basename wins); useful for quick full exports. |
| `experiments/threshold_sweep.py` | Runs the Scenario A grid locally (writes `logs/phase4_sweep_results.csv` when executed). |
| `experiments/phase4_report_assets.py` | Summarizes sweep logs from disk into `logs/phase4_sweep_results_from_logs.csv`; generates figures and [phase4_results_discussion.md](../phase4_results_discussion.md). |

### Generated report artifacts (this project state)

| Artifact | Path |
|----------|------|
| Merged logs for main table | `logs/phase4_merged/` |
| Comparison + switch-events markdown | [`../phase4_results_tables.md`](../phase4_results_tables.md) |
| Discussion, sweep table, limitations, figure links | [`../phase4_results_discussion.md`](../phase4_results_discussion.md) |
| Validation curves figure | [`../figures/phase4_val_acc_curves.png`](../figures/phase4_val_acc_curves.png) |
| Sweep scatter + ranked bars | [`../figures/phase4_sweep_summary.png`](../figures/phase4_sweep_summary.png) |
| Sweep summary from exported CSVs | `logs/phase4_sweep_results_from_logs.csv` |

Training **code** remains centered on `src/hybrid/` (Phase 3); Phase 4 is primarily **experiment matrix + logging + analysis**.

---

## 4) How to reproduce the analysis locally (Windows paths)

From the repo root (PowerShell):

```powershell
.\.venv\Scripts\python.exe experiments\merge_phase4_compare.py
.\.venv\Scripts\python.exe experiments\analyze_results.py `
  --log-dir logs/phase4_merged `
  --tta-target 40.0 `
  --output docs\phase4_results_tables.md
.\.venv\Scripts\python.exe experiments\phase4_report_assets.py
```

- Use **copies** (the merge script uses `shutil.copy2`). Symlinks are optional.  
- If raw exports move, adjust `--old-root` / `--new-root` in `merge_phase4_compare.py` or edit defaults in the script.

---

## 5) Raw logs layout (conceptual)

Each **run** normally produces a family of CSVs:

| Suffix | Content |
|--------|---------|
| `_ps.csv` | Global step, wall time, mode, **val** loss/acc, staleness aggregates, throughput, scenario id. |
| `_staleness.csv` | Per-worker staleness samples + `current_mode` (used for **sync %**). |
| `_switches.csv` | Mode transitions: step, old/new mode, trigger reason, controller signals. |
| `_worker0.csv`, `_worker1.csv` | Local train loss/acc, step time (optional for the summary table). |

Optional per-export: `phase4_run_summary.csv` (scenario/mode/seed/status) — **not** consumed by `analyze_results.py`.

---

## 6) Results summary (aggregated table, TTA target 40% val accuracy)

The **authoritative** numbers (including the **mode switch events** table) live in [phase4_results_tables.md](../phase4_results_tables.md).

Below is a snapshot of the scenario summary for quick reading (mean ± std across seeds; hybrid s0 omitted by design).

### Scenario 0 – Homogeneous

| Mode | Final Acc (%) | TTA (s) | Wall (s) | Throughput (img/s) | Switches | Staleness | Sync % |
|------|----------------|---------|----------|-------------------|----------|-----------|--------|
| sync | 44.5 ± 0.5 | 843 ± 33 | 1122 ± 43 | 22.8 | 0.0 | 0.0 | 100% |
| async | 45.9 ± 0.7 | 884 ± 16 | 1128 ± 28 | 45.2 | 0.0 | 714.2 | 0% |
| hybrid | — | — | — | — | — | — | — |

### Scenario A – Persistent straggler

| Mode | Final Acc (%) | TTA (s) | Wall (s) | Throughput (img/s) | Switches | Staleness | Sync % |
|------|----------------|---------|----------|-------------------|----------|-----------|--------|
| sync | 43.2 ± 1.9 | 1409 ± 17 | 1878 ± 25 | 13.6 | 0.0 | 0.0 | 100% |
| async | 39.0 ± 0.6 | 641 ± 0 | 809 ± 42 | 43.2 | 0.0 | 373.5 | 0% |
| hybrid | 41.9 ± 2.9 | 1233 ± 386 | 1609 ± 95 | 23.3 | 1.7 | 359.6 | 20% |

### Scenario B – Transient straggler

| Mode | Final Acc (%) | TTA (s) | Wall (s) | Throughput (img/s) | Switches | Staleness | Sync % |
|------|----------------|---------|----------|-------------------|----------|-----------|--------|
| sync | 46.3 ± 1.9 | 1323 ± 31 | 1631 ± 29 | 18.4 | 0.0 | 0.0 | 100% |
| async | 43.0 ± 0.7 | 760 ± 112 | 905 ± 17 | 45.3 | 0.0 | 458.1 | 0% |
| hybrid | 36.9 ± 3.6 | 864 ± 0 | 1160 ± 116 | 33.5 | 2.3 | 331.0 | 22% |

**Mode switch events** (hybrid only, post-fix exports): full table in [phase4_results_tables.md](../phase4_results_tables.md). Patterns include **straggler_detected** (sync → async) and **loss_rising** (async → sync) on some seeds/scenarios.

---

## 7) Threshold sweep (separate from the main table)

- **17** completed sweep configurations were recovered from `threshold_folders` into `logs/phase4_sweep_results_from_logs.csv`.  
- Interpretation, ranked table, figures, and caveats: [phase4_results_discussion.md](../phase4_results_discussion.md).  
- The full Cartesian grid in `experiments/threshold_sweep.py` can be larger than what was exported; treat the sweep as **exploratory**, not a second “official” scenario matrix.

---

## 8) Simple interpretation (plain English)

1. **End-to-end comparison machinery works.**  
   You can merge exports, enforce naming, and regenerate tables and figures from one command sequence.

2. **Baselines behave as expected qualitatively.**  
   Sync shows **100% sync-time** and **zero staleness** in the aggregate table; async shows **high staleness** and **no mode switches** (fixed mode).

3. **Hybrid is active where intended.**  
   sA/sB hybrid rows show **non-zero switch counts** and **intermediate sync %**, with interpretable triggers in the switch-events table.

4. **Scenario B hybrid shows high variance across seeds** (wide std on final acc and wall time).  
   That is a reporting fact: highlight variance and per-run detail from `analyze_results.py` stdout or future per-seed tables.

5. **Do not over-interpret async vs sync final accuracy on sA without context.**  
   Async can hit the **TTA target** earlier while ending **below** sync/hybrid final acc in this setup—trade-off to describe, not necessarily a logging bug.

6. **Mixed log vintages for sync/async vs hybrid (sA/sB)** should be stated in any formal write-up: hybrid metrics come from **post-fix** logging; sync/async from the **earlier** export batch.

---

## 9) Comparison perspective vs Phase 3

| Aspect | Phase 3 (host proof) | Phase 4 (scenario matrix) |
|--------|----------------------|-----------------------------|
| Goal | Show hybrid **can** switch and log reasons | Compare **three modes** under **three scenarios**, multi-seed |
| Scale | Short demos + smoke tests | Full experiment grid + optional sweep |
| Evidence | Single-machine logs, checkpoint switch history | Merged CSVs, markdown tables, plots, sweep CSV |
| s0 / sA / sB | Introduced in planning / configs | **Executed** and summarized |

Phase 3 proves **mechanism**; Phase 4 supplies **numbers under controlled narratives** for the report.

---

## 10) What improved vs earlier-phase “pain points”

| Earlier issue | Phase 4 handling |
|---------------|------------------|
| Nested Kaggle exports vs flat analyzer | `merge_phase4_compare.py` + `logs/phase4_merged/` |
| Wrong duplicate CSV basename | Scenario-aware path preference when deduplicating |
| Sweep vs scenario confusion | Separate markdown + CSV; sweep run tags excluded from main merge |
| Figures for the report | `phase4_report_assets.py` + `docs/figures/` |

---

## 11) Phase 5 direction (next phase)

Phase 5 should focus on **evaluation and reporting**, for example:

1. Freeze the **story** in the write-up: what hybrid is trying to optimize (latency vs stability) in each scenario.  
2. Add any required **extra plots** (per-seed bars, wall-clock CDFs) using the same merged logs.  
3. Optional: one **clean re-export** where sync/async/hybrid for sA/sB all share the **same** code revision—if reviewers demand perfectly matched vintages.  
4. Tie conclusions back to the **submitted proposal** (scenarios + controller thresholds).

---

## 12) Final statement

Phase 4 is successful for its intended scope: **scenario-structured experiments are run, logs are merged and analyzed reproducibly, hybrid behavior is visible in both tables and switch logs, and the threshold sweep is documented as a separate exploratory block.**  

You can proceed to **Phase 5** to finalize narrative, any remaining figures, and submission-ready polish.
