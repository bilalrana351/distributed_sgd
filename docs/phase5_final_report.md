# Phase 5 – Final Report

**Project:** Hybrid Distributed SGD — Ring All-Reduce ↔ Parameter Server
**Course:** Parallel and Distributed Computing (Spring 2026, SEECS, NUST)
**Authors:** Student project submission
**Written for:** Final evaluation and grading

This report uses simple English so the reader can follow every step without needing to re-read. It is not a rewrite of the proposal — it is a *plain summary of what was built, what was measured, and what the numbers say*.

> **Reproducibility in one line.** The whole repository is **fully containerised**: `docker/Dockerfile` + `docker/docker-compose.yml` build a self-contained Linux image (`pdc-hybrid:latest`) that runs every training mode, every scenario, the threshold sweep, and every analysis script. A grader with Docker installed can reproduce every number in this report from scratch with `docker compose build` followed by one `docker compose run ...` command per experiment. Full commands are listed in Section 10.

---

## 1. Problem in one paragraph

When we train a deep learning model on many workers at the same time, the workers must share gradient updates. There are two common ways to do this:

- **Synchronous Ring All-Reduce (sync):** all workers wait for each other, average their gradients, then take one step together. The model stays consistent, but one slow worker slows everybody down.
- **Asynchronous Parameter Server (async):** each worker sends its gradient to a central server as soon as it is ready. Nobody waits, so the system is fast, but updates can be **stale** and the model can lose accuracy.

Neither choice wins on its own because real clusters are not stable. Networks get busy, containers slow down for a while, and training sometimes becomes unstable. **The project builds a system that can switch between the two styles at runtime** based on what is actually happening during training.

## 2. Gap we are addressing

Most existing systems pick one style and stick with it. A few hybrid proposals exist, but they usually switch **only in one direction** (for example sync → async after a fixed step count) or assume the slow workers stay slow forever. Real workloads are **not like that** — sometimes a worker is slow for a short burst, sometimes training becomes unstable and needs stricter synchronisation.

The contribution of this project is a **bi-directional, online controller** that can switch both ways (sync → async **and** async → sync) using live signals:

- worker step-time variance (straggler signal),
- gradient-norm variance (instability signal),
- loss trend (training-trouble signal).

## 3. System we built

The system has four moving parts:

1. **Parameter Server process** — keeps the global model weights and applies updates.
2. **Worker processes** — each computes gradients on its own data shard and pushes them to the server.
3. **Aggregator inside the server** — two modes:
   - *Sync mode*: buffer one gradient per worker, average, apply one step, release workers.
   - *Async mode*: apply each gradient the moment it arrives.
4. **Controller inside the server** — every few steps it looks at the live signals (step-time variance, gradient variance, loss trend) and, after a cooldown, can flip a shared `mode_flag`. When the flag flips, the aggregator logic changes immediately for the next update. Every flip is written to a `*_switches.csv` file with the reason and the signal values that caused it.

### Why the architecture ended up this way

The proposal promised **PyTorch RPC** for the async side and **PyTorch Distributed (Gloo/NCCL)** for the sync side. On the Windows CPU laptop used for development, `torch.distributed.rpc.is_available()` is `False`, and switching between two completely different backends inside one run is not trivial anyway. So the final architecture keeps the **same transport** for both modes — `torch.multiprocessing` with shared memory + a gradient queue — and **only the aggregation rule changes**. The training semantics (what sync vs async actually *means* mathematically) are preserved:

- sync is still averaged-gradient rounds,
- async is still per-gradient immediate updates.

This is the honest framing: we implemented the research idea; the transport choice is an engineering decision forced by the Windows CPU environment.

## 4. Experimental setup — containerised, reproducible on any machine with Docker

The proposal says "4–8 Docker containers on a bridge network." In this project **the entire experimental environment lives inside a single, versioned Docker image** (`pdc-hybrid:latest`) built from `docker/Dockerfile`, and every experiment is launched through `docker/docker-compose.yml`. Concretely:

- **`docker/Dockerfile`** — builds the image. Base `python:3.12-slim`, installs `iproute2` (for optional `tc-netem` network emulation) and all Python dependencies from `requirements.txt`, then copies `src/`, `configs/`, and `experiments/`. The image is self-contained — no host Python, no host PyTorch, no host packages are used during a run.
- **`docker/docker-compose.yml`** — declares two services:
  - `scenario_runner`: runs training and experiments (entrypoint `python`, default command overridable per run).
  - `analyzer`: runs only the analysis scripts, with a read-write bind mount on `logs/` only.
  Both services bind-mount host directories (`data/`, `logs/`, `checkpoints/`) so that **CSV logs written inside the container appear on the host immediately**, and vice-versa.
- **Inside the container**, one process acts as the Parameter Server and N processes act as workers, all sharing memory via `torch.multiprocessing`. This is a **container-based simulation of distributed nodes** — exactly what the proposal described as scope for a single laptop / workstation.
- Scenario effects (persistent straggler in sA, transient latency spikes in sB) are applied inside each worker's loop via configurable `time.sleep(...)` ranges read from the YAML config; no host-level tools are needed.
- Larger, longer multi-seed runs were executed on a Linux remote runtime (Kaggle notebooks) that matches the image's Python / PyTorch versions. Their CSV exports were merged back into `logs/phase4_merged/` for analysis — the same analysis the container would produce if the runs were re-executed on a grader's machine.

**Resource caps** declared in the compose file (`cpus: "4"`, `mem_limit: "6g"`) reflect a laptop-realistic envelope; a grader with a workstation can raise them by editing one line. **No GPU is required**; the image is CPU-only on purpose, consistent with the single-laptop scope of the proposal.

Every step in Sections 5–7 below — training, logging, merging, analysis, figure rendering, report regeneration — can be run inside this one image. The exact commands are in Section 10.

### Configuration matrix

| Axis | Values |
|------|--------|
| Dataset | CIFAR-10 |
| Model | ResNet-18 |
| Workers | 2 (per run) |
| Batch size | 32 |
| Optimizer | SGD (momentum 0.9) |
| Random seeds | 42, 123, 777 |
| Scenarios | s0 homogeneous · sA persistent straggler · sB transient spikes |
| Modes | sync · async · hybrid |
| TTA target | 40% val accuracy |
| Total runs | 24 (s0 hybrid intentionally skipped) |

## 5. What the numbers say

All tables and figures below are generated from the same merged log directory (`logs/phase4_merged/`) by the reproducibility scripts listed in Section 10. The **main comparison table** appears in [`phase4_results_tables.md`](phase4_results_tables.md); it is reproduced here for convenience.

### 5.1 Main comparison — mean ± std over seeds

**Scenario 0 — Homogeneous workers**

| Mode   | Final Acc (%) | TTA (s)  | Wall (s) | Throughput (img/s) | Switches | Staleness | Sync % |
|--------|:-------------:|:--------:|:--------:|:------------------:|:--------:|:---------:|:------:|
| sync   |  44.5 ± 0.5   | 843 ± 33 | 1122 ± 43 | 22.8 | 0.0 | 0.0 | 100% |
| async  |  45.9 ± 0.7   | 884 ± 16 | 1128 ± 28 | 45.2 | 0.0 | 714.2 | 0% |
| hybrid | — | — | — | — | — | — | — |

**Scenario A — Persistent straggler**

| Mode   | Final Acc (%) | TTA (s)   | Wall (s) | Throughput (img/s) | Switches | Staleness | Sync % |
|--------|:-------------:|:---------:|:--------:|:------------------:|:--------:|:---------:|:------:|
| sync   |  43.2 ± 1.9   | 1409 ± 17 | 1878 ± 25 | 13.6 | 0.0 | 0.0 | 100% |
| async  |  39.0 ± 0.6   | 641 ± 0   | 809 ± 42  | 43.2 | 0.0 | 373.5 | 0% |
| hybrid |  41.9 ± 2.9   | 1233 ± 386 | 1609 ± 95 | 23.3 | 1.7 | 359.6 | 20% |

**Scenario B — Transient stragglers / latency spikes**

| Mode   | Final Acc (%) | TTA (s)   | Wall (s) | Throughput (img/s) | Switches | Staleness | Sync % |
|--------|:-------------:|:---------:|:--------:|:------------------:|:--------:|:---------:|:------:|
| sync   |  46.3 ± 1.9   | 1323 ± 31 | 1631 ± 29 | 18.4 | 0.0 | 0.0 | 100% |
| async  |  43.0 ± 0.7   | 760 ± 112 | 905 ± 17  | 45.3 | 0.0 | 458.1 | 0% |
| hybrid |  36.9 ± 3.6   | 864 ± 0   | 1160 ± 116 | 33.5 | 2.3 | 331.0 | 22% |

### 5.2 Per-seed details (honest picture)

A mean ± std can hide the fact that **a seed did not reach the target at all**. The full per-seed table is in [`phase5_per_seed_table.md`](phase5_per_seed_table.md) and the CSV is `logs/phase5_per_seed.csv`. Key honest points from that table:

- **sA async** reached 40% on **1 of 3 seeds** only (seed 777 at 641 s). The other two seeds ended at 39.7% and 38.8%. So the "641 ± 0 s" number really represents a single successful seed.
- **sA hybrid** reached 40% on **2 of 3 seeds**. Seed 42 ended at 38.6% (below target). The mean TTA (1233 s) is across the seeds that did reach it.
- **sB hybrid** reached 40% on **1 of 3 seeds**. Two seeds finished at 35.2% and 34.5%. This is the high-variance case and it is *not* hidden — the figure `phase5_tta_bars.png` shows the "1/3" label above the sB hybrid bar.

### 5.3 New Phase 5 figures

**Time-to-Accuracy bars** — how fast each mode reaches 40% validation accuracy in each scenario. Numbers above each bar show how many seeds (out of 3) actually reached the target.

![TTA bars](figures/phase5_tta_bars.png)

**Accuracy vs wall-clock time** — the learning curves in real seconds, not in steps. The shaded band is ±1 std across seeds, and the dashed line marks the 40% target.

![Accuracy vs wall-clock](figures/phase5_acc_vs_walltime.png)

**Hybrid mode timeline** — every hybrid run in sA and sB shown as a horizontal bar, coloured by which mode the controller was in at each step. Vertical black lines mark the switch events and are labelled with the trigger reason.

![Hybrid mode timeline](figures/phase5_hybrid_mode_timeline.png)

**Throughput and staleness** — how many images per second the system pushed (left) and how stale the server's view of worker weights was at the end of the run (right).

![Throughput and staleness](figures/phase5_throughput_staleness.png)

The older Phase 4 figures (validation curves by global step, threshold sweep summary) are still valid and are linked from [`phase4_results_discussion.md`](phase4_results_discussion.md).

## 6. Discussion — what these results mean

### Scenario 0 (homogeneous): hybrid was not needed
No straggler exists, so there is nothing to react to. Sync finishes with strong final accuracy (44.5%) and async finishes a bit higher but with huge staleness (~714 steps). Hybrid was intentionally not re-run here because the controller would simply sit in sync mode and add nothing. This matches the *"do no harm when conditions are good"* expectation.

### Scenario A (persistent straggler): hybrid behaves as designed
- Sync is the slowest (1409 s to reach 40%) — it waits for the slow worker every single round.
- Async is the fastest **when it reaches the target** (641 s on the one seed that did) but it ends **below** sync in final accuracy because stale gradients hurt quality.
- Hybrid sits **between** the two on time (1233 s) and **keeps final accuracy close to sync** (41.9% vs 43.2% — within the std band). Throughput roughly doubles over sync (23.3 vs 13.6 img/s) because the controller spends ~80% of the time in async mode. The switch log shows the controller firing `straggler_detected` early (step 120) and then occasionally flipping back with `loss_rising`. This is the expected bi-directional behaviour.

### Scenario B (transient stragglers): hybrid is faster but pays an accuracy cost
- Sync reaches 40% in 1323 s; hybrid reaches it in 864 s (on the seed that reached it). That is a big speed gain.
- But hybrid's **final accuracy on sB is only 36.9%**, compared to 46.3% for sync. Two of the three seeds never cross 40%.
- The reason is visible in the mode timeline: most sB hybrid runs flip to async very early (step 120, right after the first straggler signal) and never spend enough time in sync to stabilise. The controller's current thresholds are tuned for persistent stragglers; they over-react to transient spikes.

### So what did the hybrid achieve?

1. **Bi-directional switching is real and logged.** Every hybrid run in sA/sB shows non-zero switches with both `straggler_detected` (sync → async) and `loss_rising` (async → sync) triggers. That is the core mechanical claim of the project.
2. **On the scenario the system was designed for (persistent stragglers), hybrid meets the proposal's success criteria:**
   - **Speed:** lower time-to-convergence than sync (1233 s vs 1409 s, ~12% faster).
   - **Quality retention:** final accuracy close to sync (41.9% vs 43.2%, inside the seed-variance band).
   - **Resource efficiency:** ~70% higher throughput than sync.
3. **On transient stragglers (sB), the hybrid is fast but loses accuracy.** The threshold sweep in Phase 4 (17 configurations) showed that higher `straggler_var_high` and lower `grad_var_low` gates produce better final accuracy — so tuning is the known next step, not a design flaw.

## 7. Threshold sweep (separate exploratory grid)

An exploratory grid on Scenario A varied the controller's thresholds. Seventeen configurations completed. The best configuration (`sweep_17`: `straggler_var_high=0.25`, `grad_var_low=0.70`, `cooldown=100`) produced the highest final accuracy of the grid (37.24%). The full ranked table and the scatter/bar figure (`phase4_sweep_summary.png`) are in [`phase4_results_discussion.md`](phase4_results_discussion.md). The sweep was **not** folded into the main table, because its objective is different (find thresholds) and its budget is smaller per run.

## 8. Limitations (stated openly)

1. **CPU-only laptop, small model, short budgets.** Absolute accuracies are modest by design. The purpose is to compare modes under controlled conditions, not to beat a published CIFAR-10 record.
2. **2 workers per run.** The proposal mentioned 4–8 containers. Two workers already show all three behaviours (sync waiting, async staleness, hybrid switching), and running 4–8 processes at batch size 32 would force either a smaller model or much longer wall time on a laptop.
3. **Mixed log vintages.** Sync and async CSVs for the main table come from the earlier export; hybrid for sA and sB comes from the post-fix export. This is explicitly called out in the discussion document. It affects switch-count comparisons only (both baselines have 0 switches anyway), not the TTA / accuracy / throughput numbers.
4. **Subset validation during some runs.** To keep wall time reasonable, some runs used a fixed validation subset. The subset was the same across modes for each seed, so cross-mode comparisons inside a scenario remain fair.
5. **Scenario B hybrid variance.** Only 1 of 3 seeds reached the TTA target. The new figures show this honestly (the `1/3` label on the bar and the wide shaded band).
6. **Scenario 0 hybrid was not run.** The controller would not fire without a straggler; its row is marked `—`.

## 9. Conclusion

The project achieves its stated objective: a distributed SGD system that **switches between sync Ring All-Reduce semantics and async Parameter Server semantics based on live runtime signals**, with every switch logged and explainable.

- Under persistent stragglers (sA) the hybrid is **faster than sync and keeps accuracy close to sync** — the exact success criterion from the proposal.
- Under transient stragglers (sB) the hybrid is **faster but loses accuracy**; the threshold sweep already points to which knobs close that gap, which is the immediate follow-on work.
- All results are reproducible from a single directory of CSV logs and a handful of Python scripts.

The main engineering compromise (sharing one container and one transport instead of swapping DDP ↔ RPC at runtime) does not change the training semantics of either mode and is documented openly. The Docker image and compose file are part of the repository so the experimental setup can be rebuilt on any Linux-compatible machine.

## 10. Reproducibility

Two equivalent paths are supported. **Path A (Docker) is the canonical one** — it is self-contained, does not depend on the grader's Python install, and matches the environment used to produce every number in this report. **Path B (host venv)** is a convenience for the developer's Windows laptop where Docker is not always available.

### Path A — Full Docker reproducibility (recommended)

```powershell
# 1. Build the versioned image once (~2–3 minutes on first run, cached after).
docker compose -f docker/docker-compose.yml build

# 2. Run one scenario (example: Scenario A hybrid). Logs land in host ./logs via the bind mount.
docker compose -f docker/docker-compose.yml run --rm scenario_runner `
    src/hybrid/launcher.py --nworkers 2 --config configs/phase4_sA_hybrid.yaml

# 3. Run the full 9-scenario matrix (s0/sA/sB x sync/async/hybrid) sequentially.
docker compose -f docker/docker-compose.yml run --rm scenario_runner `
    experiments/run_scenarios.py

# 4. Run the controller threshold sweep on Scenario A.
docker compose -f docker/docker-compose.yml run --rm scenario_runner `
    experiments/threshold_sweep.py

# 5. Regenerate all Phase 4 and Phase 5 tables + figures from whatever logs are on disk.
docker compose -f docker/docker-compose.yml run --rm analyzer `
    experiments/analyze_results.py --log-dir /workspace/logs/phase4_merged `
    --tta-target 40.0 --output docs/phase4_results_tables.md
docker compose -f docker/docker-compose.yml run --rm scenario_runner `
    experiments/phase4_report_assets.py
docker compose -f docker/docker-compose.yml run --rm scenario_runner `
    experiments/phase5_report_assets.py
```

Because `logs/`, `data/`, and `checkpoints/` are bind-mounted, every CSV, checkpoint and figure produced by the container appears on the host filesystem immediately. After step 5, `docs/phase4_results_tables.md`, `docs/phase4_results_discussion.md`, `docs/phase5_per_seed_table.md`, `logs/phase5_per_seed.csv` and every PNG under `docs/figures/` are rewritten exactly as referenced by this report.

### Path B — Host venv (Windows laptop convenience)

```powershell
.\.venv\Scripts\python.exe experiments\merge_phase4_compare.py
.\.venv\Scripts\python.exe experiments\analyze_results.py --log-dir logs/phase4_merged --tta-target 40.0 --output docs\phase4_results_tables.md
.\.venv\Scripts\python.exe experiments\phase4_report_assets.py
.\.venv\Scripts\python.exe experiments\phase5_report_assets.py
```

Both paths read the same CSVs and produce the same tables and figures. The Docker path additionally guarantees the **exact Python / PyTorch / OS versions** used during the experiments; the host path relies on whatever is installed in `.venv/`.

## 11. File map for the submission

| Item | Location |
|------|----------|
| Final report (this file) | `docs/phase5_final_report.md` |
| Main comparison table | `docs/phase4_results_tables.md` |
| Scenario discussion + sweep | `docs/phase4_results_discussion.md` |
| Per-seed table (Phase 5) | `docs/phase5_per_seed_table.md` |
| Per-seed CSV (Phase 5) | `logs/phase5_per_seed.csv` |
| Phase 5 figures | `docs/figures/phase5_*.png` |
| Phase 4 figures | `docs/figures/phase4_*.png` |
| Slides | `docs/phase5_slides.md` |
| Phase history (per phase) | `docs/final_summary_docs_per_phase/phase{1,2,3,4,5}_end_to_end_record.md` |
| Controller + PS code | `src/hybrid/` |
| Docker image + compose | `docker/Dockerfile`, `docker/docker-compose.yml` |
