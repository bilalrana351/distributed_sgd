# Phase 5 End-to-End Record (Simple + Detailed)

## 1) What this phase was about

Phase 5 is **final evaluation and reporting**. No new training runs were required.
Inputs: everything Phase 4 already produced (merged CSV logs, main comparison table, scenario discussion, validation-curve figure, threshold-sweep figure).
Outputs: a complete, submission-ready report, a presentation deck, and a small number of new figures and tables that make the evaluation story more honest and easier to follow.

The principles that guided this phase:

1. **No new code inside `src/`.** The training stack is frozen at its Phase 3 / 4 state.
2. **One generator script** for every new Phase 5 asset, so each figure and each table can be regenerated from the merged logs with a single command.
3. **Simple English throughout.** The report is meant to be readable by a professor evaluating the project, not to impress with jargon.
4. **Honesty about variance.** Where a mean ± std hides a seed that failed to reach the target, that is stated out loud and shown on the figures (annotated labels like `1/3`).

---

## 2) Inputs Phase 5 consumed

| Input | Path | Role |
|-------|------|------|
| Merged Phase 4 logs | `logs/phase4_merged/` (24 × 5 CSVs) | Source for every metric, curve and bar |
| Main comparison table | `docs/phase4_results_tables.md` | Reused verbatim in the final report |
| Scenario discussion + sweep | `docs/phase4_results_discussion.md` | Referenced from report and slides |
| Phase 4 figures | `docs/figures/phase4_val_acc_curves.png`, `docs/figures/phase4_sweep_summary.png` | Kept as-is |
| Controller code | `src/hybrid/` | Only referenced, not changed |
| Docker artefacts | `docker/Dockerfile`, `docker/docker-compose.yml` | Referenced in report §4 and §10 |
| Per-phase records | `docs/final_summary_docs_per_phase/phase{1,2,3,4}_end_to_end_record.md` | History cited in the report |

---

## 3) What Phase 5 produced

| Artefact | Path | Purpose |
|----------|------|---------|
| Per-seed CSV | `logs/phase5_per_seed.csv` | 24 rows, one per run, all scalar metrics |
| Per-seed markdown table | `docs/phase5_per_seed_table.md` | Same data, readable by humans |
| TTA bar figure | `docs/figures/phase5_tta_bars.png` | Mean ± std bars per scenario, annotated with "seeds that reached target" |
| Accuracy vs wall-clock | `docs/figures/phase5_acc_vs_walltime.png` | 3 subplots (one per scenario), shaded std bands |
| Hybrid mode timeline | `docs/figures/phase5_hybrid_mode_timeline.png` | Mode regions per hybrid run, with switch-event reasons |
| Throughput + staleness | `docs/figures/phase5_throughput_staleness.png` | Grouped bars for resource signals |
| Asset generator | `experiments/phase5_report_assets.py` | Regenerates all of the above from `logs/phase4_merged/` |
| Final report | `docs/phase5_final_report.md` | Complete narrative in simple English |
| Presentation deck | `docs/phase5_slides.md` | Marp-compatible 12-slide deck |
| End-to-end record (this file) | `docs/final_summary_docs_per_phase/phase5_end_to_end_record.md` | History + how to reproduce |

---

## 4) How Phase 5 is reproduced

There are two equivalent entry points. **Path A (Docker)** is the canonical one used in the report; Path B is a local convenience.

### Path A — Docker (recommended, runs on any OS with Docker)

```powershell
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm analyzer `
    experiments/analyze_results.py --log-dir /workspace/logs/phase4_merged `
    --tta-target 40.0 --output docs/phase4_results_tables.md
docker compose -f docker/docker-compose.yml run --rm scenario_runner `
    experiments/phase4_report_assets.py
docker compose -f docker/docker-compose.yml run --rm scenario_runner `
    experiments/phase5_report_assets.py
```

The bind mounts on `data/`, `logs/`, `checkpoints/` mean every output appears on the host filesystem immediately.

### Path B — Host venv (Windows laptop shortcut)

Assumes the repo root is the current directory and the project venv is at `.venv/`.

```powershell
.\.venv\Scripts\python.exe experiments\merge_phase4_compare.py
.\.venv\Scripts\python.exe experiments\analyze_results.py `
    --log-dir logs/phase4_merged `
    --tta-target 40.0 `
    --output docs\phase4_results_tables.md
.\.venv\Scripts\python.exe experiments\phase4_report_assets.py
.\.venv\Scripts\python.exe experiments\phase5_report_assets.py
```

After the four commands:

- `logs/phase5_per_seed.csv` and `docs/phase5_per_seed_table.md` are fresh.
- `docs/figures/phase5_*.png` (four files) are fresh.
- `docs/phase4_results_tables.md` and `docs/phase4_results_discussion.md` are fresh.
- `docs/phase5_final_report.md` and `docs/phase5_slides.md` are **static authored documents** that reference those outputs — they do not need to be regenerated unless the figures or tables changed in a way that makes the text inaccurate.

To render the slide deck to PDF or PPTX (optional):

```powershell
npm install -g @marp-team/marp-cli
marp docs\phase5_slides.md --pdf
marp docs\phase5_slides.md --pptx
```

If Node is not installed, open `docs/phase5_slides.md` inside VS Code with the **Marp for VS Code** extension enabled — export from the extension UI gives the same PDF / PPTX.

---

## 5) Decisions made during Phase 5 (design log)

### 5.1 Do not re-run any Phase 4 experiment

Phase 4 was executed across a mix of local Windows and Kaggle runtime, then merged. Re-running scenarios from scratch would cost many hours of laptop time, change the numbers slightly, and add no new information. The chosen approach is to **analyse what exists** and **document the mixed log vintage openly** (see `phase4_results_discussion.md` "Limitations and caveats" §1).

### 5.2 Add a per-seed table in addition to mean ± std

The mean ± std format is compact but can silently hide a seed that did not reach the TTA target. Showing each run on its own line (`docs/phase5_per_seed_table.md`) lets a reviewer see that:

- sA async reaches the target on 1 of 3 seeds,
- sA hybrid on 2 of 3,
- sB hybrid on 1 of 3.

The TTA bar figure in Phase 5 also prints "seeds reached / total seeds" above every bar so the same information is visible at a glance.

### 5.3 Docker is the canonical reproducibility path — not an afterthought

`docs/working_phases/phase5_final_evaluation_reporting.md` §3 states *"Phase 5 does not introduce new Docker work."* That means Phase 5 did not need to **extend** the Docker setup — but the setup itself, built in Phase 4, is the **primary reproducibility story** and is now explicitly framed that way in the report and slides.

Concretely, the repo ships:

- `docker/Dockerfile` — `python:3.12-slim` base + `iproute2` + all Python dependencies (`requirements.txt`) + project `src/`, `configs/`, `experiments/`. Self-contained, no host tooling needed at runtime.
- `docker/docker-compose.yml` — declares `scenario_runner` (for training, sweep, and report assets) and `analyzer` (read-write on `logs/` only). Both services bind-mount `data/`, `logs/`, `checkpoints/` so outputs appear on the host instantly.
- The compose file sets conservative resource caps (`cpus: "4"`, `mem_limit: "6g"`) so the laptop-scale proposal scope is preserved. A grader on a bigger machine can raise them with a one-line edit.

The **report Section 4** now explicitly calls the project "fully containerised" and the **report Section 10** lists **Path A (Docker, canonical)** before **Path B (host venv, convenience)**. The slide deck carries the same framing on slides 5 and 12. A grader can therefore reproduce every table, figure, and number from scratch by cloning the repo, running `docker compose build`, then running the one-liners printed in the report.

### 5.4 Slide format: Marp markdown

Advantages:

- readable as plain markdown inside the repo and on GitHub,
- renders to PDF and PPTX with one command,
- fits inside git diffs so slide history is visible.

Alternative considered: LaTeX Beamer. Rejected because the rest of the project documentation is markdown and consistency was more valuable than Beamer's typography.

### 5.5 Report voice

The user asked for simple English so a professor can read it without effort. The report therefore:

- opens with a one-paragraph plain-language problem statement,
- avoids Greek letters,
- uses concrete numbers from the tables whenever it makes a claim,
- mentions every limitation out loud instead of burying it.

---

## 6) Acceptance checklist for Phase 5

- [x] TTA bar chart with error bars — `phase5_tta_bars.png`.
- [x] Accuracy vs wall-clock curves per scenario with ±1 std band — `phase5_acc_vs_walltime.png`.
- [x] Hybrid mode regions on the time axis — `phase5_hybrid_mode_timeline.png`.
- [x] Throughput / staleness comparison bars — `phase5_throughput_staleness.png`.
- [x] Per-seed detail table — `docs/phase5_per_seed_table.md`, `logs/phase5_per_seed.csv`.
- [x] Written report with problem → gap → design → setup → results → conclusion — `docs/phase5_final_report.md`.
- [x] Slide deck (12 slides) with one slide on reproducibility — `docs/phase5_slides.md`.
- [x] Reproducibility block at the end of the report and this record.
- [x] Limitations section calling out sB hybrid variance, mixed log vintage, CPU budget.
- [x] One-command regeneration of every Phase 5 artefact — `experiments/phase5_report_assets.py`.

---

## 7) How this connects back to the proposal

| Proposal clause | Phase 5 evidence |
|-----------------|------------------|
| "Reduce Time-to-Convergence compared to sync" | sA hybrid 1233 s vs sync 1409 s (~12% faster); sB hybrid 864 s vs sync 1323 s on the seed that reached (~35% faster). |
| "Final validation accuracy close to sync" | sA hybrid 41.9% vs sync 43.2% — within seed variance. sB gap (36.9% vs 46.3%) documented as the known open case. |
| "Measurable reduction in idle time" | Throughput: sync sA 13.6 img/s → hybrid 23.3 img/s; sync sB 18.4 → hybrid 33.5. |
| "Controlled staleness without collapse" | Hybrid final staleness is lower than pure async on both scenarios (sA: 359 vs 373; sB: 331 vs 458). |
| "Container-based distributed simulation" | `docker/Dockerfile` + `docker/docker-compose.yml` build and run the same stack in a Linux container, with logs bind-mounted to the host. |
| "Tested under homogeneous / persistent / transient scenarios" | s0, sA, sB with three seeds each; per-seed table in `docs/phase5_per_seed_table.md`. |

---

## 8) What still could be done (post-Phase 5, optional)

1. **Apply the best sweep config (`svh=0.25, gvl=0.70, cd=100`) to Scenario B** for all three seeds, to close the accuracy gap. This would be a new Phase-4-style run, not a Phase 5 task.
2. **Run Scenario 0 hybrid** on all three seeds so the homogeneous row is filled in. Expected result: very few (possibly zero) switches and performance close to pure sync. Non-critical for the current submission.
3. **Repeat the full matrix on CIFAR-100 + MobileNetV2** (the larger half of the proposal's dataset/model combinations). Also outside the Phase 5 scope — would live in a follow-up.
4. **Replace the in-container shared-memory transport with a true multi-container `torch.distributed` setup (gloo)** once the project moves off the Windows-CPU constraint. This would let us literally use 4–8 containers on a bridge network, as the original proposal text suggested. The controller logic and the report narrative stay the same.

---

## 9) Final statement

Phase 5 closes the project. The repository contains, in reproducible form:

- the working hybrid algorithm (`src/hybrid/`),
- the container image to run it (`docker/`),
- the scenario-structured experiment results with multiple seeds (`logs/phase4_merged/`),
- every table and figure the report relies on (`docs/`, `docs/figures/`),
- a generator script for each non-trivial artefact (`experiments/*_report_assets.py`),
- the written report (`docs/phase5_final_report.md`) and the slide deck (`docs/phase5_slides.md`),
- end-to-end records for every phase (`docs/final_summary_docs_per_phase/`).

The project goal stated in `docs/project_context_simple.md` — *reduce time-to-convergence while keeping accuracy close to the sync baseline under changing node conditions* — is met for the persistent-straggler case, partially met for the transient case, and the path to closing the remaining gap is documented.
