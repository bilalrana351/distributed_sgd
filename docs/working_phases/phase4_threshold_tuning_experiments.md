## Phase 4 – Docker Cluster, Network Scenarios, and Threshold Tuning

### 0. How this phase relates to the proposal (and to Phases 1–3)

Your submitted proposal describes **a single machine running a container-based simulation**: **4–8 Docker containers** acting as distributed nodes, with **controlled network conditions** (e.g. latency / jitter), plus the hybrid controller.

- **Phases 1–3 (completed work)** are the **engineering baseline**: correct sync DDP, async parameter-server semantics, and hybrid switching, validated with **multi-process** runs on the host (Windows-friendly). That was necessary to build and debug the logic without fighting containers first.
- **Phase 4** is where the proposal's **experimental setup** is realized: the **same algorithms** run inside a **Docker container** (Linux, reproducible, archivable), three realistic scenarios are swept, the controller thresholds are tuned, and multi-seed CSV logs are collected for Phase 5 tables and plots.

Nothing in Phase 4 replaces Phases 1–3; it **instantiates** what the proposal promised for evaluation.

---

### 1. What was built for Phase 4

#### 1a. Framework extensions (Phases 3 → 4 upgrades)

**`src/hybrid/controller.py`**
- Added `straggler_var_low` threshold for the **ASYNC → SYNC** direction.
  - New rule: if `worker_time_var < straggler_var_low` (workers are uniform again), the controller switches back to sync.
  - This enables the **full bi-directional cycle** needed for Scenario B (transient heterogeneity).
- `evaluate()` now returns `(mode, switched, trigger_reason)` — the reason string (`straggler_detected`, `straggler_resolved`, `grad_instability`, `loss_rising`) is logged to `*_switches.csv` for traceability.

**`src/hybrid/train_hybrid.py`**
- `mode_lock` config field (`"none"` / `"sync"` / `"async"`): disables the controller for pure-mode baseline experiments using the exact same code path.
- `scenario_id` field added to all PS log rows.
- `throughput_img_per_sec` field added to PS log rows.
- `trigger_reason` field added to switch log rows.

#### 1b. Scenario configs (under `configs/`)

| Config | Scenario | Mode | Notes |
|--------|----------|------|-------|
| `phase4_s0_sync.yaml`   | Homogeneous  | sync (locked)  | reference baseline |
| `phase4_s0_async.yaml`  | Homogeneous  | async (locked) | reference baseline |
| `phase4_s0_hybrid.yaml` | Homogeneous  | hybrid (active)| should not switch — validates stability |
| `phase4_sA_sync.yaml`   | Persistent straggler | sync (locked)  | shows wait cost |
| `phase4_sA_async.yaml`  | Persistent straggler | async (locked) | shows staleness cost |
| `phase4_sA_hybrid.yaml` | Persistent straggler | hybrid (tuned) | SYNC→ASYNC at ~step 160 |
| `phase4_sB_sync.yaml`   | Transient straggler  | sync (locked)  | slow during straggler window |
| `phase4_sB_async.yaml`  | Transient straggler  | async (locked) | fast but stale after recovery |
| `phase4_sB_hybrid.yaml` | Transient straggler  | hybrid (tuned) | bi-directional cycle |
| `phase4_threshold_sweep_base.yaml` | Scenario A | hybrid | base for grid search |

Common settings across all Phase 4 configs:
- Dataset: CIFAR-10, Model: ResNet-18, Workers: 2
- Batch size: 32, LR: 0.01
- Straggler delay: 1.5 s/step (when enabled) on worker 1

#### 1c. Experiment scripts (under `experiments/`)

| Script | Purpose |
|--------|---------|
| `run_scenarios.py`   | Sequential orchestration of all 9 experiments (configurable subset + multi-seed) |
| `threshold_sweep.py` | Automated grid search: 12 threshold combinations × Scenario A |
| `analyze_results.py` | Parse all CSVs → comparison tables + switch event log |

#### 1d. Docker infrastructure (under `docker/`)

| File | Purpose |
|------|---------|
| `Dockerfile`          | Python 3.12 + PyTorch CPU image, copies `src/`, `configs/`, `experiments/` |
| `docker-compose.yml`  | `scenario_runner` and `analyzer` services; bind-mounts `data/`, `logs/`, `checkpoints/` |

---

### 2. Three scenarios (what each tests)

#### Scenario 0 – Homogeneous (no straggler)
- **Purpose**: Fairness check. Hybrid should track sync accuracy. Zero or minimal switches confirm the controller does not oscillate under stable conditions.
- **Expected hybrid result**: `total_switches = 0`, performance ≈ sync baseline.

#### Scenario A – Persistent Straggler
- **Purpose**: Worker 1 adds 1.5 s/step for the full run. This is the regime where sync is worst (every round waits for the slow worker) and async shines (fast worker dominates throughput).
- **Expected hybrid result**: One SYNC→ASYNC switch within the first ~200 steps. After the switch, wall-clock time tracks async baseline while accuracy stays close to sync.

#### Scenario B – Transient Heterogeneity
- **Purpose**: Worker 1 adds 1.5 s/step from step 100 to step 500, then recovers. This stress-tests the **bi-directional** controller.
- **Expected hybrid result**: Switch 1 (SYNC→ASYNC, `straggler_detected`) around step 250, switch 2 (ASYNC→SYNC, `straggler_resolved`) around step 650. Final accuracy benefits from the sync phase at the end.

---

### 3. Threshold tuning logic

The controller has five key thresholds:

| Threshold | Controls | Direction |
|-----------|---------|-----------|
| `straggler_var_high` | How sensitive the straggler detector is | SYNC→ASYNC |
| `grad_var_low`        | Gradient stability guard (avoid switching during unstable training) | SYNC→ASYNC |
| `straggler_var_low`   | Straggler-resolved detector (worker times converged again) | ASYNC→SYNC |
| `grad_var_high`       | Gradient divergence signal | ASYNC→SYNC |
| `loss_trend_threshold`| Rising loss signal | ASYNC→SYNC |

Phase 4 tuned configs use:
```
straggler_var_high:   0.18    # (Phase 3 used 0.18 → confirmed from demo)
grad_var_low:         0.85    # more permissive than Phase 3's 0.75
straggler_var_low:    0.12    # new — Scenario B back-switch
cooldown_steps:       120     # shorter than Phase 3's 200
decision_window:      2       # same as Phase 3 demo
```

The threshold sweep (`experiments/threshold_sweep.py`) runs a 12-point grid:
- `straggler_var_high` ∈ {0.12, 0.18, 0.25}
- `grad_var_low` ∈ {0.70, 0.85}
- `cooldown_steps` ∈ {100, 150}

Selection criterion: the config that produces exactly **1 switch** with the **earliest first-switch step** (fastest straggler detection) is the winner.

---

### 4. How to run Phase 4

#### 4a. Prerequisites

- Python environment from `requirements.txt` activated (`.venv`).
- CIFAR-10 data downloaded (happens automatically on first run).
- Working directory: `c:\Users\786\Documents\pdc_project\`

#### 4b. Run the threshold sweep first (fastest, ~2–3 hours total)

```powershell
python experiments/threshold_sweep.py
```

Output: `logs/phase4_sweep_results.csv` + printed best-config recommendation.

#### 4c. Run all 9 scenario comparisons

```powershell
# All 9 experiments, seed 42 only (~8–12 hours total on CPU)
python experiments/run_scenarios.py

# Or run a single scenario group for faster testing:
python experiments/run_scenarios.py --scenario sA

# Or run one specific config:
python src/hybrid/launcher.py --nworkers 2 --config configs/phase4_sA_hybrid.yaml
```

#### 4d. Run multiple seeds (for Phase 5 error bars)

```powershell
python experiments/run_scenarios.py --seeds 42 123 777
```

#### 4e. Analyse results

```powershell
python experiments/analyze_results.py --log-dir ./logs --tta-target 40.0
```

This prints the comparison tables to stdout. To save to a file for the Phase 5 report:

```powershell
python experiments/analyze_results.py --output docs/phase4_results_tables.md
```

#### 4f. Docker run (for proposal alignment)

```powershell
# Build the image (one-time):
docker build -t pdc-hybrid:latest -f docker/Dockerfile .

# Run Scenario A hybrid inside the container:
docker run --rm `
  -v ${PWD}/data:/workspace/data `
  -v ${PWD}/logs:/workspace/logs `
  -v ${PWD}/checkpoints:/workspace/checkpoints `
  pdc-hybrid:latest `
  --nworkers 2 --config configs/phase4_sA_hybrid.yaml

# Or using compose:
docker compose -f docker/docker-compose.yml run --rm scenario_runner `
  python experiments/run_scenarios.py --scenario sA
```

---

### 5. Expected deliverables for Phase 5

After Phase 4 runs complete you should have:

1. **9 PS log CSVs** (one per scenario × mode combination) in `logs/` with names like `hybrid_resnet18_cifar10_nw2_p4_sA_hybrid_s42_ps.csv`.
2. **Switch event CSVs** for each hybrid run (at least 1 switch for Scenario A, at least 2 for Scenario B).
3. **`logs/phase4_sweep_results.csv`** with 12 threshold combinations.
4. **`logs/phase4_run_summary.csv`** from `run_scenarios.py`.
5. (Optional) `docs/phase4_results_tables.md` from `analyze_results.py`.

Phase 5 will consume these artifacts to build the final report tables, accuracy-vs-time plots, and the mode-overlay timeline plots.

---

### 6. What Phase 4 does *not* require

- Re-implementing Phases 1–3 inside Docker from scratch — the **same `src/`** modules are used.
- PyTorch RPC — not available on Windows CPU; the shared-memory approach is correct and honest for the scope of this project.
- True cross-container gradient transport — Docker gives the containerization story for the proposal; the multi-process shared-memory approach correctly emulates the distributed PS topology.

---

**Summary:** Phase 4 is the **Docker + scenario + threshold tuning** phase that aligns the implementation with the submitted proposal. All code, configs, and scripts are in place. Running the experiments in order (sweep → scenarios → analysis) produces the complete evidence base for Phase 5 reporting.
