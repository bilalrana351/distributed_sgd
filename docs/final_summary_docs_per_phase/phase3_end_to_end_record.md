# Phase 3 End-to-End Record (Simple + Detailed)

## 1) What this phase was about

Phase 3 goal was to build a **working hybrid training pipeline** that can **dynamically switch** between:

- **Synchronous aggregation** (same semantics as All-Reduce / DDP: one averaged gradient update per logical round when all workers have contributed), and  
- **Asynchronous aggregation** (same semantics as Phase 2: PS applies each worker’s gradient as it arrives),

and to verify that:

- the dual-mode PS loop runs end-to-end without deadlock,
- a **controller** can evaluate live signals and **log mode switches** with reasons,
- staleness and validation metrics behave sensibly in each mode,
- checkpoints capture **global step** and **switch history**.

This phase is the foundation for **Phase 4** (Docker multi-container network, scenario sweeps, and threshold tuning aligned with the submitted proposal).

---

## 2) Important design reality (no RPC on Windows host)

### Constraint from Phase 2

- **PyTorch RPC** remains unavailable on the Windows CPU build (`torch.distributed.rpc.is_available()` is `False`).
- Phase 3 therefore **does not** switch between “DDP process group” and “RPC PS” as two separate distributed backends.

### Practical and correct approach used

Phase 3 uses the **same** stack as Phase 2 for transport:

- `torch.multiprocessing.spawn`,
- shared model memory (`model.share_memory()`),
- `mp.Queue` for gradient pushes,
- `multiprocessing.Lock` for safe parameter reads/writes.

**What changes is only PS-side aggregation logic:**

- **Sync mode:** PS buffers one gradient per worker, **averages** them, applies **one** update, then releases workers via a shared **`sync_round`** counter.  
- **Async mode:** PS applies each gradient **immediately** (Phase 2 behavior).

The **controller** runs inside the PS process and flips a shared **`mode_flag`** after cooldown + consecutive triggers. This matches the **research intent** (bi-directional hybrid SGD) without requiring RPC or tearing down DDP groups on Windows.

---

## 3) What was implemented in Phase 3

Core code and structure created:

- `src/hybrid/__init__.py`  
  - Package marker.

- `src/hybrid/metrics_collector.py`  
  - EMA-smoothed signals: **worker step-time variance** (straggler proxy), **gradient-norm variance** (diversity / instability proxy), **loss trend**.

- `src/hybrid/controller.py`  
  - `ModeController`: threshold rules, **cooldown**, **decision window**, `SWITCH_LOG_FIELDS` for CSV.

- `src/hybrid/train_hybrid.py`  
  - PS loop: sync buffer + async immediate apply; transition helper; CSV schemas (`hybrid_*_ps`, `staleness`, `switches`, `worker*`); optional **straggler simulation** via config (`time.sleep` per worker step range).  
  - Worker loop: push `(grads, worker_id, step, loss, step_time, grad_norm)`; sync wait on `sync_round`; async periodic param pull.

- `src/hybrid/launcher.py`  
  - Spawns 1 PS + N workers; creates `mode_flag`, `sync_round`, queues, locks.

- `configs/phase3_hybrid_safe.yaml`  
  - Conservative **smoke test** (long run, sync-only expected without straggler).

- `configs/phase3_hybrid_straggler.yaml`  
  - Shorter run with **straggler enabled** + tuned controller + `run_id: straggler_demo` for clean log filenames.

- **`run_id` support** in `train_hybrid.py`  
  - Optional suffix on log and checkpoint names (addresses Phase 2 note about mixed append-only CSV history).

Reused from earlier phases:

- `src/common/*` (datasets, models, metrics, logging).  
- `StalenessTracker` from `src/async_ps/ps_node.py` for PS interval stats.

---

## 4) Pipeline flow (simple English)

### Runtime roles

- **Rank 0** = Parameter Server (hybrid: sync or async aggregation)  
- **Rank 1..N** = Workers  

### Sync mode (high level)

1. Each worker computes a batch → backward → pushes gradient tuple to `grad_queue`.  
2. PS collects **one gradient per worker** into a buffer.  
3. When the buffer is full, PS **averages** gradients, applies **one** SGD step, increments **`global_step`**, increments **`sync_round`**.  
4. Workers spin until `sync_round` advances, then **pull** parameters from shared PS model.  
5. Staleness logged per worker for that round (typically **0** when workers stay aligned).

### Async mode (high level)

1. Worker pushes gradient; PS applies **immediately** (no wait for other workers).  
2. Workers pull parameters every `param_pull_every` steps.  
3. Staleness grows as in Phase 2 (`global_step_before_update - worker_step`).

### Controller

- Every `eval_interval` **global steps**, PS evaluates EMA signals.  
- If rules fire for **`decision_window`** consecutive evaluations **and** **cooldown** has passed, PS switches **`mode_flag`** and logs a row to `*_switches.csv`.

---

## 5) Run commands and configs used

### Smoke test (stay in sync; no straggler)

```bash
python src/hybrid/launcher.py --nworkers 2 --config configs/phase3_hybrid_safe.yaml
```

Main settings (see YAML): CIFAR-10, ResNet-18, **1500** steps per worker, batch 32, `start_mode: sync`, straggler **off**, periodic subset validation.

### Straggler demo (prove SYNC → ASYNC switch)

```bash
python src/hybrid/launcher.py --nworkers 2 --config configs/phase3_hybrid_straggler.yaml
```

Main settings: **1200** steps per worker, straggler on **worker_id 1** with **2.0 s** extra delay between local steps **150–800**, aggressive controller thresholds, **`run_id: straggler_demo`**.

---

## 6) Raw logs and checkpoints produced

### Smoke test (default filenames, no `run_id`)

| Artifact | Path |
|----------|------|
| PS log | `logs/hybrid_resnet18_cifar10_nw2_ps.csv` |
| Staleness | `logs/hybrid_resnet18_cifar10_nw2_staleness.csv` |
| Switches | `logs/hybrid_resnet18_cifar10_nw2_switches.csv` (header only if no switch) |
| Workers | `logs/hybrid_resnet18_cifar10_nw2_worker0.csv`, `..._worker1.csv` |
| Checkpoint | `checkpoints/hybrid_resnet18_cifar10_nw2_final.pt` |

### Straggler demo (`run_id: straggler_demo`)

| Artifact | Path |
|----------|------|
| PS log | `logs/hybrid_resnet18_cifar10_nw2_straggler_demo_ps.csv` |
| Staleness | `logs/hybrid_resnet18_cifar10_nw2_straggler_demo_staleness.csv` |
| Switches | `logs/hybrid_resnet18_cifar10_nw2_straggler_demo_switches.csv` |
| Workers | `logs/hybrid_resnet18_cifar10_nw2_straggler_demo_worker0.csv`, `..._worker1.csv` |
| Checkpoint | `checkpoints/hybrid_resnet18_cifar10_nw2_straggler_demo_final.pt` |

### Checkpoint integrity (straggler demo)

From `hybrid_resnet18_cifar10_nw2_straggler_demo_final.pt`:

- `global_step`: **1599**  
- `total_switches`: **1**  
- `model_state`: **122** tensors  
- `switch_history`: one entry at **global_step 800** (sync → async)

---

## 7) Results summary (from your logs)

### 7a) Smoke test — PS summary rows

From `hybrid_resnet18_cifar10_nw2_ps.csv`:

| global_step | current_mode | val_acc_top1 | avg_staleness | Notes |
|---:|:---|---:|---:|---|
| 500 | sync | **39.74%** | 0.0 | Subset / quick val settings |
| 1000 | sync | **45.00%** | 0.0 | Learning progressing |
| 1500 | sync | **51.82%** | 0.0 | Still sync only |

**Switches file:** no data rows (expected: no straggler, thresholds not crossed).

**Checkpoint:** `total_switches = 0`, `global_step = 1500`.

### 7b) Straggler demo — switch event

From `hybrid_resnet18_cifar10_nw2_straggler_demo_switches.csv`:

| global_step | old_mode | new_mode | worker_time_var | grad_norm_var | loss_trend |
|---:|:---|:---|---:|---:|---:|
| 800 | sync | async | **0.5638** | **0.0239** | **0.0191** |

Interpretation: **high** relative spread in worker step times (straggler visible) with **low** gradient-norm variance → policy correctly favors **async** under the SYNC→ASYNC rule.

### 7c) Straggler demo — PS summary rows

From `hybrid_resnet18_cifar10_nw2_straggler_demo_ps.csv`:

| global_step | current_mode | val_acc_top1 | avg_staleness | max_staleness |
|---:|:---|---:|---:|---:|
| 400 | sync | **37.24%** | 0.0 | 0 |
| 813 | async | **44.79%** | (see note) | 0 |
| 1599 | async | **44.43%** | **202.56** | **399** |

Note: the middle row can show **negative** average/min staleness briefly right after a mode change (edge of `global_step` vs `worker_step`); later async rows show **positive** staleness growth as expected.

### 7d) Staleness file sanity (straggler demo)

- Last **sync** rows at **global_step 800** (both workers, staleness **0**).  
- First **async** rows at **global_step 801+** with `current_mode: async`.

---

## 8) Simple interpretation (plain English)

1. **Hybrid pipeline correctness is confirmed.**  
   Smoke test completed 1500 local steps per worker with **stable sync** aggregation and sensible validation trend.

2. **Controlled mode switch is confirmed.**  
   Straggler demo produced exactly **one** logged transition **sync → async** at step **800**, with signal values that match the intended policy story.

3. **Async segment behaves like Phase 2 qualitatively.**  
   After the switch, **staleness** rises (non-zero averages, high max), unlike the sync-only smoke run.

4. **Bi-directional switch (async → sync) was not required to close Phase 3 on host.**  
   The remaining direction can be stressed in **Phase 4** (threshold sweeps, longer runs, or Docker network scenarios) if the rubric asks for both directions in the final evaluation.

5. **Do not compare Phase 3 val numbers directly to Phase 1 epoch-5 numbers** without matching step budget, LR schedule, and full vs subset validation — the point of Phase 3 is **mechanism + logging + switch**, not a fair final accuracy bake-off.

---

## 9) Comparison perspective vs Phase 1 and Phase 2

| Aspect | Phase 1 (DDP sync) | Phase 2 (Async PS) | Phase 3 (Hybrid) |
|--------|-------------------|-------------------|------------------|
| Consistency | Strong (all-reduce semantics) | Weaker (stale grads) | **Selectable** per segment |
| Straggler tolerance | Waits for slowest | Does not wait | Can **move to async** when straggler signal fires |
| Implementation on your laptop | `gloo` DDP | Shared-memory PS | **Same PS topology**, dual aggregation + controller |

Phase 3 **unifies** the two training styles in one runnable system (on host), preparing the story for Phase 4’s **container + network** evaluation.

---

## 10) What Phase 2 “to improve” items were addressed

| Phase 2 suggestion | Phase 3 status |
|--------------------|----------------|
| Run-id style log naming | **`run_id`** in config + filenames |
| Staleness + step-time + loss trend for controller | **Implemented** in metrics + rules |
| Parser for latest run block | Optional; not required to complete Phase 3 |
| Full val for official comparisons | Still recommended for **Phase 4/5** reports |

---

## 11) Phase 4 direction (next phase)

Phase 4 (per updated `docs/working_phases/phase4_threshold_tuning_experiments.md`) should:

1. Run the **same logical training modes** inside **Docker** (**4–8 containers** where feasible) on a **bridge network**.  
2. Apply **network-level** scenarios (homogeneous, persistent straggler, transient latency) aligned with the **submitted proposal**.  
3. Sweep and freeze controller thresholds; collect multi-seed logs for **Phase 5** tables and plots.

Phase 3 remains the **algorithm and host-side proof**; Phase 4 is the **proposal-aligned experimental apparatus**.

---

## 12) Final statement

Phase 3 is successful.  
Your hybrid controller runs in-process with the parameter server, logs **when** and **why** a mode switch occurs, and checkpoints preserve **global step** and **switch history**.  
You are ready to proceed to **Phase 4** (Docker + scenarios + tuning) and then **Phase 5** (evaluation and reporting).
