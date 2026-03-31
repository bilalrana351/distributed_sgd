## Phase 2 – Asynchronous Parameter Server Baseline (RPC)

### 1. Objectives

- Implement an **asynchronous parameter server (PS)** using PyTorch RPC.
- Compare performance vs DDP baseline:
  - Time per epoch / per fixed number of steps
  - Throughput
  - Final accuracy
- Measure gradient staleness and update lag.

### 2. Prerequisites

- Phase 1 completed (dataset/model utilities, basic training loop).
- PyTorch RPC available in your install.

### 3. Project Structure (Additions)

- `src/async_ps/`
  - `ps_node.py` (parameter server logic)
  - `worker_node.py` (worker training logic)
  - `launcher.py` (starts PS + workers as separate processes)
  - `staleness_metrics.py` (optional helper)

### 4. Implementation Steps

#### Step 1 – Define Parameter Server Interface

- In `ps_node.py`:
  - Maintain global model parameters (on CPU or GPU).
  - RPC-exposed methods:
    - `get_parameters()` – returns current model state dict / flattened tensor.
    - `apply_gradients(gradients, worker_step)` – updates global params, tracks staleness.
  - Keep metadata:
    - `global_step`
    - map of `worker_id -> last_applied_step`.

#### Step 2 – Worker Logic

- In `worker_node.py`:
  - At startup:
    - Connect to RPC.
    - Pull initial parameters from PS.
  - Training loop:
    - For each local step:
      - Compute gradients on local batch.
      - Push gradients asynchronously to PS (via RPC).
      - Option: occasionally pull fresh parameters (e.g. every `k` steps).
  - Track local:
    - `local_step`
    - when parameters were last refreshed.

#### Step 3 – RPC Setup and Launch

- In `async_ps/launcher.py`:
  - Start:
    - 1 PS process (role `"ps"`)
    - N worker processes (roles `"worker_0"`, `"worker_1"`, …).
  - Initialize RPC with unique `rank` and `world_size`:
    - `rpc.init_rpc(name, rank, world_size, rpc_backend_options=...)`.
- Ensure:
  - Proper shutdown via `rpc.shutdown()` in all roles.
  - No deadlocks when stopping training.

#### Step 4 – Staleness Measurement

- For each `apply_gradients` call on PS:
  - Compute `staleness = global_step - worker_step` (or similar).
  - Log:
    - timestamp
    - worker id
    - staleness
- Aggregate per epoch or per fixed number of steps:
  - average staleness
  - max staleness

#### Step 5 – Baseline Experiments

- Use same dataset/model as Phase 1:
  - CIFAR-10 + ResNet-18.
- Fix number of steps per run (easier to compare vs DDP).
- Log:
  - `step, train_loss, val_acc (periodic), time_since_start, avg_staleness`.
- Compare with Phase 1:
  - Throughput (samples/sec).
  - Time to reach a given accuracy.
  - Final accuracy gap.

### 5. Deliverables for Phase 2

- Working PS + N workers training script with RPC.
- Logs of:
  - Accuracy vs time
  - Staleness statistics
- Short comparison vs DDP baseline:
  - Pros/cons in speed and accuracy.

