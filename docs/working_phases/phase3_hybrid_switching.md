## Phase 3 – Hybrid Switching Framework (Sync ↔ Async)

### 1. Objectives

- Build the **core novelty**: a controller that dynamically switches between:
  - Synchronous All-Reduce (DDP-style)
  - Asynchronous Parameter Server (RPC-style)
- Use runtime metrics:
  - Communication latency
  - Gradient variance / diversity
  - Possibly staleness
- Ensure stable behavior (no rapid oscillation).

### 2. Prerequisites

- Phase 1: working DDP setup.
- Phase 2: working PS + RPC setup.
- Common metric logging utilities.

### 3. Project Structure (Additions)

- `src/hybrid/`
  - `controller.py` (switching logic / policy)
  - `hybrid_trainer.py` (coordinates workers and mode transitions)
  - `metrics_collector.py` (latency, variance, staleness aggregation)
  - `config.py` (thresholds, windows, cooldowns)

### 4. Implementation Steps

#### Step 1 – Metric Definitions

- **Latency metric**:
  - For sync: measure all-reduce latency per step.
  - For async: measure RPC round-trip time for gradient push / param pull.
- **Gradient variance / diversity**:
  - Compute simple proxy:
    - variance of gradient norm across workers (sync),
    - variance of local loss or gradient norm over time (async).
- Use moving averages:
  - e.g. exponential moving average over last 50–100 steps.

#### Step 2 – Controller Policy Design

- In `controller.py`:
  - Maintain current mode: `"sync"` or `"async"`.
  - Inputs each decision:
    - `avg_latency`
    - `avg_gradient_variance`
    - `avg_staleness` (if available)
  - Simple rule example:
    - If `latency > L_high` or `staleness > S_high` for `W` consecutive steps → switch to `"async"`.
    - If `gradient_variance > V_high` for `W` steps and `latency < L_low` → switch to `"sync"`.
  - Add **cooldown**:
    - After a switch, must stay in new mode for at least `C` steps before next switch.

#### Step 3 – Mode Abstractions

- In `hybrid_trainer.py`:
  - Implement two “engines” behind a common interface:
    - `SyncEngine` – wraps DDP step.
    - `AsyncEngine` – wraps RPC PS step.
  - Training loop:
    - At each global step:
      - Ask `controller` which mode is active.
      - Call `SyncEngine.step()` or `AsyncEngine.step()`.
      - Collect and pass metrics back to controller.

#### Step 4 – State Transfer Between Modes

- When switching from async → sync:
  - Ensure all workers retrieve the **same, latest global parameters** from PS.
  - Reinitialize DDP with that state.
- When switching from sync → async:
  - Pick rank 0 model as reference state.
  - Broadcast to PS and workers as starting point.
- Keep this simple and robust:
  - Only allow switching at safe synchronization points (e.g. end of an epoch or fixed number of steps).

#### Step 5 – Logging and Debugging

- Log every switch event:
  - `step, old_mode, new_mode, avg_latency, avg_variance, avg_staleness`.
- Plot (later):
  - mode vs time (for understanding behavior).
  - latency and variance around switch events.

### 5. Deliverables for Phase 3

- Working hybrid training loop that:
  - Runs without crashing.
  - Can switch modes at least in controlled tests.
- Logs that show:
  - When and why mode switches occurred.
  - Metric values before/after switch.

