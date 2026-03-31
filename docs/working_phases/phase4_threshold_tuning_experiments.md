## Phase 4 – Threshold Tuning and Experimental Scenarios

### 1. Objectives

- Tune switching thresholds for:
  - Latency
  - Gradient variance
  - Staleness (if used)
- Run defined scenarios:
  - Baseline homogeneous
  - Persistent stragglers
  - Transient heterogeneity
- Collect reproducible experimental results.

### 2. Prerequisites

- Hybrid framework from Phase 3.
- Basic plotting/notebook environment (e.g. Jupyter or simple Python plotting script).

### 3. Project Structure (Additions)

- `experiments/`
  - `scenario_configs.py` (definitions of different scenarios)
  - `run_experiments.py` (script to launch repeated runs per scenario)
  - `analyze_results.ipynb` or `analyze_results.py` (plots and summary tables)

### 4. Implementation Steps

#### Step 1 – Scenario Definitions

- **Scenario 0 – Homogeneous Baseline**
  - No artificial delays.
  - Equal resources for all workers.
- **Scenario A – Persistent Stragglers**
  - Mark ~25% workers as slow:
    - Sleep or extra CPU work each step.
    - Or higher artificial network latency.
- **Scenario B – Transient Heterogeneity**
  - Randomly insert latency spikes in different workers over time.

- Encapsulate each scenario:
  - `apply_scenario(worker_id, step)` to determine injected delay or config.

#### Step 2 – Initial Threshold Sweep (Coarse)

- Choose coarse grid for thresholds, for example:
  - `L_high` in {low, medium, high}
  - `V_high` in {low, medium, high}
  - `S_high` in {low, medium} (if used)
- For each scenario:
  - Run short experiments (fewer epochs or smaller subsets):
    - Collect:
      - time-to-reach-small-target-accuracy
      - number of mode switches
      - final accuracy
- Select threshold combinations that:
  - Avoid excessive switching.
  - Show improved TTC vs pure sync in A & B.

#### Step 3 – Final Configuration Selection

- Pick one or two “best” threshold configs for each scenario:
  - Or a single robust config that works reasonably well across all scenarios.
- Fix this configuration for **final evaluation** in Phase 5.

#### Step 4 – Full-Length Runs

- For each scenario (0, A, B):
  - Run:
    - pure sync baseline
    - pure async baseline
    - hybrid (tuned)
  - Repeat each configuration 3 times with different random seeds for stability.
- Log:
  - per-epoch accuracy, loss, and time
  - total training time
  - mode switch events
  - staleness (if applicable)

### 5. Deliverables for Phase 4

- Scenario configuration code.
- CSV logs for all runs and scenarios.
- Initial plots/notebooks showing:
  - Impact of different thresholds.
  - Stability of switching behavior.

