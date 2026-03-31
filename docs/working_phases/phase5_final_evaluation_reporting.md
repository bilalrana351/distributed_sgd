## Phase 5 – Final Evaluation, Analysis, and Reporting

### 1. Objectives

- Perform **final comparison** between:
  - Pure synchronous All-Reduce
  - Pure asynchronous Parameter Server
  - Hybrid switching method
- Evaluate against course requirements:
  - Scalability
  - Failure/heterogeneity handling
  - Sensitivity analysis
  - Measurable improvement
- Prepare material for final presentation and report.

### 2. Prerequisites

- All previous phases completed.
- Logs from repeated runs available and organized.

### 3. Evaluation Dimensions

- **Time-to-Convergence (TTC)**:
  - Time to reach a fixed target accuracy (e.g. 70% on CIFAR-10).
- **Final Validation Accuracy**:
  - After fixed number of epochs/steps.
- **Throughput**:
  - Samples per second.
- **Resource Utilization / Idle Time**:
  - Approximate waiting time in sync vs async vs hybrid.
- **Stability of Training**:
  - Loss/accuracy curves (no divergence).
- **Switching Behavior (Hybrid)**:
  - Number of switches.
  - Situations in which switching occurs.

### 4. Analysis Steps

#### Step 1 – Aggregate Metrics

- For each configuration and scenario:
  - Compute:
    - mean and std of TTC over 3 runs
    - mean and std of final accuracy
    - mean throughput
    - average/max staleness (async/hybrid)
- Store results in tables (CSV / LaTeX / Markdown).

#### Step 2 – Plots

- Plot per scenario:
  - Accuracy vs time for sync, async, hybrid.
  - TTC bar charts with error bars.
  - For hybrid:
    - overlay mode (sync/async) on accuracy/time plot (e.g. shaded regions).
- Optional:
  - latency/variance metrics around switch events.

#### Step 3 – Comparative Discussion

- For each scenario:
  - **Homogeneous**:
    - Does hybrid perform similarly to sync?
    - Is there any unnecessary switching?
  - **Persistent Stragglers**:
    - Does hybrid reduce TTC vs sync?
    - How much accuracy gap vs sync (if any)?
  - **Transient Heterogeneity**:
    - Does hybrid respond quickly enough?
    - Is switching pattern intuitive given latency spikes?

#### Step 4 – Trade-off and C-6 Justification

- Explicitly discuss:
  - Where hybrid is better (TTC, idling, robustness).
  - Where hybrid might be worse (slightly more complexity, overhead, occasional accuracy drop).
- Argue novelty:
  - Bi-directional, real-time switching guided by live telemetry (latency + gradient variance) in a single framework.
  - Experimental evidence that it improves TTC with bounded impact on accuracy under realistic heterogeneity.

#### Step 5 – Final Report and Presentation

- Integrate results into final project report sections:
  - Experimental setup
  - Workload modeling
  - Scalability results
  - Failure scenario evaluation
  - Sensitivity analysis
  - Measurable improvement
  - Novelty justification
- Prepare 10–15 slide deck:
  - Problem → Gap → Design → Implementation → Results → Lessons.

### 5. Deliverables for Phase 5

- Final aggregated tables and plots.
- Final written analysis (report sections).
- Presentation slides.
- Clear statement of project conclusion:
  - When and how the hybrid approach provides value.

