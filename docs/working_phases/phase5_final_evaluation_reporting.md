## Phase 5 – Final Evaluation, Analysis, and Reporting

### 1. Objectives

- Perform the **final comparison** between:
  - Pure synchronous All-Reduce (sync baseline)
  - Pure asynchronous Parameter Server (async baseline)
  - **Hybrid** switching (your contribution)
- Present results in a way that matches the **submitted proposal**: **container-based** distributed nodes, **network scenarios**, and **measurable** gains (e.g. time-to-target accuracy, final accuracy, throughput, switching behavior).
- Prepare **report + slides** for your professor.

### 2. Prerequisites

- **Phase 4 completed** in the sense that your **primary experimental evidence** comes from **Docker multi-container runs** (4–8 containers as feasible) under the defined scenarios.  
- **Phases 1–3 logs** remain valid as **development / sanity-check** evidence (algorithm correctness, hybrid switch demonstration on host), cited explicitly as such in the write-up so the narrative is honest: *implementation proven on host; evaluation aligned with proposal in containers.*

### 3. Does Phase 5 “use Docker”?

**Phase 5 does not introduce new Docker work.** Phase 5 **consumes** the artifacts from Phase 4:

- Aggregated metrics from container runs  
- Plots and tables  
- Optional appendix: compose file + image tag + command line to reproduce  

If Phase 4 is done correctly, **Phase 5 is mostly analysis and writing** — but your **report’s Experimental Setup section must describe the Docker topology** (number of containers, network, how stragglers/latency were applied) because that is what your proposal promised.

### 4. Evaluation dimensions

- **Time-to-Convergence (TTC):** time to reach a fixed validation accuracy (e.g. 70% on CIFAR-10), or a proxy if runs are shorter.
- **Final validation accuracy** after a fixed budget (epochs or steps).
- **Throughput** (images/sec or samples/sec) **per configuration**, noting container overhead vs bare-metal Phase 1–3 if compared.
- **Resource / straggler impact:** idle time or wall-clock gap in sync vs async vs hybrid under **Scenario A/B** (from Phase 4).
- **Stability:** loss/accuracy curves; no divergence.
- **Hybrid-specific:** number of switches, modes over time, signals at switch (from `*_switches.csv` or equivalent).

### 5. Analysis steps

#### Step 1 – Aggregate metrics

Per scenario and method (sync / async / hybrid):

- Mean ± std over seeds for TTC, final accuracy, throughput, switch count, mean/max staleness.

#### Step 2 – Plots

- Accuracy vs wall-clock for all three methods per scenario.
- TTC bar charts with error bars.
- Hybrid: overlay **mode** (sync/async) on the time axis (shaded regions), using logs.

#### Step 3 – Discussion per scenario

- **Homogeneous:** Does hybrid track sync? Unnecessary switching?
- **Persistent stragglers:** Does hybrid improve TTC vs sync? Accuracy cost vs async-only?
- **Transient heterogeneity:** Does the controller react on sensible timescales?

#### Step 4 – Novelty and limitations

- State clearly: **bi-directional** hybrid policy, **live** signals, **containerized** evaluation matching the proposal.
- Limitations: CPU-only laptop, fewer than 8 containers if applicable, subset validation during debugging, etc.

#### Step 5 – Final report and presentation

- Problem → gap → design → **Docker experimental setup** → results → conclusion.
- 10–15 slides, with one slide on **reproducibility** (Docker image + compose).

### 6. Deliverables for Phase 5

- Final tables and plots (from **Phase 4** runs as primary).
- Written report sections integrated with course requirements.
- Presentation slides.
- Clear conclusion: **when** hybrid helps and **how** it was validated in the **proposed** environment.
