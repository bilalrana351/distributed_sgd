# Adaptive Distributed SGD

An experimental distributed training system that compares synchronous SGD,
asynchronous parameter-server training, and a hybrid controller that switches
between the two modes at runtime.

The project trains ResNet-18 or MobileNetV2 on CIFAR-10/CIFAR-100 and studies
the trade-off between convergence speed, model quality, throughput, and
gradient staleness under worker slowdowns.

## What the project implements

- **Synchronous baseline:** workers average gradients before each update.
- **Asynchronous parameter server:** gradients are applied as workers produce
  them, without waiting for the slowest worker.
- **Adaptive hybrid mode:** a controller monitors worker-time variance,
  gradient-norm variance, and loss trends, then switches between synchronous
  and asynchronous aggregation.
- **Straggler scenarios:** reproducible persistent and transient worker delays.
- **Experiment tooling:** YAML configurations, CSV logging, scenario runners,
  threshold sweeps, analysis scripts, and Docker-based execution.

## Architecture

The hybrid system runs on one host and uses multiple processes to simulate a
distributed training environment:

```text
                  +-------------------------------+
                  |       Parameter Server        |
                  |  aggregator | metrics | model |
                  +---------------+---------------+
                                  |
                         adaptive controller
                       sync <----------> async
                                  |
                    shared-memory gradient queue
                         /        |        \
                    worker 0   worker 1   worker N
```

Workers compute gradients on separate data partitions. The parameter server
owns the shared model, applies updates, records metrics, and exposes the signals
used by the controller. See
[`docs/assignment2_system_design.md`](docs/assignment2_system_design.md) for the
full design.

## Reported results

In the recorded persistent-straggler experiments, hybrid training reached 40%
validation accuracy in 1,233 seconds, compared with 1,409 seconds for the
synchronous baseline. In the transient-straggler experiments, it reached the
same target in 864 seconds, compared with 1,323 seconds for synchronous
training.

These results demonstrate the controller's speed advantage under stragglers,
while also exposing an accuracy trade-off under transient slowdowns. The full
methodology, caveats, and per-scenario results are documented in
[`docs/phase4_results_discussion.md`](docs/phase4_results_discussion.md) and
[`docs/phase4_results_tables.md`](docs/phase4_results_tables.md).

## Requirements

- Python 3.11 or newer
- PyTorch 2.2 or newer
- Docker and Docker Compose, optional

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Quick start

Run the hybrid controller with two workers:

```bash
python src/hybrid/launcher.py \
  --nworkers 2 \
  --config configs/phase4_sA_hybrid.yaml
```

Run the synchronous baseline:

```bash
python src/sync_allreduce/launcher.py \
  --nproc 2 \
  --config configs/phase1_ddp_world2_safe.yaml
```

Run the asynchronous parameter server:

```bash
python src/async_ps/launcher.py \
  --nworkers 2 \
  --config configs/phase2_async_ps_final.yaml
```

CIFAR data is downloaded to `data/` on first use. Experiment logs and model
checkpoints are written to `logs/` and `checkpoints/`.

## Reproduce the scenario matrix

The Phase 4 runner covers three environments and three aggregation modes:

| Scenario | Description | Modes |
| --- | --- | --- |
| `s0` | Homogeneous workers | sync, async, hybrid |
| `sA` | Persistent straggler | sync, async, hybrid |
| `sB` | Transient straggler | sync, async, hybrid |

Preview the commands without running the experiments:

```bash
python experiments/run_scenarios.py --dry-run
```

Run a single scenario across all modes:

```bash
python experiments/run_scenarios.py --scenario sA
```

Run the complete matrix across multiple seeds:

```bash
python experiments/run_scenarios.py --seeds 42 123 777
```

Full CPU runs may take several hours.

## Docker

Build the image and run the key persistent-straggler experiment:

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm scenario_runner \
  python src/hybrid/launcher.py \
  --nworkers 2 \
  --config configs/phase4_sA_hybrid.yaml
```

The Compose setup mounts `data/`, `logs/`, and `checkpoints/` so outputs persist
on the host.

## Analyze results

Generate an aggregate results table from the experiment logs:

```bash
python experiments/analyze_results.py \
  --log-dir logs \
  --tta-target 40.0 \
  --output docs/phase4_results_tables.md
```

The analysis tracks validation accuracy, time-to-accuracy, wall-clock time,
throughput, controller switches, mode occupancy, and gradient staleness.

## Repository layout

```text
configs/       Experiment and controller configurations
docker/        Reproducible container environment
docs/          Design, setup, results, and report material
experiments/   Scenario runners, sweeps, and analysis scripts
logs/          Recorded experiment outputs
notebooks/     Hosted-notebook experiment workflow
src/
  async_ps/    Asynchronous parameter-server implementation
  common/      Datasets, models, logging, and shared metrics
  hybrid/      Adaptive controller and hybrid trainer
  sync_allreduce/  Synchronous distributed baseline
```

## Further documentation

- [System design](docs/assignment2_system_design.md)
- [Detailed project overview](docs/project_overview_demo.md)
- [Windows setup guide](docs/phase1_setup_guide.md)
- [Results discussion](docs/phase4_results_discussion.md)
- [Final report](docs/phase5_final_report.md)

## Current limitations

- The implementation simulates distributed workers as processes on one host;
  it is not a multi-node cluster runtime.
- Controller thresholds are workload-sensitive and require tuning.
- Recorded baseline and hybrid results include mixed log vintages; consult the
  results discussion before drawing direct comparisons.
- The transient-straggler scenario retains a speed-accuracy trade-off that is
  identified as future tuning work.
