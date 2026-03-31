## Phase 1 – Synchronous All-Reduce Baseline (DDP)

### 1. Objectives

- Establish a **clean synchronous baseline** using PyTorch Distributed Data Parallel (DDP).
- Measure:
  - Per-epoch training time
  - Throughput (images/sec)
  - Validation accuracy vs time/epochs
- Validate that the single-machine, multi-process, multi-container setup works end to end.

### 2. Prerequisites

- Python (3.9+ recommended)
- PyTorch with `torchvision`
- CUDA optional (CPU-only is acceptable with smaller batch sizes)
- Docker Desktop + WSL2 (for later phases, but you can start without Docker)
- Git for version control (recommended)

### 3. Project Structure (Minimum)

- `src/`
  - `common/`
    - `datasets.py` (CIFAR-10/100 loaders, transforms)
    - `models.py` (ResNet-18, MobileNetV2 factory functions)
    - `metrics.py` (accuracy, loss tracking)
    - `logging_utils.py` (CSV logging, console formatting)
  - `sync_allreduce/`
    - `train_ddp.py` (entry point for DDP training)
    - `launcher.py` (spawns multiple processes via `torch.multiprocessing` or `torchrun`)

### 4. Implementation Steps

#### Step 1 – Dataset and Model Utilities

- Implement `datasets.py`:
  - Function `get_dataloaders(dataset_name, batch_size, num_workers, distributed)`  
    - Downloads CIFAR-10/100.
    - Uses `DistributedSampler` when `distributed=True`.
- Implement `models.py`:
  - Functions `create_resnet18(num_classes)` and `create_mobilenet_v2(num_classes)`.

#### Step 2 – Basic Single-Process Training Script

- In `sync_allreduce/train_ddp.py`, first write a **single-process** training loop:
  - Instantiate model, loss, optimizer.
  - Train for a few epochs on CIFAR-10.
  - Compute validation accuracy.
- Verify:
  - Code runs successfully.
  - Loss decreases.
  - Accuracy is reasonable (>40–50% after a few epochs on CIFAR-10).

#### Step 3 – Convert to DDP (Multi-Process, Single Machine)

- Initialize the process group:
  - Use `torch.distributed.init_process_group(backend="nccl" or "gloo", init_method="env://")`.
- Wrap the model in DDP:
  - `model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])` (for GPU).
  - For CPU-only, omit `device_ids`.
- Use `DistributedSampler` in dataloader to partition data across ranks.
- Only let **rank 0** handle logging and checkpoint saving.

#### Step 4 – Launch Script

- Create `sync_allreduce/launcher.py`:
  - Uses `torch.multiprocessing.spawn` or rely on `torchrun` CLI.
  - Example invocation (conceptually):
    - `torchrun --nproc_per_node=4 src/sync_allreduce/train_ddp.py --config ...`
- Verify that 2–4 processes run correctly and all ranks exit cleanly.

#### Step 5 – Baseline Experiments and Logging

- Define a simple config:
  - `dataset = CIFAR-10`
  - `model = ResNet-18`
  - `epochs = 10–20` (start with 5 for debugging)
  - `world_size = 2, 4`
- Log per-epoch:
  - `epoch, train_loss, val_loss, val_acc, epoch_time_seconds`
- Export logs to CSV:
  - `logs/sync_resnet18_cifar10_world4.csv`.

### 5. Deliverables for Phase 1

- Working DDP training script (single machine).
- CSV logs for at least:
  - 2 workers
  - 4 workers
- Short text summary:
  - How training time and throughput scale with more workers.

