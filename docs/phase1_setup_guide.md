# Phase 1 – Setup & Run Guide (Windows 10, Fresh Machine)

---

## 1. Install Prerequisites

### 1.1 Python 3.11

1. Download the installer from https://www.python.org/downloads/
2. Run the installer.  **Check "Add Python to PATH"** before clicking Install.
3. Verify:
   ```powershell
   python --version    # should print Python 3.11.x
   pip --version
   ```

### 1.2 Git (optional but recommended)

Download from https://git-scm.com/download/win and install with defaults.

---

## 2. Create a Virtual Environment

Open PowerShell inside the project folder (`pdc_project`):

```powershell
# Create the venv
python -m venv .venv

# Activate it (run this every time you open a new terminal)
.venv\Scripts\Activate.ps1
```

> **Execution policy issue?**  Run this once as Administrator and retry:
> ```powershell
> Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## 3. Install PyTorch and Dependencies

### CPU-only (no NVIDIA GPU) – recommended starting point

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### NVIDIA GPU (CUDA 12.1) – only if you have a dedicated NVIDIA GPU

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Then open `configs/phase1_config.yaml` and set:
```yaml
use_cuda: true
backend:  "gloo"   # gloo works for both CPU and CUDA on Windows
                   # nccl is Linux-only; keep gloo on Windows
```

Verify the installation:
```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 4. Project Structure (What Was Created)

```
pdc_project/
├── src/
│   ├── common/
│   │   ├── datasets.py          ← CIFAR-10/100 loaders + DistributedSampler
│   │   ├── models.py            ← ResNet-18 & MobileNetV2 factory functions
│   │   ├── metrics.py           ← AverageMeter, accuracy, throughput
│   │   └── logging_utils.py     ← CSVLogger, console logger
│   └── sync_allreduce/
│       ├── train_ddp.py         ← main DDP training script (entry point)
│       └── launcher.py          ← torch.multiprocessing.spawn launcher
├── configs/
│   └── phase1_config.yaml       ← all hyperparameters in one place
├── logs/                        ← CSV logs written here automatically
├── checkpoints/                 ← .pt checkpoints saved here
├── data/                        ← CIFAR dataset downloads go here
├── docs/
│   ├── project_context_simple.md
│   ├── working_phases/
│   │   └── phase1_sync_baseline.md
│   └── phase1_setup_guide.md    ← this file
└── requirements.txt
```

---

## 5. Running Phase 1 Experiments

All commands are run from the **project root** (`pdc_project/`) with the venv active.

### 5.1 Quick Smoke Test (Single Process, No DDP)

Use this first to check that the code runs without any errors:

```powershell
# Temporarily set distributed=false in the config, or just override:
python src/sync_allreduce/train_ddp.py --config configs/phase1_config.yaml
```

Expected output after epoch 1:
```
[R0] HH:MM:SS | INFO     | Starting Phase-1 DDP baseline | world_size=1 ...
[R0] HH:MM:SS | INFO     | Model: resnet18 | Params: 11,173,962 | Dataset: cifar10 (10 classes)
[R0] HH:MM:SS | INFO     | CSV log → logs/sync_resnet18_cifar10_world1.csv
[R0] HH:MM:SS | INFO     | Epoch [  1/20] lr=0.10000 | train_loss=... val_acc@1=...%
```

Check that `logs/sync_resnet18_cifar10_world1.csv` is created and loss is decreasing.

---

### 5.1.1 Windows Stable Profile (recommended if PC hangs on epoch 1)

If the machine becomes unresponsive, use the low-resource profile first:

```powershell
python src/sync_allreduce/train_ddp.py --config configs/phase1_windows_stable.yaml
```

This profile uses:
- `batch_size: 32`
- `epochs: 5`
- `distributed: false`
- `max_cpu_threads: 4` (caps PyTorch CPU thread usage)

After this run completes reliably, move back to `configs/phase1_config.yaml` and scale gradually.

---

### 5.2 Multi-Process DDP – 2 Workers (torchrun, recommended)

```powershell
torchrun --nproc_per_node=2 src/sync_allreduce/train_ddp.py --config configs/phase1_config.yaml
```

Output CSV: `logs/sync_resnet18_cifar10_world2.csv`

---

### 5.3 Multi-Process DDP – 4 Workers (torchrun)

```powershell
torchrun --nproc_per_node=4 src/sync_allreduce/train_ddp.py --config configs/phase1_config.yaml
```

Output CSV: `logs/sync_resnet18_cifar10_world4.csv`

---

### 5.4 Alternative: launcher.py (if torchrun has issues)

```powershell
python src/sync_allreduce/launcher.py --nproc 2 --config configs/phase1_config.yaml
python src/sync_allreduce/launcher.py --nproc 4 --config configs/phase1_config.yaml
```

---

### 5.5 MobileNetV2 Variant

Change `model: "mobilenet_v2"` in the config, or copy the config:

```powershell
# Edit configs/phase1_config.yaml: set model to mobilenet_v2
torchrun --nproc_per_node=4 src/sync_allreduce/train_ddp.py --config configs/phase1_config.yaml
```

---

### 5.6 CIFAR-100 Variant

Change `dataset: "cifar100"` in the config (num_classes automatically becomes 100).

---

## 6. Understanding the Config

| Key              | Default        | Notes                                          |
|------------------|----------------|------------------------------------------------|
| `dataset`        | `cifar10`      | `cifar10` or `cifar100`                        |
| `model`          | `resnet18`     | `resnet18` or `mobilenet_v2`                   |
| `epochs`         | `20`           | Set to `5` for a quick debug run               |
| `batch_size`     | `128`          | Per-worker; effective = `128 × world_size`     |
| `lr`             | `0.1`          | Initial learning rate (SGD)                    |
| `lr_milestones`  | `[10, 15]`     | Epochs where LR is multiplied by `lr_gamma`    |
| `backend`        | `gloo`         | Use `gloo` on Windows; `nccl` on Linux+GPU     |
| `use_cuda`       | `false`        | Set `true` if NVIDIA GPU available             |
| `num_workers`    | `0`            | Dataloader threads; keep `0` on Windows        |
| `checkpoint_every` | `5`          | Save `.pt` checkpoint every N epochs           |

---

## 7. Deliverables Checklist

After running all experiments, you should have:

- [ ] `logs/sync_resnet18_cifar10_world1.csv`   – single-process baseline
- [ ] `logs/sync_resnet18_cifar10_world2.csv`   – 2-worker DDP
- [ ] `logs/sync_resnet18_cifar10_world4.csv`   – 4-worker DDP
- [ ] Checkpoints in `checkpoints/`
- [ ] Validation accuracy ≥ 40–50% after 5 epochs (sanity check)
- [ ] Validation accuracy ≥ 85–88% after 20 epochs (full CIFAR-10 ResNet-18 benchmark)

---

## 8. Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `RuntimeError: Address already in use` | Change `master_port` in config (e.g. 29501) |
| Slow training on CPU | Reduce `batch_size` to 64 or `epochs` to 5 for testing |
| `FileNotFoundError` for config | Make sure you are running from the project root directory |
| `ImportError` for tqdm/yaml | Run `pip install -r requirements.txt` again inside the activated venv |
| Processes hang on Windows | Set `num_workers: 0` in config (already the default) |
| `Set-ExecutionPolicy` error | Run PowerShell as Administrator; run the Set-ExecutionPolicy command |

---

## 9. What to Analyse from CSV Logs

Open the CSV files in Excel or Python:

```python
import pandas as pd
import matplotlib.pyplot as plt

df2 = pd.read_csv("logs/sync_resnet18_cifar10_world2.csv")
df4 = pd.read_csv("logs/sync_resnet18_cifar10_world4.csv")

# Training time comparison
print("Avg epoch time (2 workers):", df2["epoch_time_seconds"].mean())
print("Avg epoch time (4 workers):", df4["epoch_time_seconds"].mean())

# Throughput comparison
print("Avg throughput (2 workers):", df2["throughput_imgs_per_sec"].mean())
print("Avg throughput (4 workers):", df4["throughput_imgs_per_sec"].mean())

# Accuracy curves
plt.plot(df2["epoch"], df2["val_acc_top1"], label="world=2")
plt.plot(df4["epoch"], df4["val_acc_top1"], label="world=4")
plt.xlabel("Epoch"); plt.ylabel("Val Acc Top-1 (%)"); plt.legend(); plt.show()
```

Key observations to document:
1. Does training time per epoch decrease as world_size increases?
2. Does throughput scale linearly (ideal) or sub-linearly (communication overhead)?
3. Does final accuracy stay similar across world sizes (confirms DDP correctness)?
