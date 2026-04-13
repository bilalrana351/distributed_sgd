"""
Phase 1 – Synchronous All-Reduce Baseline (PyTorch DDP).

Entry point for both:
  a) torchrun (recommended):
       torchrun --nproc_per_node=4 src/sync_allreduce/train_ddp.py --config configs/phase1_config.yaml

  b) launcher.py (torch.multiprocessing.spawn fallback):
       python src/sync_allreduce/launcher.py --nproc 4 --config configs/phase1_config.yaml

  c) Single-process smoke test (no DDP):
       python src/sync_allreduce/train_ddp.py --config configs/phase1_config.yaml

Environment variables set by torchrun / launcher.py:
    RANK          – global rank of this process
    LOCAL_RANK    – rank on this machine (used for GPU assignment)
    WORLD_SIZE    – total number of processes
    MASTER_ADDR   – rendezvous host (127.0.0.1 for single-machine)
    MASTER_PORT   – rendezvous port
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make sure the project root is importable regardless of cwd
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.common.datasets import get_dataloaders, get_num_classes
from src.common.logging_utils import CSV_FIELDNAMES, CSVLogger, setup_logger
from src.common.metrics import AverageMeter, compute_accuracy, compute_throughput
from src.common.models import count_parameters, create_model


# ─────────────────────────────────────────────────────────────────────────────
#  Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_distributed(backend: str) -> Tuple[int, int, int]:
    """
    Initialise the process group and return (rank, local_rank, world_size).

    Uses the 'env://' init method so it works equally well when called by
    torchrun (which sets RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT)
    or by our custom launcher.py.
    """
    dist.init_process_group(backend=backend, init_method="env://")
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()
    return rank, local_rank, world_size


def _cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
#  Training / evaluation loops
# ─────────────────────────────────────────────────────────────────────────────

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    rank: int,
    epoch: int,
    total_epochs: int,
    log_interval_batches: int,
) -> Tuple[float, float, int]:
    """
    Run one full training epoch.

    Returns:
        (avg_train_loss, avg_train_acc_top1, num_batches)
    """
    model.train()
    loss_meter = AverageMeter("train_loss")
    acc_meter  = AverageMeter("train_acc@1")
    num_batches = 0

    # tqdm progress bar only on rank 0
    iterable = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False) \
               if rank == 0 else loader

    for batch_idx, (images, targets) in enumerate(iterable, start=1):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # marginally faster than zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        top1, = compute_accuracy(outputs.detach(), targets, topk=(1,))
        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(top1,         n=images.size(0))
        num_batches += 1
        if rank == 0 and log_interval_batches > 0 and batch_idx % log_interval_batches == 0:
            print(
                f"[progress] epoch={epoch}/{total_epochs} "
                f"batch={batch_idx}/{len(loader)} "
                f"loss={loss_meter.avg:.4f} acc@1={acc_meter.avg:.2f}%",
                flush=True,
            )

    return loss_meter.avg, acc_meter.avg, num_batches


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float]:
    """
    Run validation.

    Returns:
        (avg_val_loss, val_acc_top1, val_acc_top5)
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    acc1_meter = AverageMeter("val_acc@1")
    acc5_meter = AverageMeter("val_acc@5")

    # Top-5 only makes sense when there are ≥ 5 classes
    topk = (1, 5) if num_classes >= 5 else (1,)

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss    = criterion(outputs, targets)

        accs = compute_accuracy(outputs, targets, topk=topk)
        top1 = accs[0]
        top5 = accs[1] if len(accs) > 1 else 100.0   # trivially 100 % for <5 classes

        loss_meter.update(loss.item(), n=images.size(0))
        acc1_meter.update(top1,        n=images.size(0))
        acc5_meter.update(top5,        n=images.size(0))

    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config: dict) -> None:
    # Keep CPU usage bounded on low-resource laptops to avoid full-system freezes.
    max_cpu_threads = int(config.get("max_cpu_threads", 0))
    if max_cpu_threads > 0:
        torch.set_num_threads(max_cpu_threads)
        torch.set_num_interop_threads(max(1, min(2, max_cpu_threads // 2)))

    # ── Distributed setup ────────────────────────────────────────────────────
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))
    use_distributed = config.get("distributed", True) and world_size_env > 1

    if use_distributed:
        backend = config.get("backend", "gloo")
        rank, local_rank, world_size = _init_distributed(backend)
    else:
        rank = local_rank = 0
        world_size = 1

    # ── Device selection ─────────────────────────────────────────────────────
    if torch.cuda.is_available() and config.get("use_cuda", True):
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    logger = setup_logger(rank)
    if rank == 0:
        logger.info(
            f"Starting Phase-1 DDP baseline | "
            f"world_size={world_size} | device={device} | "
            f"backend={config.get('backend', 'gloo') if use_distributed else 'N/A (single process)'}"
        )
        if max_cpu_threads > 0:
            logger.info(f"CPU thread cap enabled: max_cpu_threads={max_cpu_threads}")

    # ── Dataset & data loaders ───────────────────────────────────────────────
    num_classes = get_num_classes(config["dataset"])
    train_loader, val_loader, train_sampler = get_dataloaders(
        dataset_name=config["dataset"],
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 0),
        distributed=use_distributed,
        rank=rank,
        world_size=world_size,
        data_root=config.get("data_root", "./data"),
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = create_model(config["model"], num_classes=num_classes)
    model = model.to(device)

    if rank == 0:
        logger.info(
            f"Model: {config['model']} | "
            f"Params: {count_parameters(model):,} | "
            f"Dataset: {config['dataset']} ({num_classes} classes)"
        )

    if use_distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )

    # ── Loss / optimiser / scheduler ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.1),
        momentum=config.get("momentum", 0.9),
        weight_decay=config.get("weight_decay", 5e-4),
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.get("lr_milestones", [100, 150]),
        gamma=config.get("lr_gamma", 0.1),
    )

    # ── CSV logger (rank-0 only) ─────────────────────────────────────────────
    csv_logger: Optional[CSVLogger] = None
    if rank == 0:
        log_filename = (
            f"sync_{config['model']}_{config['dataset']}_world{world_size}.csv"
        )
        log_path = os.path.join(config.get("log_dir", "./logs"), log_filename)
        csv_logger = CSVLogger(log_path, CSV_FIELDNAMES)
        logger.info(f"CSV log → {log_path}")

    # ── Training loop ────────────────────────────────────────────────────────
    total_epochs = config["epochs"]

    for epoch in range(1, total_epochs + 1):

        # DistributedSampler must know the epoch to reshuffle deterministically
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t_start = time.perf_counter()
        train_loss, train_acc, num_batches = _train_epoch(
            model, train_loader, criterion, optimizer, device,
            rank, epoch, total_epochs, int(config.get("log_interval_batches", 100)),
        )
        epoch_time = time.perf_counter() - t_start

        val_loss, val_acc1, val_acc5 = _evaluate(
            model, val_loader, criterion, device, num_classes,
        )
        scheduler.step()

        throughput = compute_throughput(
            config["batch_size"], world_size, epoch_time, num_batches
        )

        # ── Rank-0 logging ───────────────────────────────────────────────────
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch [{epoch:>3}/{total_epochs}] "
                f"lr={current_lr:.5f} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}% | "
                f"val_loss={val_loss:.4f}  val_acc@1={val_acc1:.2f}%  val_acc@5={val_acc5:.2f}% | "
                f"time={epoch_time:.1f}s  throughput={throughput:.0f} img/s"
            )

            assert csv_logger is not None
            csv_logger.log({
                "epoch":                  epoch,
                "train_loss":             round(train_loss,  6),
                "val_loss":               round(val_loss,    6),
                "val_acc_top1":           round(val_acc1,    4),
                "val_acc_top5":           round(val_acc5,    4),
                "epoch_time_seconds":     round(epoch_time,  3),
                "throughput_imgs_per_sec": round(throughput, 2),
                "world_size":             world_size,
                "batch_size":             config["batch_size"],
            })

            # Checkpoint
            if epoch % config.get("checkpoint_every", 5) == 0:
                ckpt_dir  = config.get("checkpoint_dir", "./checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(
                    ckpt_dir,
                    f"sync_{config['model']}_epoch{epoch:03d}_world{world_size}.pt",
                )
                # Unwrap DDP wrapper before saving
                raw_model = model.module if use_distributed else model
                torch.save(
                    {
                        "epoch":       epoch,
                        "model_state": raw_model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "val_acc":     val_acc1,
                        "config":      config,
                    },
                    ckpt_path,
                )
                logger.info(f"Checkpoint saved → {ckpt_path}")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    _cleanup()
    if rank == 0:
        logger.info("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 – Synchronous DDP Baseline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase1_config.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)
    main(cfg)
