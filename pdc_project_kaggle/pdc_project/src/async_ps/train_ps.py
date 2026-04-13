"""
Phase 2 – Asynchronous Parameter Server training.

Architecture (no PyTorch RPC – not available in this CPU-only build):
  - 1 Parameter Server process  (rank 0)
  - N Worker processes          (rank 1 .. N)

Shared objects (created in launcher.py before mp.spawn):
  ps_model          – nn.Module with parameters in shared memory (.share_memory())
  grad_queue        – torch.multiprocessing.Queue  (workers push gradients here)
  param_lock        – multiprocessing.Lock  (protects writes/reads to ps_model params)
  val_queue         – torch.multiprocessing.Queue  (worker-0 pushes val results here)
  workers_done      – multiprocessing.Value('i', 0)
  workers_done_lock – multiprocessing.Lock  (protects workers_done increment)

Key design decisions:
  • Workers push gradients NON-BLOCKING (queue.put) → true async behaviour.
  • PS processes gradients one-at-a-time from the queue.
  • Staleness = global_step_before_this_update - worker_local_step.
  • Parameter pulls by workers: acquire param_lock, copy ps_model params to
    local model, release lock.  Workers only copy learnable params (BN running
    stats stay on the worker → correct eval behaviour for worker-0 validation).
  • Validation: worker-0 runs it on its LOCAL model (has correct BN stats)
    and sends results to PS via val_queue.  PS records in PS CSV.

Launch via:
    python src/async_ps/launcher.py --nworkers 2 --config configs/phase2_async_ps_safe.yaml
"""

from __future__ import annotations

import os
import queue
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.async_ps.ps_node import (
    PS_LOG_FIELDS,
    STALENESS_LOG_FIELDS,
    WORKER_LOG_FIELDS,
    StalenessTracker,
)
from src.common.datasets import get_dataloaders, get_num_classes
from src.common.logging_utils import CSVLogger, setup_logger
from src.common.metrics import AverageMeter, compute_accuracy
from src.common.models import create_model


# ─────────────────────────────────────────────────────────────────────────────
#  Shared validation helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Run validation; returns (avg_loss, acc_top1, acc_top5).

    max_batches: if set, stop after this many batches (quick approximate eval).
    Full val on CIFAR-10 with ResNet-18 on CPU takes ~2 min; setting
    max_batches=30 gives a ~20-second estimate which is practical mid-training.
    """
    model.eval()
    loss_m = AverageMeter("val_loss")
    acc1_m = AverageMeter("val_acc1")
    acc5_m = AverageMeter("val_acc5")
    topk   = (1, 5) if num_classes >= 5 else (1,)

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images  = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss    = criterion(outputs, targets)
        accs    = compute_accuracy(outputs, targets, topk=topk)
        top1    = accs[0]
        top5    = accs[1] if len(accs) > 1 else 100.0
        loss_m.update(loss.item(), n=images.size(0))
        acc1_m.update(top1,        n=images.size(0))
        acc5_m.update(top5,        n=images.size(0))

    model.train()  # restore training mode
    return loss_m.avg, acc1_m.avg, acc5_m.avg


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter Server role  (rank 0)
# ─────────────────────────────────────────────────────────────────────────────

def _run_ps(
    ps_model:          nn.Module,
    grad_queue:        Any,
    param_lock:        Any,
    val_queue:         Any,
    workers_done:      Any,
    num_workers:       int,
    config:            dict,
) -> None:
    """
    PS main loop.

    Continuously dequeues gradient pushes, applies async SGD updates to the
    shared model, and logs staleness.  Validation metrics arrive from worker-0
    via val_queue and are flushed to the PS summary CSV.
    """
    logger = setup_logger(rank=0, name="async_ps")
    logger.info(f"[PS] Waiting for {num_workers} worker(s).")

    device      = torch.device("cpu")
    lr          = float(config.get("lr", 0.05))
    log_dir     = config.get("log_dir", "./logs")
    num_classes = get_num_classes(config["dataset"])
    run_tag     = (
        f"{config['model']}_{config['dataset']}"
        f"_nw{num_workers}"
    )
    ckpt_dir    = config.get("checkpoint_dir", "./checkpoints")
    val_interval   = int(config.get("val_interval_steps", 100))
    log_interval   = int(config.get("log_interval_steps", 50))
    poll_timeout   = float(config.get("ps_poll_timeout", 0.5))

    # ── CSV loggers ──────────────────────────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    ps_logger = CSVLogger(
        os.path.join(log_dir, f"async_ps_{run_tag}_ps.csv"),
        PS_LOG_FIELDS,
    )
    staleness_logger = CSVLogger(
        os.path.join(log_dir, f"async_ps_{run_tag}_staleness.csv"),
        STALENESS_LOG_FIELDS,
    )
    logger.info(f"[PS] Logs → {log_dir}/async_ps_{run_tag}_*.csv")

    # ── State ────────────────────────────────────────────────────────────────
    tracker         = StalenessTracker()
    global_step     = 0
    start_time      = time.perf_counter()
    last_log_step   = 0

    # ── Main loop ────────────────────────────────────────────────────────────
    while workers_done.value < num_workers:

        # ── Drain val_queue (non-blocking) ───────────────────────────────────
        try:
            while True:
                val_result: Dict = val_queue.get_nowait()
                elapsed = time.perf_counter() - start_time
                avg_s, max_s, min_s, cnt = tracker.drain()
                ps_logger.log({
                    "global_step":         global_step,
                    "elapsed_time":        round(elapsed, 2),
                    "val_loss":            round(val_result["val_loss"], 6),
                    "val_acc_top1":        round(val_result["val_acc1"], 4),
                    "avg_staleness":       avg_s,
                    "max_staleness":       max_s,
                    "min_staleness":       min_s,
                    "updates_in_interval": cnt,
                    "num_workers":         num_workers,
                    "batch_size":          int(config.get("batch_size", 32)),
                })
                logger.info(
                    f"[PS] step={global_step} | "
                    f"val_loss={val_result['val_loss']:.4f} | "
                    f"val_acc@1={val_result['val_acc1']:.2f}% | "
                    f"avg_staleness={avg_s:.2f} | "
                    f"elapsed={elapsed:.1f}s"
                )
        except Exception:
            pass  # queue.Empty expected; any other exception is transient

        # ── Dequeue one gradient push ────────────────────────────────────────
        try:
            item = grad_queue.get(timeout=poll_timeout)
        except Exception:
            continue  # timeout; loop back and check workers_done again

        grads: List[torch.Tensor]
        worker_id: int
        worker_step: int
        grads, worker_id, worker_step = item

        # Staleness: how many PS updates happened since worker computed grads
        staleness = global_step - worker_step
        tracker.record(staleness)

        elapsed = time.perf_counter() - start_time

        # ── Apply gradient update (lock protects shared parameter tensors) ───
        with param_lock:
            for p_ps, grad in zip(ps_model.parameters(), grads):
                p_ps.data.add_(grad, alpha=-lr)

        global_step += 1

        # ── Log staleness detail ─────────────────────────────────────────────
        staleness_logger.log({
            "global_step": global_step,
            "worker_id":   worker_id,
            "worker_step": worker_step,
            "staleness":   staleness,
            "elapsed_time": round(elapsed, 3),
        })

        # ── Periodic console log ─────────────────────────────────────────────
        if global_step % log_interval == 0 and global_step != last_log_step:
            logger.info(
                f"[PS] global_step={global_step} | "
                f"staleness={staleness} | "
                f"from worker_{worker_id} | "
                f"elapsed={elapsed:.1f}s"
            )
            last_log_step = global_step

    # ── All workers done: flush remaining val_queue entries ──────────────────
    logger.info(f"[PS] All {num_workers} workers done. global_step={global_step}")
    try:
        while True:
            val_result = val_queue.get_nowait()
            elapsed = time.perf_counter() - start_time
            avg_s, max_s, min_s, cnt = tracker.drain()
            ps_logger.log({
                "global_step":         global_step,
                "elapsed_time":        round(elapsed, 2),
                "val_loss":            round(val_result["val_loss"], 6),
                "val_acc_top1":        round(val_result["val_acc1"], 4),
                "avg_staleness":       avg_s,
                "max_staleness":       max_s,
                "min_staleness":       min_s,
                "updates_in_interval": cnt,
                "num_workers":         num_workers,
                "batch_size":          int(config.get("batch_size", 32)),
            })
    except Exception:
        pass

    # ── Save final checkpoint ─────────────────────────────────────────────────
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"async_ps_{run_tag}_final.pt")
    with param_lock:
        state = {k: v.clone() for k, v in ps_model.state_dict().items()}
    torch.save({"global_step": global_step, "model_state": state, "config": config}, ckpt_path)
    logger.info(f"[PS] Checkpoint saved → {ckpt_path}")
    logger.info(f"[PS] Total gradient updates: {global_step}")


# ─────────────────────────────────────────────────────────────────────────────
#  Worker role  (rank 1 .. N)
# ─────────────────────────────────────────────────────────────────────────────

def _run_worker(
    rank:              int,
    num_workers:       int,
    ps_model:          nn.Module,
    grad_queue:        Any,
    param_lock:        Any,
    val_queue:         Any,
    workers_done:      Any,
    workers_done_lock: Any,
    config:            dict,
) -> None:
    """
    Worker training loop.

    Steps per iteration:
      1. Forward + backward on a local mini-batch.
      2. Push gradients to PS via grad_queue  (NON-BLOCKING put).
      3. Every param_pull_every steps: pull fresh global params from PS.
      4. Worker-0 only: every val_interval_steps steps, evaluate local model
         and push (val_loss, val_acc1) to val_queue for PS to log.
    """
    worker_id = rank - 1

    # ── Resource caps ────────────────────────────────────────────────────────
    max_threads = int(config.get("max_cpu_threads", 2))
    if max_threads > 0:
        torch.set_num_threads(max_threads)
        torch.set_num_interop_threads(max(1, min(2, max_threads // 2)))

    device         = torch.device("cpu")
    num_classes    = get_num_classes(config["dataset"])
    total_steps    = int(config["total_steps"])
    batch_size     = int(config["batch_size"])
    pull_every      = int(config.get("param_pull_every", 10))
    log_interval    = int(config.get("log_interval_steps", 50))
    val_interval    = int(config.get("val_interval_steps", 100))
    val_max_batches = config.get("val_num_batches", None)   # None = full val set
    log_dir         = config.get("log_dir", "./logs")
    run_tag        = f"{config['model']}_{config['dataset']}_nw{num_workers}"

    # Unique seed per worker so DataLoaders shuffle differently.
    base_seed = int(config.get("seed", 42))
    torch.manual_seed(base_seed + worker_id * 1000)

    # ── Data ────────────────────────────────────────────────────────────────
    num_dl_workers = int(config.get("num_workers_dataloader", 0))
    train_loader, val_loader, _ = get_dataloaders(
        dataset_name = config["dataset"],
        batch_size   = batch_size,
        num_workers  = num_dl_workers,
        distributed  = False,
        data_root    = config.get("data_root", "./data"),
    )

    # ── Local model (forward/backward only; PS model is the authority) ───────
    local_model = create_model(config["model"], num_classes=num_classes)
    local_model = local_model.to(device)
    criterion   = nn.CrossEntropyLoss()

    # Pull initial parameters from PS.
    with param_lock:
        for p_local, p_ps in zip(local_model.parameters(), ps_model.parameters()):
            p_local.data.copy_(p_ps.data)
    print(f"[worker_{worker_id}] Pulled initial params from PS. "
          f"Starting {total_steps} steps.", flush=True)

    # ── CSV logger ───────────────────────────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    w_logger = CSVLogger(
        os.path.join(log_dir, f"async_ps_{run_tag}_worker{worker_id}.csv"),
        WORKER_LOG_FIELDS,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    start_time   = time.perf_counter()
    data_iter    = iter(train_loader)
    local_step   = 0
    param_pulls  = 0
    loss_meter   = AverageMeter("loss")
    acc_meter    = AverageMeter("acc@1")

    for _step in range(total_steps):

        # Cycle through dataset indefinitely.
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)

        images  = images.to(device)
        targets = targets.to(device)

        # ── Forward + backward ───────────────────────────────────────────────
        local_model.train()
        # Zero grads manually (no optimizer to call zero_grad on).
        for p in local_model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        outputs = local_model(images)
        loss    = criterion(outputs, targets)
        loss.backward()

        top1, = compute_accuracy(outputs.detach(), targets, topk=(1,))
        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(top1,         n=images.size(0))

        # ── Push gradients to PS (non-blocking) ──────────────────────────────
        grads = [
            p.grad.detach().clone()
            if p.grad is not None
            else torch.zeros_like(p.data)
            for p in local_model.parameters()
        ]
        grad_queue.put((grads, worker_id, local_step))

        local_step += 1

        # ── Periodic parameter pull from PS ──────────────────────────────────
        if local_step % pull_every == 0:
            with param_lock:
                for p_local, p_ps in zip(local_model.parameters(), ps_model.parameters()):
                    p_local.data.copy_(p_ps.data)
            param_pulls += 1

        # ── Worker-0: periodic validation ────────────────────────────────────
        if worker_id == 0 and local_step % val_interval == 0:
            # Use local model (has fresh BN running stats from training).
            v_loss, v_acc1, _ = _evaluate(
                local_model, val_loader, criterion, device, num_classes,
                max_batches=val_max_batches,
            )
            val_queue.put({"val_loss": v_loss, "val_acc1": v_acc1})
            elapsed = time.perf_counter() - start_time
            print(
                f"[worker_0] step={local_step}/{total_steps} | "
                f"val_loss={v_loss:.4f} | val_acc@1={v_acc1:.2f}% | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        # ── Periodic console + CSV log ────────────────────────────────────────
        elapsed = time.perf_counter() - start_time
        w_logger.log({
            "worker_id":      worker_id,
            "local_step":     local_step,
            "train_loss":     round(loss.item(), 6),
            "train_acc_top1": round(top1, 4),
            "elapsed_time":   round(elapsed, 3),
        })

        if local_step % log_interval == 0:
            print(
                f"[worker_{worker_id}] step={local_step}/{total_steps} | "
                f"loss={loss_meter.avg:.4f} | acc@1={acc_meter.avg:.2f}% | "
                f"pulls={param_pulls} | elapsed={elapsed:.1f}s",
                flush=True,
            )

    # ── Signal completion ─────────────────────────────────────────────────────
    with workers_done_lock:
        workers_done.value += 1

    elapsed = time.perf_counter() - start_time
    print(
        f"[worker_{worker_id}] DONE. "
        f"local_steps={local_step} | param_pulls={param_pulls} | "
        f"total_elapsed={elapsed:.1f}s",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Per-process dispatch  (called by mp.spawn in launcher.py)
# ─────────────────────────────────────────────────────────────────────────────

def main_worker(
    rank:              int,
    world_size:        int,
    ps_model:          nn.Module,
    grad_queue:        Any,
    param_lock:        Any,
    val_queue:         Any,
    workers_done:      Any,
    workers_done_lock: Any,
    config:            dict,
) -> None:
    """
    Entry point for every spawned process.

    rank 0   → Parameter Server
    rank 1…N → Worker (worker_id = rank - 1)
    """
    num_workers = world_size - 1  # rank-0 is PS

    if rank == 0:
        _run_ps(
            ps_model      = ps_model,
            grad_queue    = grad_queue,
            param_lock    = param_lock,
            val_queue     = val_queue,
            workers_done  = workers_done,
            num_workers   = num_workers,
            config        = config,
        )
    else:
        _run_worker(
            rank              = rank,
            num_workers       = num_workers,
            ps_model          = ps_model,
            grad_queue        = grad_queue,
            param_lock        = param_lock,
            val_queue         = val_queue,
            workers_done      = workers_done,
            workers_done_lock = workers_done_lock,
            config            = config,
        )
