"""
Phase 3 – Hybrid Sync/Async training with mode-dependent aggregation.

Architecture (shared-memory, unified PS with dual-mode aggregation):
  - 1 Parameter Server process  (rank 0)
  - N Worker processes          (rank 1 .. N)

The PS operates in two aggregation modes within the **same** process
topology (no process restarts on mode switch):

  SYNC  – PS collects exactly one gradient from each worker, averages
          them, and applies a single update.  Workers wait for the
          round to complete via the ``sync_round`` counter.

  ASYNC – PS applies each gradient immediately as it arrives.  Workers
          push and continue without waiting (same as Phase 2).

A ``ModeController`` evaluates runtime signals at fixed intervals and
may trigger a bi-directional switch with cooldown/hysteresis.

Shared objects (created in launcher.py before mp.spawn):
  ps_model          – nn.Module with parameters in shared memory
  grad_queue        – mp.Queue  (workers push gradients)
  param_lock        – Lock      (protects ps_model reads/writes)
  val_queue         – mp.Queue  (worker-0 pushes validation results)
  workers_done      – Value('i')  (completion counter)
  workers_done_lock – Lock        (protects counter)
  mode_flag         – Value('i')  (MODE_SYNC=0 or MODE_ASYNC=1)
  sync_round        – Value('i')  (monotonically increasing; incremented
                                    when a sync round completes)

Launch via:
    python src/hybrid/launcher.py --nworkers 2 \\
        --config configs/phase3_hybrid_safe.yaml
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.async_ps.ps_node import StalenessTracker
from src.common.datasets import get_dataloaders, get_num_classes
from src.common.logging_utils import CSVLogger, setup_logger
from src.common.metrics import AverageMeter, compute_accuracy
from src.common.models import create_model
from src.hybrid.controller import (
    MODE_ASYNC,
    MODE_NAMES,
    MODE_SYNC,
    SWITCH_LOG_FIELDS,
    ModeController,
)
from src.hybrid.metrics_collector import MetricsCollector

# ─────────────────────────────────────────────────────────────────────────────
#  CSV schemas for Phase 3 logs
# ─────────────────────────────────────────────────────────────────────────────

HYBRID_PS_LOG_FIELDS: List[str] = [
    "global_step",
    "elapsed_time",
    "current_mode",
    "val_loss",
    "val_acc_top1",
    "avg_staleness",
    "max_staleness",
    "min_staleness",
    "updates_in_interval",
    "throughput_img_per_sec",
    "num_workers",
    "batch_size",
    "scenario_id",
]

HYBRID_STALENESS_LOG_FIELDS: List[str] = [
    "global_step",
    "worker_id",
    "worker_step",
    "staleness",
    "current_mode",
    "elapsed_time",
]

HYBRID_WORKER_LOG_FIELDS: List[str] = [
    "worker_id",
    "local_step",
    "train_loss",
    "train_acc_top1",
    "current_mode",
    "step_time",
    "elapsed_time",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Shared validation helper (identical to Phase 2)
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
    """Run validation; returns (avg_loss, acc_top1, acc_top5)."""
    model.eval()
    loss_m = AverageMeter("val_loss")
    acc1_m = AverageMeter("val_acc1")
    acc5_m = AverageMeter("val_acc5")
    topk = (1, 5) if num_classes >= 5 else (1,)

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        accs = compute_accuracy(outputs, targets, topk=topk)
        top1 = accs[0]
        top5 = accs[1] if len(accs) > 1 else 100.0
        loss_m.update(loss.item(), n=images.size(0))
        acc1_m.update(top1, n=images.size(0))
        acc5_m.update(top5, n=images.size(0))

    model.train()
    return loss_m.avg, acc1_m.avg, acc5_m.avg


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter Server role  (rank 0)
# ─────────────────────────────────────────────────────────────────────────────

def _run_ps(
    ps_model: nn.Module,
    grad_queue: Any,
    param_lock: Any,
    val_queue: Any,
    workers_done: Any,
    mode_flag: Any,
    sync_round: Any,
    num_workers: int,
    config: dict,
) -> None:
    """
    PS main loop with dual-mode gradient aggregation.

    SYNC mode  → buffer N gradients, average, apply once, release workers.
    ASYNC mode → apply each gradient immediately as it arrives.
    """
    logger = setup_logger(rank=0, name="hybrid_ps")
    logger.info(f"[PS] Starting hybrid PS with {num_workers} worker(s).")

    device = torch.device("cpu")
    lr = float(config.get("lr", 0.05))
    log_dir = config.get("log_dir", "./logs")
    run_tag = f"{config['model']}_{config['dataset']}_nw{num_workers}"
    _rid = str(config.get("run_id", "")).strip()
    if _rid:
        run_tag = f"{run_tag}_{_rid}"
    ckpt_dir = config.get("checkpoint_dir", "./checkpoints")
    log_interval = int(config.get("log_interval_steps", 50))
    poll_timeout = float(config.get("ps_poll_timeout", 0.5))
    scenario_id = str(config.get("scenario_id", "")).strip()
    batch_size = int(config.get("batch_size", 32))

    # mode_lock: "none" (default) | "sync" | "async"
    # When locked the controller is disabled and the mode never changes.
    mode_lock = str(config.get("mode_lock", "none")).strip().lower()
    controller_enabled = mode_lock == "none"

    # ── Controller + metrics ─────────────────────────────────────────────
    ctrl_cfg = config.get("controller", {})
    # If mode_lock overrides start_mode, patch ctrl_cfg before constructing
    if mode_lock in ("sync", "async"):
        ctrl_cfg = dict(ctrl_cfg)
        ctrl_cfg["start_mode"] = mode_lock
    controller = ModeController(ctrl_cfg)
    metrics = MetricsCollector(
        num_workers=num_workers,
        ema_alpha=float(ctrl_cfg.get("ema_alpha", 0.05)),
        window_size=int(ctrl_cfg.get("window_size", 20)),
    )

    current_mode = controller.current_mode
    mode_flag.value = current_mode
    sync_round.value = 0

    # ── CSV loggers ──────────────────────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    ps_csv = CSVLogger(
        os.path.join(log_dir, f"hybrid_{run_tag}_ps.csv"),
        HYBRID_PS_LOG_FIELDS,
    )
    stale_csv = CSVLogger(
        os.path.join(log_dir, f"hybrid_{run_tag}_staleness.csv"),
        HYBRID_STALENESS_LOG_FIELDS,
    )
    switch_csv = CSVLogger(
        os.path.join(log_dir, f"hybrid_{run_tag}_switches.csv"),
        SWITCH_LOG_FIELDS,
    )
    logger.info(
        f"[PS] Start mode: {MODE_NAMES[current_mode]} | "
        f"Logs → {log_dir}/hybrid_{run_tag}_*.csv"
    )

    # ── State ────────────────────────────────────────────────────────────
    tracker = StalenessTracker()
    global_step = 0
    start_time = time.perf_counter()
    last_log_step = 0
    last_throughput_step = 0
    last_throughput_time = start_time

    # Sync-mode gradient buffer: worker_id → full payload
    sync_buffer: Dict[int, Tuple] = {}

    # ── Main loop ────────────────────────────────────────────────────────
    while workers_done.value < num_workers:

        # ── Drain val_queue (non-blocking) ───────────────────────────────
        try:
            while True:
                val_result: Dict = val_queue.get_nowait()
                elapsed = time.perf_counter() - start_time
                avg_s, max_s, min_s, cnt = tracker.drain()
                # Throughput: images/sec since last throughput checkpoint
                tp_steps = global_step - last_throughput_step
                tp_elapsed = elapsed - (last_throughput_time - start_time)
                throughput = (
                    round(tp_steps * batch_size / tp_elapsed, 2)
                    if tp_elapsed > 0 else 0.0
                )
                last_throughput_step = global_step
                last_throughput_time = time.perf_counter()
                ps_csv.log({
                    "global_step": global_step,
                    "elapsed_time": round(elapsed, 2),
                    "current_mode": MODE_NAMES[current_mode],
                    "val_loss": round(val_result["val_loss"], 6),
                    "val_acc_top1": round(val_result["val_acc1"], 4),
                    "avg_staleness": avg_s,
                    "max_staleness": max_s,
                    "min_staleness": min_s,
                    "updates_in_interval": cnt,
                    "throughput_img_per_sec": throughput,
                    "num_workers": num_workers,
                    "batch_size": batch_size,
                    "scenario_id": scenario_id,
                })
                logger.info(
                    f"[PS] step={global_step} | "
                    f"mode={MODE_NAMES[current_mode]} | "
                    f"val_loss={val_result['val_loss']:.4f} | "
                    f"val_acc@1={val_result['val_acc1']:.2f}% | "
                    f"throughput={throughput:.1f} img/s | "
                    f"avg_staleness={avg_s:.2f} | "
                    f"elapsed={elapsed:.1f}s"
                )
        except Exception:
            pass

        # ── Dequeue one gradient push ────────────────────────────────────
        try:
            item = grad_queue.get(timeout=poll_timeout)
        except Exception:
            continue

        grads: List[torch.Tensor]
        worker_id: int
        worker_step: int
        loss_val: float
        step_time: float
        grad_norm: float
        grads, worker_id, worker_step, loss_val, step_time, grad_norm = item

        metrics.record_gradient(worker_id, grad_norm, loss_val, step_time)
        elapsed = time.perf_counter() - start_time

        # ── Mode-dependent processing ────────────────────────────────────
        if current_mode == MODE_SYNC:
            sync_buffer[worker_id] = (
                grads, worker_step, loss_val, step_time, grad_norm,
            )

            if len(sync_buffer) == num_workers:
                # All workers contributed — average and apply
                num_params = len(grads)
                avg_grads = [
                    torch.zeros_like(p.data) for p in ps_model.parameters()
                ]
                for wid_buf in sync_buffer.values():
                    for pidx, g in enumerate(wid_buf[0]):
                        avg_grads[pidx].add_(g)
                for g in avg_grads:
                    g.div_(num_workers)

                # Record staleness per worker
                for wid, (_, ws, *_rest) in sync_buffer.items():
                    staleness = global_step - ws
                    tracker.record(staleness)
                    stale_csv.log({
                        "global_step": global_step + 1,
                        "worker_id": wid,
                        "worker_step": ws,
                        "staleness": staleness,
                        "current_mode": "sync",
                        "elapsed_time": round(elapsed, 3),
                    })

                with param_lock:
                    for p_ps, avg_g in zip(ps_model.parameters(), avg_grads):
                        p_ps.data.add_(avg_g, alpha=-lr)

                global_step += 1
                controller.step()

                sync_round.value += 1
                sync_buffer.clear()

        else:  # ASYNC
            staleness = global_step - worker_step
            tracker.record(staleness)
            stale_csv.log({
                "global_step": global_step + 1,
                "worker_id": worker_id,
                "worker_step": worker_step,
                "staleness": staleness,
                "current_mode": "async",
                "elapsed_time": round(elapsed, 3),
            })

            with param_lock:
                for p_ps, grad in zip(ps_model.parameters(), grads):
                    p_ps.data.add_(grad, alpha=-lr)

            global_step += 1
            controller.step()

        # ── Periodic console log ─────────────────────────────────────────
        if global_step % log_interval == 0 and global_step != last_log_step:
            signals = metrics.get_signals()
            logger.info(
                f"[PS] step={global_step} | "
                f"mode={MODE_NAMES[current_mode]} | "
                f"w_time_var={signals['worker_time_var']:.4f} | "
                f"g_norm_var={signals['grad_norm_var']:.4f} | "
                f"loss_trend={signals['loss_trend']:.6f} | "
                f"elapsed={elapsed:.1f}s"
            )
            last_log_step = global_step

        # ── Controller evaluation (skipped when mode_lock is set) ────────
        if (
            controller_enabled
            and global_step > 0
            and global_step % controller.eval_interval == 0
        ):
            signals = metrics.get_signals()
            new_mode, switched, reason = controller.evaluate(signals, global_step)

            if switched:
                old_name = MODE_NAMES[
                    MODE_ASYNC if new_mode == MODE_SYNC else MODE_SYNC
                ]
                new_name = MODE_NAMES[new_mode]
                logger.info(
                    f"[PS] *** MODE SWITCH: {old_name} → {new_name} "
                    f"at step {global_step} | reason={reason} ***"
                )
                switch_csv.log({
                    "global_step": global_step,
                    "elapsed_time": round(elapsed, 2),
                    "old_mode": old_name,
                    "new_mode": new_name,
                    "trigger_reason": reason,
                    **signals,
                })

                _handle_transition(
                    current_mode, new_mode,
                    ps_model, param_lock, grad_queue,
                    sync_buffer, sync_round,
                    lr, global_step,
                )
                current_mode = new_mode
                mode_flag.value = current_mode

    # ── All workers done ─────────────────────────────────────────────────
    logger.info(
        f"[PS] All {num_workers} workers done. "
        f"global_step={global_step}"
    )

    # Flush remaining val_queue entries
    try:
        while True:
            val_result = val_queue.get_nowait()
            elapsed = time.perf_counter() - start_time
            avg_s, max_s, min_s, cnt = tracker.drain()
            tp_steps = global_step - last_throughput_step
            tp_elapsed = elapsed - (last_throughput_time - start_time)
            throughput = (
                round(tp_steps * batch_size / tp_elapsed, 2)
                if tp_elapsed > 0 else 0.0
            )
            last_throughput_step = global_step
            last_throughput_time = time.perf_counter()
            ps_csv.log({
                "global_step": global_step,
                "elapsed_time": round(elapsed, 2),
                "current_mode": MODE_NAMES[current_mode],
                "val_loss": round(val_result["val_loss"], 6),
                "val_acc_top1": round(val_result["val_acc1"], 4),
                "avg_staleness": avg_s,
                "max_staleness": max_s,
                "min_staleness": min_s,
                "updates_in_interval": cnt,
                "throughput_img_per_sec": throughput,
                "num_workers": num_workers,
                "batch_size": batch_size,
                "scenario_id": scenario_id,
            })
    except Exception:
        pass

    # Save final checkpoint
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"hybrid_{run_tag}_final.pt")
    with param_lock:
        state = {k: v.clone() for k, v in ps_model.state_dict().items()}
    torch.save(
        {
            "global_step": global_step,
            "model_state": state,
            "config": config,
            "total_switches": controller.total_switches,
            "switch_history": controller.switch_history,
        },
        ckpt_path,
    )
    logger.info(f"[PS] Checkpoint → {ckpt_path}")
    logger.info(
        f"[PS] Total updates: {global_step} | "
        f"Total mode switches: {controller.total_switches}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Mode transition helper
# ─────────────────────────────────────────────────────────────────────────────

def _handle_transition(
    old_mode: int,
    new_mode: int,
    ps_model: nn.Module,
    param_lock: Any,
    grad_queue: Any,
    sync_buffer: Dict[int, Tuple],
    sync_round: Any,
    lr: float,
    global_step: int,
) -> None:
    """
    Perform safe bookkeeping when the PS switches aggregation mode.

    SYNC → ASYNC:
        Apply any buffered (partial-round) gradients individually,
        then increment sync_round to release workers blocked in
        the polling loop.

    ASYNC → SYNC:
        Drain remaining async-era gradients from the queue so that
        sync mode starts with a clean slate.
    """
    if old_mode == MODE_SYNC and new_mode == MODE_ASYNC:
        # Apply partial-round gradients individually
        for wid in list(sync_buffer.keys()):
            wid_grads = sync_buffer[wid][0]
            with param_lock:
                for p_ps, grad in zip(ps_model.parameters(), wid_grads):
                    p_ps.data.add_(grad, alpha=-lr)
        # Release any workers waiting on the sync round
        sync_round.value += 1
        sync_buffer.clear()

    elif old_mode == MODE_ASYNC and new_mode == MODE_SYNC:
        # Drain queue — process remaining async gradients before
        # switching, so sync mode starts with a clean queue.
        while True:
            try:
                item = grad_queue.get_nowait()
                grads = item[0]
                with param_lock:
                    for p_ps, grad in zip(ps_model.parameters(), grads):
                        p_ps.data.add_(grad, alpha=-lr)
            except Exception:
                break


# ─────────────────────────────────────────────────────────────────────────────
#  Worker role  (rank 1 .. N)
# ─────────────────────────────────────────────────────────────────────────────

def _run_worker(
    rank: int,
    num_workers: int,
    ps_model: nn.Module,
    grad_queue: Any,
    param_lock: Any,
    val_queue: Any,
    workers_done: Any,
    workers_done_lock: Any,
    mode_flag: Any,
    sync_round: Any,
    config: dict,
) -> None:
    """
    Worker training loop with mode-aware gradient push behaviour.

    In SYNC  mode: push gradient → wait for sync_round increment → pull.
    In ASYNC mode: push gradient → continue (pull every N steps).
    """
    worker_id = rank - 1

    # ── Resource caps ────────────────────────────────────────────────────
    max_threads = int(config.get("max_cpu_threads", 2))
    if max_threads > 0:
        torch.set_num_threads(max_threads)
        torch.set_num_interop_threads(max(1, min(2, max_threads // 2)))

    device = torch.device("cpu")
    num_classes = get_num_classes(config["dataset"])
    total_steps = int(config["total_steps"])
    batch_size = int(config["batch_size"])
    pull_every = int(config.get("param_pull_every", 10))
    log_interval = int(config.get("log_interval_steps", 50))
    val_interval = int(config.get("val_interval_steps", 100))
    val_max_batches = config.get("val_num_batches", None)
    log_dir = config.get("log_dir", "./logs")
    run_tag = f"{config['model']}_{config['dataset']}_nw{num_workers}"
    _rid = str(config.get("run_id", "")).strip()
    if _rid:
        run_tag = f"{run_tag}_{_rid}"
    sync_poll_interval = float(config.get("sync_poll_interval", 0.005))

    # ── Straggler simulation (Phase 3 testing) ───────────────────────────
    straggler_cfg = config.get("straggler", {})
    straggler_on = (
        straggler_cfg.get("enabled", False)
        and worker_id == int(straggler_cfg.get("worker_id", -1))
    )
    straggler_delay = float(straggler_cfg.get("delay_seconds", 0.0))
    straggler_start = int(straggler_cfg.get("start_step", 0))
    straggler_end = int(straggler_cfg.get("end_step", total_steps))

    # ── Reproducibility ──────────────────────────────────────────────────
    base_seed = int(config.get("seed", 42))
    torch.manual_seed(base_seed + worker_id * 1000)

    # ── Data ─────────────────────────────────────────────────────────────
    num_dl_workers = int(config.get("num_workers_dataloader", 0))
    train_loader, val_loader, _ = get_dataloaders(
        dataset_name=config["dataset"],
        batch_size=batch_size,
        num_workers=num_dl_workers,
        distributed=False,
        data_root=config.get("data_root", "./data"),
    )

    # ── Local model ──────────────────────────────────────────────────────
    local_model = create_model(config["model"], num_classes=num_classes)
    local_model = local_model.to(device)
    criterion = nn.CrossEntropyLoss()

    with param_lock:
        for p_local, p_ps in zip(
            local_model.parameters(), ps_model.parameters()
        ):
            p_local.data.copy_(p_ps.data)

    print(
        f"[worker_{worker_id}] Pulled initial params. "
        f"Starting {total_steps} steps.",
        flush=True,
    )

    # ── CSV logger ───────────────────────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    w_csv = CSVLogger(
        os.path.join(log_dir, f"hybrid_{run_tag}_worker{worker_id}.csv"),
        HYBRID_WORKER_LOG_FIELDS,
    )

    # ── Training loop ────────────────────────────────────────────────────
    start_time = time.perf_counter()
    data_iter = iter(train_loader)
    local_step = 0
    param_pulls = 0
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc@1")

    for _step in range(total_steps):
        mode = mode_flag.value

        # ── Optional straggler delay ─────────────────────────────────────
        if (
            straggler_on
            and straggler_start <= local_step < straggler_end
        ):
            time.sleep(straggler_delay)

        # ── Get batch (cycle dataset indefinitely) ───────────────────────
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)

        images = images.to(device)
        targets = targets.to(device)

        # ── Forward + backward (timed) ───────────────────────────────────
        local_model.train()
        for p in local_model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        t0 = time.perf_counter()
        outputs = local_model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        step_time = time.perf_counter() - t0

        top1, = compute_accuracy(outputs.detach(), targets, topk=(1,))
        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(top1, n=images.size(0))

        # ── Gradient norm (scalar, for controller metrics) ───────────────
        grad_norm = sum(
            p.grad.pow(2).sum().item()
            for p in local_model.parameters()
            if p.grad is not None
        ) ** 0.5

        # ── Clone gradients and push to PS ───────────────────────────────
        grads = [
            p.grad.detach().clone()
            if p.grad is not None
            else torch.zeros_like(p.data)
            for p in local_model.parameters()
        ]
        grad_queue.put(
            (grads, worker_id, local_step, loss.item(), step_time, grad_norm)
        )

        local_step += 1

        # ── Mode-dependent waiting / param pull ──────────────────────────
        if mode == MODE_SYNC:
            current_round = sync_round.value
            while sync_round.value <= current_round:
                if mode_flag.value != MODE_SYNC:
                    break
                time.sleep(sync_poll_interval)

            with param_lock:
                for p_local, p_ps in zip(
                    local_model.parameters(), ps_model.parameters()
                ):
                    p_local.data.copy_(p_ps.data)
            param_pulls += 1

        else:  # ASYNC
            if local_step % pull_every == 0:
                with param_lock:
                    for p_local, p_ps in zip(
                        local_model.parameters(), ps_model.parameters()
                    ):
                        p_local.data.copy_(p_ps.data)
                param_pulls += 1

        # ── Worker-0: periodic validation ────────────────────────────────
        if worker_id == 0 and local_step % val_interval == 0:
            v_loss, v_acc1, _ = _evaluate(
                local_model, val_loader, criterion, device, num_classes,
                max_batches=val_max_batches,
            )
            val_queue.put({"val_loss": v_loss, "val_acc1": v_acc1})
            elapsed = time.perf_counter() - start_time
            print(
                f"[worker_0] step={local_step}/{total_steps} | "
                f"mode={MODE_NAMES.get(mode, '?')} | "
                f"val_loss={v_loss:.4f} | val_acc@1={v_acc1:.2f}% | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        # ── Periodic CSV + console log ───────────────────────────────────
        elapsed = time.perf_counter() - start_time
        w_csv.log({
            "worker_id": worker_id,
            "local_step": local_step,
            "train_loss": round(loss.item(), 6),
            "train_acc_top1": round(top1, 4),
            "current_mode": MODE_NAMES.get(mode, "unknown"),
            "step_time": round(step_time, 6),
            "elapsed_time": round(elapsed, 3),
        })

        if local_step % log_interval == 0:
            print(
                f"[worker_{worker_id}] step={local_step}/{total_steps} | "
                f"mode={MODE_NAMES.get(mode, '?')} | "
                f"loss={loss_meter.avg:.4f} | acc@1={acc_meter.avg:.2f}% | "
                f"pulls={param_pulls} | elapsed={elapsed:.1f}s",
                flush=True,
            )

    # ── Signal completion ────────────────────────────────────────────────
    with workers_done_lock:
        workers_done.value += 1

    elapsed = time.perf_counter() - start_time
    print(
        f"[worker_{worker_id}] DONE. "
        f"steps={local_step} | pulls={param_pulls} | "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Per-process dispatch  (called by mp.spawn in launcher.py)
# ─────────────────────────────────────────────────────────────────────────────

def main_worker(
    rank: int,
    world_size: int,
    ps_model: nn.Module,
    grad_queue: Any,
    param_lock: Any,
    val_queue: Any,
    workers_done: Any,
    workers_done_lock: Any,
    mode_flag: Any,
    sync_round: Any,
    config: dict,
) -> None:
    """
    Entry point for every spawned process.

    rank 0   → Parameter Server (hybrid dual-mode)
    rank 1…N → Worker (worker_id = rank - 1)
    """
    num_workers = world_size - 1

    if rank == 0:
        _run_ps(
            ps_model=ps_model,
            grad_queue=grad_queue,
            param_lock=param_lock,
            val_queue=val_queue,
            workers_done=workers_done,
            mode_flag=mode_flag,
            sync_round=sync_round,
            num_workers=num_workers,
            config=config,
        )
    else:
        _run_worker(
            rank=rank,
            num_workers=num_workers,
            ps_model=ps_model,
            grad_queue=grad_queue,
            param_lock=param_lock,
            val_queue=val_queue,
            workers_done=workers_done,
            workers_done_lock=workers_done_lock,
            mode_flag=mode_flag,
            sync_round=sync_round,
            config=config,
        )
