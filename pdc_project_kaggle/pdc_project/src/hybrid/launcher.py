"""
Phase 3 – Hybrid Sync/Async launcher.

Creates all shared inter-process objects and uses torch.multiprocessing.spawn
to start (1 PS process + N worker processes) all running
train_hybrid.main_worker().

Shared objects created here:
  ps_model          – nn.Module with parameters in OS shared memory.
  grad_queue        – Queue through which workers push
                      (grads, worker_id, step, loss, step_time, grad_norm).
  param_lock        – Cross-process Lock protecting ps_model param I/O.
  val_queue         – Queue for worker-0 validation results.
  workers_done      – Shared int counter.
  workers_done_lock – Lock protecting workers_done.
  mode_flag         – Shared int (0 = SYNC, 1 = ASYNC).
  sync_round        – Shared int (monotonically increasing sync-round counter).

Usage:
    python src/hybrid/launcher.py --nworkers 2 \\
        --config configs/phase3_hybrid_safe.yaml
"""

from __future__ import annotations

import argparse
import multiprocessing as std_mp
import os
import sys

import torch
import torch.multiprocessing as mp
import yaml

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.common.datasets import get_num_classes
from src.common.models import count_parameters, create_model


# ─────────────────────────────────────────────────────────────────────────────
#  Launch helper
# ─────────────────────────────────────────────────────────────────────────────

def launch(config: dict, nworkers: int) -> None:
    """Spawn 1 PS process + ``nworkers`` worker processes."""
    if nworkers < 1:
        raise ValueError("nworkers must be >= 1")

    world_size = nworkers + 1
    num_classes = get_num_classes(config["dataset"])

    # ── Build shared PS model ────────────────────────────────────────────
    ps_model = create_model(config["model"], num_classes=num_classes)
    ps_model.share_memory()

    n_params = count_parameters(ps_model)
    param_mb = n_params * 4 / (1024 ** 2)

    ctrl_cfg = config.get("controller", {})
    start_mode = ctrl_cfg.get("start_mode", "sync")

    print(
        f"[launcher] Model: {config['model']} | "
        f"params: {n_params:,} ({param_mb:.1f} MB) | "
        f"dataset: {config['dataset']} | "
        f"start_mode: {start_mode}"
    )

    # ── Shared synchronisation primitives (single spawn context) ────────
    ctx = mp.get_context("spawn")
    grad_queue = ctx.Queue(
        maxsize=int(config.get("grad_queue_maxsize", 200))
    )
    val_queue = ctx.Queue(maxsize=64)

    param_lock = ctx.Lock()
    workers_done = ctx.Value("i", 0)
    workers_done_lock = ctx.Lock()
    mode_flag = ctx.Value("i", 0)
    sync_round = ctx.Value("i", 0)

    print(
        f"[launcher] Spawning {world_size} processes "
        f"(1 PS + {nworkers} workers) | "
        f"total_steps={config['total_steps']} per worker | "
        f"batch_size={config['batch_size']}"
    )

    from src.hybrid.train_hybrid import main_worker

    mp.spawn(
        fn=main_worker,
        args=(
            world_size,
            ps_model,
            grad_queue,
            param_lock,
            val_queue,
            workers_done,
            workers_done_lock,
            mode_flag,
            sync_round,
            config,
        ),
        nprocs=world_size,
        join=True,
        daemon=False,
    )

    print("[launcher] All processes finished.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 – Hybrid Sync/Async launcher."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase3_hybrid_safe.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--nworkers",
        type=int,
        default=2,
        help="Number of worker processes (PS is additional).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    std_mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    args = _parse_args()
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    cfg["num_worker_procs"] = args.nworkers
    launch(cfg, nworkers=args.nworkers)
