"""
Phase 2 – Asynchronous Parameter Server launcher.

Creates all shared inter-process objects, then uses torch.multiprocessing.spawn
to start (1 PS process + N worker processes) all running train_ps.main_worker().

Shared objects created here:
  ps_model          – nn.Module with parameters in OS shared memory.
  grad_queue        – Queue through which workers push (grads, worker_id, step).
  param_lock        – Cross-process Lock protecting ps_model param reads/writes.
  val_queue         – Queue through which worker-0 pushes validation results.
  workers_done      – Shared integer counter (incremented by each finished worker).
  workers_done_lock – Lock protecting workers_done increment.

Why no PyTorch RPC?
  torch.distributed.rpc.is_available() returns False for this PyTorch 2.11+cpu
  build.  Instead we use direct shared-memory tensors + multiprocessing primitives,
  which is actually more efficient on a single machine and demonstrates identical
  async-SGD semantics (workers do not synchronise with each other; PS applies
  each gradient push independently as it arrives).

Usage:
    python src/async_ps/launcher.py --nworkers 2 --config configs/phase2_async_ps_safe.yaml
    python src/async_ps/launcher.py --nworkers 1 --config configs/phase2_async_ps_safe.yaml
"""

from __future__ import annotations

import argparse
import multiprocessing as std_mp
import os
import sys

import torch
import torch.multiprocessing as mp
import yaml

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.common.datasets import get_num_classes
from src.common.models import count_parameters, create_model


# ─────────────────────────────────────────────────────────────────────────────
#  Launch helper
# ─────────────────────────────────────────────────────────────────────────────

def launch(config: dict, nworkers: int) -> None:
    """
    Spawn 1 PS process + `nworkers` worker processes.

    All processes run train_ps.main_worker(rank, world_size, shared_objects…).
    """
    if nworkers < 1:
        raise ValueError("nworkers must be >= 1")

    world_size  = nworkers + 1          # rank 0 = PS, ranks 1..N = workers
    num_classes = get_num_classes(config["dataset"])

    # ── Build shared PS model ─────────────────────────────────────────────────
    ps_model = create_model(config["model"], num_classes=num_classes)
    # Move all parameter/buffer tensors to OS shared memory so every spawned
    # process maps the SAME physical pages (zero-copy cross-process reads).
    ps_model.share_memory()

    n_params = count_parameters(ps_model)
    param_mb = n_params * 4 / (1024 ** 2)      # float32 → bytes → MB
    print(
        f"[launcher] Model: {config['model']} | "
        f"params: {n_params:,} ({param_mb:.1f} MB) | "
        f"dataset: {config['dataset']} ({num_classes} classes)"
    )

    # ── Shared synchronisation primitives ─────────────────────────────────────
    # Use mp.Queue (torch.multiprocessing version) so that tensors placed in
    # the queue are serialised via shared-memory handles, not full pickle.
    grad_queue = mp.Queue(maxsize=int(config.get("grad_queue_maxsize", 200)))
    val_queue  = mp.Queue(maxsize=64)

    # Standard multiprocessing Lock / Value for counters and parameter access.
    # We use std_mp here because mp.Lock() and mp.Value() are identical – both
    # come from the same underlying multiprocessing module on Windows.
    param_lock        = std_mp.Lock()
    workers_done      = std_mp.Value("i", 0)
    workers_done_lock = std_mp.Lock()

    print(
        f"[launcher] Spawning {world_size} processes "
        f"(1 PS + {nworkers} worker(s)) | "
        f"total_steps={config['total_steps']} per worker | "
        f"batch_size={config['batch_size']}"
    )

    from src.async_ps.train_ps import main_worker

    mp.spawn(
        fn      = main_worker,
        args    = (
            world_size,
            ps_model,
            grad_queue,
            param_lock,
            val_queue,
            workers_done,
            workers_done_lock,
            config,
        ),
        nprocs  = world_size,
        join    = True,
        daemon  = False,
    )

    print("[launcher] All processes finished.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 – Async PS launcher (shared-memory + mp.Queue)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase2_async_ps_safe.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--nworkers",
        type=int,
        default=2,
        help="Number of worker processes (PS is additional, so total = nworkers+1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Windows requires the freeze_support() / spawn guard.
    std_mp.freeze_support()

    args   = _parse_args()
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    # Allow CLI --nworkers to override config value.
    cfg["num_worker_procs"] = args.nworkers
    launch(cfg, nworkers=args.nworkers)
