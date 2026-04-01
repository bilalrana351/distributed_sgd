"""
Optional local launcher using torch.multiprocessing.spawn.

Use this only when torchrun is unavailable or you prefer the Python API.
For most cases, torchrun is the recommended and simpler approach.

Usage:
    # Launch with 4 worker processes
    python src/sync_allreduce/launcher.py --nproc 4 --config configs/phase1_config.yaml

    # Launch with 2 worker processes
    python src/sync_allreduce/launcher.py --nproc 2 --config configs/phase1_config.yaml

torchrun equivalent (recommended):
    torchrun --nproc_per_node=4 src/sync_allreduce/train_ddp.py --config configs/phase1_config.yaml

Windows note:
    torch.multiprocessing uses 'spawn' by default on Windows, which is
    correct.  Do NOT set start_method='fork' on Windows.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch.multiprocessing as mp
import yaml

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-worker entry function
# ─────────────────────────────────────────────────────────────────────────────

def _worker(rank: int, world_size: int, config: dict) -> None:
    """
    This function is called in each spawned subprocess.

    We set the distributed env-vars here so that train_ddp.main() sees them
    regardless of whether it was launched via torchrun or this script.
    """
    os.environ["RANK"]       = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)    # single-machine: local == global rank
    os.environ["WORLD_SIZE"] = str(world_size)
    # MASTER_ADDR / MASTER_PORT are set by launch() before spawning

    from src.sync_allreduce.train_ddp import main
    main(config)


# ─────────────────────────────────────────────────────────────────────────────
#  Launch helper
# ─────────────────────────────────────────────────────────────────────────────

def launch(config: dict, nproc: int) -> None:
    """Spawn `nproc` training workers on the local machine."""
    master_addr = config.get("master_addr", "127.0.0.1")
    master_port = str(config.get("master_port", 29500))

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    print(
        f"[launcher] Spawning {nproc} worker(s) | "
        f"master={master_addr}:{master_port}"
    )

    mp.spawn(
        fn=_worker,
        args=(nproc, config),
        nprocs=nproc,
        join=True,
        daemon=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spawn local DDP workers for Phase-1 training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase1_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=2,
        help="Number of worker processes to spawn (2 or 4 for Phase 1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)
    launch(cfg, args.nproc)
