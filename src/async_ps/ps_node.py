"""
Phase 2 – Parameter Server helpers.

Defines:
  - CSV field name constants for all Phase 2 logs.
  - StalenessTracker: lightweight rolling tracker for per-gradient staleness.

These objects are used inside _run_ps() in train_ps.py.
The PS never uses PyTorch RPC (rpc.is_available() == False in this build).
Communication is done via torch.multiprocessing.Queue + shared model memory.
"""

from __future__ import annotations

from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  CSV schemas
# ─────────────────────────────────────────────────────────────────────────────

# Written by the PS once per val_interval_steps global steps.
PS_LOG_FIELDS: List[str] = [
    "global_step",            # PS update counter at the time of this log row
    "elapsed_time",           # seconds since training began
    "val_loss",               # validation cross-entropy (from worker-0 eval)
    "val_acc_top1",           # validation top-1 accuracy %
    "avg_staleness",          # mean staleness over the last log interval
    "max_staleness",          # maximum staleness observed in the interval
    "min_staleness",          # minimum staleness observed in the interval
    "updates_in_interval",    # gradient updates processed since last log row
    "num_workers",            # number of worker processes
    "batch_size",             # per-worker batch size
]

# Written by the PS for every single gradient push (detailed staleness log).
STALENESS_LOG_FIELDS: List[str] = [
    "global_step",    # PS global step AFTER this update
    "worker_id",      # which worker sent this gradient
    "worker_step",    # worker's local step when gradient was computed
    "staleness",      # global_step_before_update - worker_step
    "elapsed_time",   # seconds since start at time of application
]

# Written by each worker once per log_interval_steps local steps.
WORKER_LOG_FIELDS: List[str] = [
    "worker_id",
    "local_step",
    "train_loss",
    "train_acc_top1",
    "elapsed_time",
]


# ─────────────────────────────────────────────────────────────────────────────
#  StalenessTracker
# ─────────────────────────────────────────────────────────────────────────────

class StalenessTracker:
    """
    Rolling statistics for gradient staleness.

    Each gradient push records one staleness value.
    Call drain() to get and reset the accumulated stats.
    """

    def __init__(self) -> None:
        self._buf: List[int] = []
        self._total_updates: int = 0

    def record(self, staleness: int) -> None:
        self._buf.append(staleness)
        self._total_updates += 1

    def drain(self) -> Tuple[float, int, int, int]:
        """
        Return (avg, max, min, count) of buffered values since last drain,
        then reset the buffer.  Returns zeros when buffer is empty.
        """
        if not self._buf:
            return 0.0, 0, 0, 0
        avg = sum(self._buf) / len(self._buf)
        mx  = max(self._buf)
        mn  = min(self._buf)
        cnt = len(self._buf)
        self._buf.clear()
        return round(avg, 4), mx, mn, cnt

    @property
    def total_updates(self) -> int:
        return self._total_updates
