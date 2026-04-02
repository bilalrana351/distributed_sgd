"""
Phase 3 – Runtime metrics collector for the hybrid mode controller.

Tracks three signals using exponential moving averages (EMA):

  1. **Worker time variance** – relative standard deviation of per-worker
     mean step-compute times.  High value → one or more workers are
     significantly slower (straggler signal).

  2. **Gradient norm variance** – relative standard deviation of recent
     per-worker gradient L2 norms.  High value → workers are producing
     very different gradient directions (diversity / instability signal).

  3. **Loss trend** – EMA of successive loss deltas.  Positive → loss
     is rising (training instability signal).

All arithmetic is pure Python (no numpy dependency).
"""

from __future__ import annotations

from typing import Dict, List


class MetricsCollector:
    """
    Collects per-gradient telemetry from the PS loop and maintains
    smoothed controller input signals.

    Call ``record_gradient()`` once per gradient received by the PS.
    Call ``get_signals()`` when the controller needs to evaluate.
    """

    def __init__(
        self,
        num_workers: int,
        ema_alpha: float = 0.05,
        window_size: int = 20,
    ) -> None:
        self.num_workers = num_workers
        self.alpha = ema_alpha
        self.window = window_size

        self._step_times: Dict[int, List[float]] = {
            i: [] for i in range(num_workers)
        }
        self._grad_norms: Dict[int, List[float]] = {
            i: [] for i in range(num_workers)
        }

        self._ema_loss: float | None = None
        self._ema_loss_trend: float = 0.0

        self.ema_time_var: float = 0.0
        self.ema_grad_var: float = 0.0

        self._update_count: int = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def record_gradient(
        self,
        worker_id: int,
        grad_norm: float,
        loss: float,
        step_time: float,
    ) -> None:
        """Record one gradient arrival and recompute EMA signals."""
        buf = self._step_times[worker_id]
        buf.append(step_time)
        if len(buf) > self.window:
            del buf[: -self.window]

        buf2 = self._grad_norms[worker_id]
        buf2.append(grad_norm)
        if len(buf2) > self.window:
            del buf2[: -self.window]

        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            diff = loss - self._ema_loss
            self._ema_loss += self.alpha * diff
            self._ema_loss_trend = (
                (1 - self.alpha) * self._ema_loss_trend + self.alpha * diff
            )

        self._update_time_var()
        self._update_grad_var()
        self._update_count += 1

    def get_signals(self) -> Dict[str, float]:
        """Return current smoothed signal values for the controller."""
        return {
            "worker_time_var": round(self.ema_time_var, 6),
            "grad_norm_var": round(self.ema_grad_var, 6),
            "loss_trend": round(self._ema_loss_trend, 6),
        }

    @property
    def update_count(self) -> int:
        return self._update_count

    # ------------------------------------------------------------------ #
    #  Internal signal updates                                            #
    # ------------------------------------------------------------------ #

    def _update_time_var(self) -> None:
        """Relative std-dev of per-worker mean step times."""
        means: List[float] = []
        for wid in range(self.num_workers):
            t = self._step_times[wid]
            if t:
                means.append(sum(t) / len(t))
        if len(means) < 2:
            return
        overall = sum(means) / len(means)
        if overall <= 0:
            return
        var = sum((m - overall) ** 2 for m in means) / len(means)
        rel_std = var ** 0.5 / overall
        self.ema_time_var = (
            (1 - self.alpha) * self.ema_time_var + self.alpha * rel_std
        )

    def _update_grad_var(self) -> None:
        """Relative std-dev of recent per-worker gradient norms."""
        norms: List[float] = []
        for wid in range(self.num_workers):
            g = self._grad_norms[wid]
            if g:
                recent = g[-5:]
                norms.append(sum(recent) / len(recent))
        if len(norms) < 2:
            return
        mean_n = sum(norms) / len(norms)
        if mean_n <= 0:
            return
        var = sum((n - mean_n) ** 2 for n in norms) / len(norms)
        rel_std = var ** 0.5 / mean_n
        self.ema_grad_var = (
            (1 - self.alpha) * self.ema_grad_var + self.alpha * rel_std
        )
