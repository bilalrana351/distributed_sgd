"""
Phase 3 – Mode controller for hybrid Sync ↔ Async switching.

Implements a threshold-based policy with **cooldown** and
**decision window** to decide when to switch between synchronous
(All-Reduce-style) and asynchronous (Parameter-Server-style)
gradient aggregation.

Switching rules
───────────────
  SYNC → ASYNC :  worker_time_var  > straggler_var_high
                  AND grad_norm_var < grad_var_low
                  (straggler detected AND training is stable)

  ASYNC → SYNC :  grad_norm_var > grad_var_high
                  OR  loss_trend  > loss_trend_threshold
                  (training becoming unstable OR loss rising)

The condition must hold for ``decision_window`` consecutive
evaluation points, and the mode must have been active for at
least ``cooldown_steps`` global steps before any switch.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

MODE_SYNC: int = 0
MODE_ASYNC: int = 1

MODE_NAMES: Dict[int, str] = {MODE_SYNC: "sync", MODE_ASYNC: "async"}

SWITCH_LOG_FIELDS: List[str] = [
    "global_step",
    "elapsed_time",
    "old_mode",
    "new_mode",
    "worker_time_var",
    "grad_norm_var",
    "loss_trend",
]


class ModeController:
    """
    Threshold-based switching policy.

    Usage inside the PS loop::

        controller = ModeController(ctrl_cfg)
        ...
        controller.step()                      # call once per global_step
        if global_step % controller.eval_interval == 0:
            signals = metrics.get_signals()
            new_mode, switched = controller.evaluate(signals, global_step)
    """

    def __init__(self, controller_cfg: dict) -> None:
        start = controller_cfg.get("start_mode", "sync")
        self.current_mode: int = MODE_SYNC if start == "sync" else MODE_ASYNC

        self.eval_interval: int = int(controller_cfg.get("eval_interval", 50))
        self.cooldown_steps: int = int(
            controller_cfg.get("cooldown_steps", 200)
        )
        self.decision_window: int = int(
            controller_cfg.get("decision_window", 5)
        )

        # SYNC → ASYNC thresholds
        self.straggler_var_high: float = float(
            controller_cfg.get("straggler_var_high", 0.3)
        )
        self.grad_var_low: float = float(
            controller_cfg.get("grad_var_low", 0.5)
        )

        # ASYNC → SYNC thresholds
        self.grad_var_high: float = float(
            controller_cfg.get("grad_var_high", 1.0)
        )
        self.loss_trend_high: float = float(
            controller_cfg.get("loss_trend_threshold", 0.05)
        )

        # Internal bookkeeping
        self._steps_in_mode: int = 0
        self._consecutive_triggers: int = 0
        self._total_switches: int = 0
        self.switch_history: List[Dict] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def step(self) -> None:
        """Call once per ``global_step`` increment to track cooldown."""
        self._steps_in_mode += 1

    def evaluate(
        self, signals: Dict[str, float], global_step: int
    ) -> Tuple[int, bool]:
        """
        Decide whether to switch modes.

        Returns ``(current_mode, switched)`` where *switched* is True
        only when a transition actually occurred.
        """
        if self._steps_in_mode < self.cooldown_steps:
            return self.current_mode, False

        trigger = False

        if self.current_mode == MODE_SYNC:
            if (
                signals["worker_time_var"] > self.straggler_var_high
                and signals["grad_norm_var"] < self.grad_var_low
            ):
                trigger = True
        else:  # ASYNC
            if (
                signals["grad_norm_var"] > self.grad_var_high
                or signals["loss_trend"] > self.loss_trend_high
            ):
                trigger = True

        if trigger:
            self._consecutive_triggers += 1
        else:
            self._consecutive_triggers = 0

        if self._consecutive_triggers >= self.decision_window:
            old_mode = self.current_mode
            new_mode = MODE_ASYNC if old_mode == MODE_SYNC else MODE_SYNC

            self.switch_history.append(
                {
                    "global_step": global_step,
                    "old_mode": MODE_NAMES[old_mode],
                    "new_mode": MODE_NAMES[new_mode],
                    **signals,
                }
            )

            self.current_mode = new_mode
            self._steps_in_mode = 0
            self._consecutive_triggers = 0
            self._total_switches += 1
            return self.current_mode, True

        return self.current_mode, False

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def total_switches(self) -> int:
        return self._total_switches

    @property
    def mode_name(self) -> str:
        return MODE_NAMES[self.current_mode]
