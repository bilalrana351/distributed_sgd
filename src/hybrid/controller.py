"""
Phase 3/4 – Mode controller for hybrid Sync ↔ Async switching.

Implements a threshold-based policy with **cooldown** and
**decision window** to decide when to switch between synchronous
(All-Reduce-style) and asynchronous (Parameter-Server-style)
gradient aggregation.

Switching rules
───────────────
  SYNC → ASYNC :  worker_time_var  > straggler_var_high
                  AND grad_norm_var < grad_var_low
                  (straggler detected AND training is stable)

  ASYNC → SYNC :  grad_norm_var    > grad_var_high          (instability)
                  OR  loss_trend   > loss_trend_threshold   (rising loss)
                  OR  worker_time_var < straggler_var_low   (straggler gone,
                      sync is efficient again — Phase 4 transient scenario)

The condition must hold for ``decision_window`` consecutive
evaluation points, and the mode must have been active for at
least ``cooldown_steps`` global steps before any switch.

Phase 4 addition
────────────────
``straggler_var_low`` enables the full bi-directional cycle needed for
Scenario B (transient heterogeneity): once the straggler disappears and
worker step-times converge, the controller switches back to sync,
recovering the consistency benefits of All-Reduce aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

MODE_SYNC: int = 0
MODE_ASYNC: int = 1

MODE_NAMES: Dict[int, str] = {MODE_SYNC: "sync", MODE_ASYNC: "async"}

SWITCH_LOG_FIELDS: List[str] = [
    "global_step",
    "elapsed_time",
    "old_mode",
    "new_mode",
    "trigger_reason",
    "worker_time_var",
    "grad_norm_var",
    "loss_trend",
]


@dataclass(frozen=True)
class ControllerConfig:
    """Validated configuration for the hybrid mode controller."""

    start_mode: str = "sync"
    eval_interval: int = 50
    cooldown_steps: int = 200
    decision_window: int = 5
    straggler_var_high: float = 0.3
    grad_var_low: float = 0.5
    grad_var_high: float = 1.0
    loss_trend_high: float = 0.05
    straggler_var_low: float = 0.10

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ControllerConfig":
        """Parse the YAML-style mapping and reject unsafe control values."""
        config = cls(
            start_mode=str(values.get("start_mode", "sync")).strip().lower(),
            eval_interval=int(values.get("eval_interval", 50)),
            cooldown_steps=int(values.get("cooldown_steps", 200)),
            decision_window=int(values.get("decision_window", 5)),
            straggler_var_high=float(values.get("straggler_var_high", 0.3)),
            grad_var_low=float(values.get("grad_var_low", 0.5)),
            grad_var_high=float(values.get("grad_var_high", 1.0)),
            loss_trend_high=float(values.get("loss_trend_threshold", 0.05)),
            straggler_var_low=float(values.get("straggler_var_low", 0.10)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.start_mode not in {"sync", "async"}:
            raise ValueError(
                "start_mode must be either 'sync' or 'async', "
                f"got {self.start_mode!r}"
            )
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be greater than zero")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be zero or greater")
        if self.decision_window <= 0:
            raise ValueError("decision_window must be greater than zero")


class ModeController:
    """
    Threshold-based switching policy with bi-directional support.

    Usage inside the PS loop::

        controller = ModeController(ctrl_cfg)
        ...
        controller.step()                      # call once per global_step
        if global_step % controller.eval_interval == 0:
            signals = metrics.get_signals()
            new_mode, switched, reason = controller.evaluate(signals, global_step)
    """

    def __init__(
        self,
        controller_cfg: Mapping[str, Any] | ControllerConfig,
    ) -> None:
        config = (
            controller_cfg
            if isinstance(controller_cfg, ControllerConfig)
            else ControllerConfig.from_mapping(controller_cfg)
        )
        config.validate()
        self.config = config
        self.current_mode = (
            MODE_SYNC if config.start_mode == "sync" else MODE_ASYNC
        )

        # Public aliases retained for callers and experiment logging.
        self.eval_interval = config.eval_interval
        self.cooldown_steps = config.cooldown_steps
        self.decision_window = config.decision_window
        self.straggler_var_high = config.straggler_var_high
        self.grad_var_low = config.grad_var_low
        self.grad_var_high = config.grad_var_high
        self.loss_trend_high = config.loss_trend_high
        self.straggler_var_low = config.straggler_var_low

        # Internal bookkeeping
        self._steps_in_mode: int = 0
        self._consecutive_triggers: int = 0
        self._trigger_reason: str = ""
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
    ) -> Tuple[int, bool, str]:
        """
        Decide whether to switch modes.

        Returns ``(current_mode, switched, trigger_reason)`` where
        *switched* is True only when a transition actually occurred
        and *trigger_reason* is a short label logged to the switch CSV.
        """
        if self._steps_in_mode < self.cooldown_steps:
            return self.current_mode, False, ""

        reason = self._find_trigger_reason(signals)

        if reason:
            self._consecutive_triggers += 1
            self._trigger_reason = reason
        else:
            self._consecutive_triggers = 0
            self._trigger_reason = ""

        if self._consecutive_triggers >= self.decision_window:
            old_mode = self.current_mode
            new_mode = MODE_ASYNC if old_mode == MODE_SYNC else MODE_SYNC
            final_reason = self._trigger_reason

            self.switch_history.append(
                {
                    "global_step": global_step,
                    "old_mode": MODE_NAMES[old_mode],
                    "new_mode": MODE_NAMES[new_mode],
                    "trigger_reason": final_reason,
                    **signals,
                }
            )

            self.current_mode = new_mode
            self._steps_in_mode = 0
            self._consecutive_triggers = 0
            self._total_switches += 1
            return self.current_mode, True, final_reason

        return self.current_mode, False, ""

    def _find_trigger_reason(self, signals: Mapping[str, float]) -> str:
        """Return the active transition reason, or an empty string."""
        if self.current_mode == MODE_SYNC:
            if (
                signals["worker_time_var"] > self.straggler_var_high
                and signals["grad_norm_var"] < self.grad_var_low
            ):
                return "straggler_detected"
            return ""

        if signals["grad_norm_var"] > self.grad_var_high:
            return "grad_instability"
        if signals["loss_trend"] > self.loss_trend_high:
            return "loss_rising"
        if signals["worker_time_var"] < self.straggler_var_low:
            return "straggler_resolved"
        return ""

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def total_switches(self) -> int:
        return self._total_switches

    @property
    def mode_name(self) -> str:
        return MODE_NAMES[self.current_mode]
