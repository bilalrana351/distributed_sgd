import unittest

from src.hybrid.controller import (
    MODE_ASYNC,
    MODE_SYNC,
    ModeController,
)


NEUTRAL_SIGNALS = {
    "worker_time_var": 0.15,
    "grad_norm_var": 0.75,
    "loss_trend": 0.0,
}


class ModeControllerConfigurationTests(unittest.TestCase):
    def test_rejects_unknown_start_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "start_mode"):
            ModeController({"start_mode": "automatic"})

    def test_rejects_non_positive_evaluation_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "eval_interval"):
            ModeController({"eval_interval": 0})

    def test_rejects_non_positive_decision_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "decision_window"):
            ModeController({"decision_window": 0})

    def test_rejects_negative_cooldown(self) -> None:
        with self.assertRaisesRegex(ValueError, "cooldown_steps"):
            ModeController({"cooldown_steps": -1})


class ModeControllerTransitionTests(unittest.TestCase):
    def test_switches_to_async_after_consecutive_straggler_signals(self) -> None:
        controller = ModeController(
            {
                "cooldown_steps": 0,
                "decision_window": 2,
                "straggler_var_high": 0.2,
                "grad_var_low": 0.5,
            }
        )
        signals = {
            "worker_time_var": 0.3,
            "grad_norm_var": 0.4,
            "loss_trend": 0.0,
        }

        self.assertEqual(
            controller.evaluate(signals, global_step=1),
            (MODE_SYNC, False, ""),
        )
        self.assertEqual(
            controller.evaluate(signals, global_step=2),
            (MODE_ASYNC, True, "straggler_detected"),
        )

    def test_neutral_signal_resets_the_decision_window(self) -> None:
        controller = ModeController(
            {
                "cooldown_steps": 0,
                "decision_window": 2,
                "straggler_var_high": 0.2,
                "grad_var_low": 0.5,
            }
        )
        trigger = {
            "worker_time_var": 0.3,
            "grad_norm_var": 0.4,
            "loss_trend": 0.0,
        }

        controller.evaluate(trigger, global_step=1)
        controller.evaluate(NEUTRAL_SIGNALS, global_step=2)
        mode, switched, _ = controller.evaluate(trigger, global_step=3)

        self.assertEqual(mode, MODE_SYNC)
        self.assertFalse(switched)


if __name__ == "__main__":
    unittest.main()
