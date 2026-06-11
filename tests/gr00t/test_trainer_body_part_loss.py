"""Unit tests for per-body-part loss decomposition in Gr00tTrainer.compute_loss.

Tests that action_loss is correctly sliced into left_arm / right_arm /
left_hand / right_hand and logged via self.log().

Run:
    cd DIDEN_Core/diden_vla/Isaac-GR00T
    python -m pytest tests/gr00t/test_trainer_body_part_loss.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Ensure gr00t is importable
_HERE = Path(__file__).parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_outputs(B: int, H: int, D: int, fill: float = 1.0) -> dict:
    """Build a fake model output dict with known action_loss values."""
    action_loss = torch.full((B, H, D), fill)
    action_mask = torch.ones(B, H, D)
    return {
        "loss": torch.tensor(fill),
        "action_loss": action_loss,
        "action_mask": action_mask,
    }


def _make_trainer_stub():
    """Build a minimal Gr00tTrainer-like object without touching Transformers."""
    from gr00t.experiment.trainer import Gr00tTrainer

    trainer = object.__new__(Gr00tTrainer)
    # Minimal state / args needed by compute_loss
    trainer.loss = None
    trainer.action_offset = None
    trainer._log_calls = []

    class _Args:
        logging_steps = 1   # log every step
        local_rank = -1     # single process

    class _State:
        global_step = 1

    trainer.args = _Args()
    trainer.state = _State()
    trainer._nested_gather = lambda t: t

    def _fake_log(logs, start_time=None):
        trainer._log_calls.append(dict(logs))

    trainer.log = _fake_log
    return trainer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBodyPartLossDecomposition:

    def _run_compute_loss(self, trainer, B, H, D, fill=1.0):
        """Call compute_loss with mocked super() returning fixed outputs."""
        outputs = _make_outputs(B, H, D, fill)

        # Patch super().compute_loss to return our fake (loss, outputs)
        with patch(
            "gr00t.experiment.trainer.Trainer.compute_loss",
            return_value=(outputs["loss"], outputs),
        ):
            from gr00t.experiment.trainer import Gr00tTrainer
            result = Gr00tTrainer.compute_loss(
                trainer,
                model=MagicMock(training=True),
                inputs={},
                return_outputs=False,
                num_items_in_batch=None,
            )
        return result

    def test_arm_only_14dof_logs_left_and_right_arm(self):
        """With 14-DOF action, logs left_arm and right_arm only."""
        trainer = _make_trainer_stub()
        self._run_compute_loss(trainer, B=2, H=4, D=14)

        keys = {k for call in trainer._log_calls for k in call}
        assert "train_loss/left_arm"  in keys
        assert "train_loss/right_arm" in keys
        assert "train_loss/left_hand"  not in keys
        assert "train_loss/right_hand" not in keys

    def test_full_54dof_logs_all_four_parts(self):
        """With 54-DOF action, logs all four body parts."""
        trainer = _make_trainer_stub()
        self._run_compute_loss(trainer, B=2, H=4, D=54)

        keys = {k for call in trainer._log_calls for k in call}
        assert "train_loss/left_arm"   in keys
        assert "train_loss/right_arm"  in keys
        assert "train_loss/left_hand"  in keys
        assert "train_loss/right_hand" in keys

    def test_loss_values_are_correct(self):
        """Logged values equal the per-slice mean (all-ones mask, uniform fill)."""
        fill = 0.42
        trainer = _make_trainer_stub()
        self._run_compute_loss(trainer, B=2, H=4, D=54, fill=fill)

        logged = {k: v for call in trainer._log_calls for k, v in call.items()}
        for key in ["train_loss/left_arm", "train_loss/right_arm",
                    "train_loss/left_hand", "train_loss/right_hand"]:
            assert abs(logged[key] - fill) < 1e-5, f"{key}: expected {fill}, got {logged[key]}"

    def test_zero_mask_yields_zero_loss(self):
        """A fully-masked body part should contribute 0 loss."""
        from gr00t.experiment.trainer import Gr00tTrainer

        trainer = _make_trainer_stub()
        B, H, D = 2, 4, 54
        action_loss = torch.ones(B, H, D)
        action_mask = torch.ones(B, H, D)
        action_mask[..., 14:34] = 0.0   # zero out left_hand

        outputs = {"loss": torch.tensor(1.0), "action_loss": action_loss, "action_mask": action_mask}

        with patch(
            "gr00t.experiment.trainer.Trainer.compute_loss",
            return_value=(outputs["loss"], outputs),
        ):
            Gr00tTrainer.compute_loss(
                trainer,
                model=MagicMock(training=True),
                inputs={},
                return_outputs=False,
                num_items_in_batch=None,
            )

        logged = {k: v for call in trainer._log_calls for k, v in call.items()}
        assert logged["train_loss/left_hand"] == pytest.approx(0.0, abs=1e-5)
        assert logged["train_loss/left_arm"]  == pytest.approx(1.0, abs=1e-5)

    def test_no_action_loss_in_outputs_skips_logging(self):
        """If outputs has no action_loss key, no body-part logs are emitted."""
        from gr00t.experiment.trainer import Gr00tTrainer

        trainer = _make_trainer_stub()
        plain_outputs = {"loss": torch.tensor(1.0)}  # no action_loss

        with patch(
            "gr00t.experiment.trainer.Trainer.compute_loss",
            return_value=(plain_outputs["loss"], plain_outputs),
        ):
            Gr00tTrainer.compute_loss(
                trainer,
                model=MagicMock(training=True),
                inputs={},
                return_outputs=False,
                num_items_in_batch=None,
            )

        keys = {k for call in trainer._log_calls for k in call}
        assert "train_loss/left_arm" not in keys

    def test_non_training_mode_skips_logging(self):
        """During eval (model.training=False), no per-part logs emitted."""
        from gr00t.experiment.trainer import Gr00tTrainer

        trainer = _make_trainer_stub()
        outputs = _make_outputs(2, 4, 54)

        with patch(
            "gr00t.experiment.trainer.Trainer.compute_loss",
            return_value=(outputs["loss"], outputs),
        ):
            Gr00tTrainer.compute_loss(
                trainer,
                model=MagicMock(training=False),   # eval mode
                inputs={},
                return_outputs=False,
                num_items_in_batch=None,
            )

        keys = {k for call in trainer._log_calls for k in call}
        assert "train_loss/left_arm" not in keys
