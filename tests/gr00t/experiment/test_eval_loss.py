# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An eval pass has to report eval_loss, not just eval_runtime.

Trainer finds the loss during evaluation either through label_names (from
find_labels(), which scans forward() for a parameter named like "labels") or
through the return_loss fallback (a `return_loss` parameter). Gr00tN1d7.forward
is `forward(self, inputs)`, so neither applies and upstream prediction_step
returns loss=None — evaluation then logs only timing, and the eval-loss ↔
success-rate sweep has nothing to correlate. Gr00tTrainer.prediction_step
computes it instead.

CPU only: a module with GR00T's forward signature is all the label detection and
the loss path look at.
"""

from __future__ import annotations

from gr00t.experiment.trainer import Gr00tTrainer
import torch
from torch import nn
from transformers import TrainingArguments
from transformers.utils.generic import can_return_loss, find_labels


class _TinyGr00tLike(nn.Module):
    """Same forward contract as Gr00tN1d7: one `inputs` dict in, loss out."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 1)

    def forward(self, inputs: dict):
        return {"loss": self.proj(inputs["state"]).mean()}


def _trainer(tmp_path) -> Gr00tTrainer:
    return Gr00tTrainer(
        model=_TinyGr00tLike(),
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True),
    )


def _batch():
    return {"inputs": {"state": torch.ones(2, 4)}}


def test_forward_signature_hides_the_loss_from_trainer():
    """Why the override exists — both of Trainer's loss detectors come up empty."""
    assert find_labels(_TinyGr00tLike) == []
    assert can_return_loss(_TinyGr00tLike) is False


def test_prediction_step_returns_a_loss(tmp_path):
    trainer = _trainer(tmp_path)
    loss, logits, labels = trainer.prediction_step(
        trainer.model, _batch(), prediction_loss_only=True
    )
    assert loss is not None
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert not loss.requires_grad  # detached, so eval does not hold the graph
    assert logits is None and labels is None  # loss only: compute_metrics is unset


def test_upstream_prediction_step_would_return_no_loss(tmp_path):
    """Pins the upstream behaviour the override replaces."""
    trainer = _trainer(tmp_path)
    assert trainer.label_names == []
    assert trainer.can_return_loss is False
    from transformers import Trainer

    loss, _, _ = Trainer.prediction_step(
        trainer, trainer.model, _batch(), prediction_loss_only=True
    )
    assert loss is None


def test_evaluate_logs_eval_loss(tmp_path):
    """End of the chain: the metric the sweep reads is present."""

    class _EvalSet(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return {"state": torch.ones(4)}

    trainer = Gr00tTrainer(
        model=_TinyGr00tLike(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            report_to=[],
            use_cpu=True,
            per_device_eval_batch_size=2,
            # as TrainingConfig sets it: forward() takes one `inputs` dict, so
            # column pruning against its signature would drop the whole sample
            remove_unused_columns=False,
        ),
        eval_dataset=_EvalSet(),
        data_collator=lambda feats: {"inputs": {"state": torch.stack([f["state"] for f in feats])}},
    )
    metrics = trainer.evaluate()
    assert "eval_loss" in metrics
    assert metrics["eval_loss"] == metrics["eval_loss"]  # not NaN
