# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The eval dataloader must not let accelerate slice pixel_values.

Only the eval loader is wrapped by accelerator.prepare() (Gr00tTrainer overrides
get_train_dataloader with a raw DataLoader), and prepare() defaults
dispatch_batches to True for an IterableDataset — which ShardedMixtureDataset is.
DataLoaderDispatcher then takes the batch size from the first tensor it finds and
slices every tensor to it along dim 0. Qwen VL's pixel_values is
[total_patches, D] with no batch axis, so it lost all but eval_batch_size rows
while image_grid_thw still declared every patch, and the first eval step died
with `size of tensor a (2) must match the size of tensor b (512)` — 1000 steps
into a 100k-step run.

No GPU, model or dataset: the batch structure Gr00tN1d7DataCollator emits is
enough to exercise the wrapping, and building the real collator would download
the gated backbone just to reach a processor these tests never call.
"""

from __future__ import annotations

from accelerate import Accelerator, DataLoaderConfiguration
from gr00t.experiment.experiment import EVAL_ACCELERATOR_CONFIG
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import TrainingArguments
from transformers.feature_extraction_utils import BatchFeature


PATCHES_PER_SAMPLE = 256
EVAL_BATCH_SIZE = 2


class _ShardedLike(IterableDataset):
    """Stands in for ShardedMixtureDataset: iterable, so accelerate dispatches."""

    def __iter__(self):
        while True:
            yield {"state": torch.zeros(7)}


def _collate(features):
    """The shape contract of Gr00tN1d7DataCollator: everything batch-major except
    pixel_values, which the Qwen processor concatenates over the batch's images."""
    n = len(features)
    return BatchFeature(
        data={
            "inputs": {
                "state": torch.stack([f["state"] for f in features]),
                "input_ids": torch.zeros(n, 12, dtype=torch.long),
                "pixel_values": torch.zeros(n * PATCHES_PER_SAMPLE, 1176),
                "image_grid_thw": torch.tensor([[1, 16, 16]] * n),
            }
        }
    )


def _first_batch(dispatch_batches):
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(dispatch_batches=dispatch_batches)
    )
    loader = accelerator.prepare(
        DataLoader(_ShardedLike(), batch_size=EVAL_BATCH_SIZE, collate_fn=_collate)
    )
    return type(loader).__name__, next(iter(loader))["inputs"]


def _declared_patches(batch):
    return int(batch["image_grid_thw"].prod(-1).sum())


def test_config_disables_dispatch_batches():
    """What run() passes to TrainingArguments must survive into the accelerator."""
    args = TrainingArguments(output_dir="/tmp", accelerator_config=EVAL_ACCELERATOR_CONFIG)
    assert args.accelerator_config.dispatch_batches is False


def test_pixel_values_survive_the_eval_loader():
    loader_type, batch = _first_batch(EVAL_ACCELERATOR_CONFIG["dispatch_batches"])
    assert loader_type == "DataLoaderShard"
    # every patch still there, and consistent with what grid_thw declares
    assert batch["pixel_values"].shape[0] == EVAL_BATCH_SIZE * PATCHES_PER_SAMPLE
    assert batch["pixel_values"].shape[0] == _declared_patches(batch)
    # the batch-major tensors are untouched by the setting
    assert batch["state"].shape[0] == EVAL_BATCH_SIZE
    assert batch["input_ids"].shape[0] == EVAL_BATCH_SIZE


def test_the_default_would_truncate_pixel_values():
    """Pins the failure the setting exists for: without it accelerate slices
    pixel_values down to the batch size and the vision tower cannot add
    pos_embeds to it."""
    loader_type, batch = _first_batch(None)
    assert loader_type == "DataLoaderDispatcher"
    assert batch["pixel_values"].shape[0] == EVAL_BATCH_SIZE
    assert _declared_patches(batch) == EVAL_BATCH_SIZE * PATCHES_PER_SAMPLE
    assert batch["pixel_values"].shape[0] != _declared_patches(batch)
