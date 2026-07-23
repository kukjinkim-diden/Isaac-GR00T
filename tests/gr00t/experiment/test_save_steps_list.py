# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test SaveStepsListCallback flips should_save on exactly the listed steps."""

from types import SimpleNamespace

from gr00t.experiment.utils import SaveStepsListCallback


def test_saves_only_on_listed_steps():
    cb = SaveStepsListCallback([1000, 2000, 5000, 100000])
    saved = []
    for step in [999, 1000, 1001, 2000, 3000, 5000, 99999, 100000]:
        control = SimpleNamespace(should_save=False)
        cb.on_step_end(args=None, state=SimpleNamespace(global_step=step), control=control)
        if control.should_save:
            saved.append(step)
    assert saved == [1000, 2000, 5000, 100000]
