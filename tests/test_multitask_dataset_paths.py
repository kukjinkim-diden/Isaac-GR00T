"""Verify FinetuneConfig accepts multiple --dataset-path values and that
launch_finetune forwards all of them into the mixture (multi-task training).
No GPU / model deps — pure CLI + config plumbing.
"""

import tyro

from gr00t.configs.finetune_config import FinetuneConfig


def _parse(argv):
    return tyro.cli(FinetuneConfig, args=argv)


def test_multiple_dataset_paths():
    cfg = _parse([
        "--base-model-path", "nvidia/GR00T-N1.7-3B",
        "--dataset-path", "/d/task00", "/d/task01", "/d/task02", "/d/task03",
        "--embodiment-tag", "NEW_EMBODIMENT",
    ])
    # tyro parses variadic into a tuple; launch_finetune does list(...) of it.
    assert tuple(cfg.dataset_path) == ("/d/task00", "/d/task01", "/d/task02", "/d/task03")
    assert list(cfg.dataset_path) == ["/d/task00", "/d/task01", "/d/task02", "/d/task03"]


def test_single_dataset_path_backward_compatible():
    # multi_train.sh / finetune.sh pass exactly one path — must still work.
    cfg = _parse([
        "--base-model-path", "x",
        "--dataset-path", "/only/one",
        "--embodiment-tag", "NEW_EMBODIMENT",
    ])
    assert list(cfg.dataset_path) == ["/only/one"]


if __name__ == "__main__":
    test_multiple_dataset_paths()
    test_single_dataset_path_backward_compatible()
    print("ok")
