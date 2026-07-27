from __future__ import annotations

import numpy as np

from toposemiseg.metrics import (
    betti_error,
    object_dice,
    segmentation_metrics,
    variation_of_information,
)


def test_identical_segmentation_metrics_are_perfect() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[2:8, 2:8] = True
    mask[20:28, 20:28] = True
    metrics = segmentation_metrics(mask, mask)
    assert metrics["accuracy"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["dice_obj"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["betti_error"] == 0.0
    assert abs(metrics["voi"]) < 1e-12


def test_component_merge_is_topological_error() -> None:
    target = np.zeros((16, 16), dtype=bool)
    target[2:6, 2:6] = True
    target[2:6, 9:13] = True
    prediction = target.copy()
    prediction[3, 6:10] = True
    assert betti_error(prediction, target) == 1.0
    assert object_dice(prediction, target) < 1.0
    assert variation_of_information(prediction, target) > 0.0
