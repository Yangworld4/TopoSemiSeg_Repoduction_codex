"""Pixel-, object-, and topology-aware segmentation metrics."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    denominator = first.sum() + second.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(first, second).sum() / denominator)


def object_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    """Symmetric area-weighted object Dice used by gland benchmarks."""

    prediction_labels, prediction_count = ndimage.label(prediction)
    target_labels, target_count = ndimage.label(target)
    if prediction_count == 0 or target_count == 0:
        return 1.0 if prediction_count == target_count else 0.0

    def directed(
        source_labels: np.ndarray,
        source_count: int,
        destination_labels: np.ndarray,
        destination_count: int,
    ) -> float:
        total_area = np.count_nonzero(source_labels)
        score = 0.0
        for source_index in range(1, source_count + 1):
            source = source_labels == source_index
            overlap_ids, overlap_counts = np.unique(
                destination_labels[source], return_counts=True
            )
            valid = overlap_ids > 0
            if np.any(valid):
                destination_index = int(
                    overlap_ids[valid][np.argmax(overlap_counts[valid])]
                )
                destination = destination_labels == destination_index
                matched_dice = _dice(source, destination)
            else:
                matched_dice = 0.0
            score += (source.sum() / total_area) * matched_dice
        return score

    forward = directed(target_labels, target_count, prediction_labels, prediction_count)
    backward = directed(
        prediction_labels, prediction_count, target_labels, target_count
    )
    return float((forward + backward) / 2.0)


def betti_error(
    prediction: np.ndarray,
    target: np.ndarray,
    window_size: int = 256,
    stride: int | None = None,
) -> float:
    """Mean absolute 0-D Betti discrepancy over sliding windows."""

    stride = stride or window_size
    height, width = prediction.shape
    errors: list[float] = []
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            predicted_patch = prediction[
                top : min(top + window_size, height),
                left : min(left + window_size, width),
            ]
            target_patch = target[
                top : min(top + window_size, height),
                left : min(left + window_size, width),
            ]
            predicted_components = ndimage.label(predicted_patch)[1]
            target_components = ndimage.label(target_patch)[1]
            errors.append(float(abs(predicted_components - target_components)))
    return float(np.mean(errors)) if errors else 0.0


def variation_of_information(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """Variation of information between connected-component labelings."""

    predicted_labels = ndimage.label(prediction)[0].ravel()
    target_labels = ndimage.label(target)[0].ravel()
    predicted_count = int(predicted_labels.max()) + 1
    target_count = int(target_labels.max()) + 1
    contingency = np.zeros((predicted_count, target_count), dtype=np.float64)
    np.add.at(contingency, (predicted_labels, target_labels), 1.0)
    contingency /= contingency.sum()
    predicted_probability = contingency.sum(axis=1)
    target_probability = contingency.sum(axis=0)

    def entropy(probability: np.ndarray) -> float:
        positive = probability > 0
        return float(-(probability[positive] * np.log2(probability[positive])).sum())

    expected = predicted_probability[:, None] * target_probability[None, :]
    positive = contingency > 0
    mutual_information = float(
        (
            contingency[positive] * np.log2(contingency[positive] / expected[positive])
        ).sum()
    )
    return (
        entropy(predicted_probability)
        + entropy(target_probability)
        - 2.0 * mutual_information
    )


def segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape}, {target.shape}"
        )
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return {
        "accuracy": float((prediction == target).mean()),
        "dice": _dice(prediction, target),
        "dice_obj": object_dice(prediction, target),
        "iou": float(intersection / union) if union else 1.0,
        "betti_error": betti_error(prediction, target),
        "voi": variation_of_information(prediction, target),
    }
