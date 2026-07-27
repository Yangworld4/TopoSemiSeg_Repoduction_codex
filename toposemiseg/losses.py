"""Pixel-wise losses from Eq. (1)--(3) of TopoSemiSeg."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def dice_loss(
    logits: Tensor,
    target: Tensor,
    foreground_class: int = 1,
    smooth: float = 1e-5,
) -> Tensor:
    probability = torch.softmax(logits, dim=1)[:, foreground_class]
    foreground = (target == foreground_class).to(probability.dtype)
    dimensions = tuple(range(1, probability.ndim))
    intersection = (probability * foreground).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + foreground.sum(dim=dimensions)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def supervised_segmentation_loss(
    logits: Tensor,
    target: Tensor,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    foreground_class: int = 1,
) -> Tensor:
    return ce_weight * F.cross_entropy(logits, target.long()) + dice_weight * dice_loss(
        logits, target, foreground_class=foreground_class
    )


def soft_cross_entropy(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    """CE between student output and detached teacher probability (Eq. 3)."""

    teacher_probability = torch.softmax(teacher_logits.detach(), dim=1)
    return (
        -(teacher_probability * torch.log_softmax(student_logits, dim=1))
        .sum(dim=1)
        .mean()
    )


def gaussian_rampup(step: int, total_steps: int, maximum: float = 0.1) -> float:
    """Paper's ``k * exp(-5 * (1 - tau/T)^2)`` schedule."""

    if total_steps <= 0:
        return maximum
    phase = 1.0 - min(max(step / total_steps, 0.0), 1.0)
    return float(maximum * math.exp(-5.0 * phase * phase))
