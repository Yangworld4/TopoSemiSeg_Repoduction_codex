"""Exponential-moving-average teacher utilities."""

from __future__ import annotations

import copy

import torch
from torch import nn


def create_ema_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema(
    teacher: nn.Module,
    student: nn.Module,
    decay: float = 0.999,
) -> None:
    if not 0.0 <= decay <= 1.0:
        raise ValueError("EMA decay must be in [0, 1]")
    student_parameters = dict(student.named_parameters())
    for name, teacher_parameter in teacher.named_parameters():
        teacher_parameter.mul_(decay).add_(
            student_parameters[name].detach(), alpha=1.0 - decay
        )

    # Integer buffers such as num_batches_tracked cannot be averaged.
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        source = student_buffers[name].detach()
        if teacher_buffer.is_floating_point():
            teacher_buffer.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            teacher_buffer.copy_(source)
