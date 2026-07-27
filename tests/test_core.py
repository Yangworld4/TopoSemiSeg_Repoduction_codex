from __future__ import annotations

import torch

from toposemiseg.ema import create_ema_teacher, update_ema
from toposemiseg.losses import gaussian_rampup, supervised_segmentation_loss
from toposemiseg.model import UNetPlusPlus


def test_unetplusplus_preserves_odd_spatial_shape() -> None:
    model = UNetPlusPlus(base_channels=4)
    inputs = torch.randn(2, 3, 65, 71)
    output = model(inputs)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 2, 65, 71)


def test_supervised_loss_backpropagates() -> None:
    logits = torch.randn(2, 2, 32, 32, requires_grad=True)
    target = torch.randint(0, 2, (2, 32, 32))
    loss = supervised_segmentation_loss(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_ema_update() -> None:
    student = torch.nn.Conv2d(1, 1, 1, bias=False)
    with torch.no_grad():
        student.weight.fill_(2.0)
    teacher = create_ema_teacher(student)
    with torch.no_grad():
        student.weight.fill_(4.0)
    update_ema(teacher, student, decay=0.5)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 3.0))
    assert not teacher.weight.requires_grad


def test_gaussian_rampup_endpoints() -> None:
    assert gaussian_rampup(0, 100, 0.1) < 0.001
    assert gaussian_rampup(100, 100, 0.1) == 0.1
