from __future__ import annotations

import numpy as np
import pytest
import torch

import topo_consistency_loss as topology


def diagram(
    points: list[list[float]],
    births: list[list[int]],
    deaths: list[list[int]],
    signal: list[int],
    noise: list[int],
) -> topology.PersistenceDiagram:
    return topology.PersistenceDiagram(
        points=np.asarray(points, dtype=np.float64).reshape(-1, 2),
        birth_pixels=np.asarray(births, dtype=np.int64).reshape(-1, 2),
        death_pixels=np.asarray(deaths, dtype=np.int64).reshape(-1, 2),
        signal_indices=np.asarray(signal, dtype=np.int64),
        noise_indices=np.asarray(noise, dtype=np.int64),
    )


def test_signal_and_noise_terms_keep_student_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    student_diagram = diagram(
        [[0.1, 0.9], [0.4, 0.5]],
        [[0, 0], [0, 1]],
        [[1, 0], [1, 1]],
        [0],
        [1],
    )
    teacher_diagram = diagram(
        [[0.2, 0.8]],
        [[0, 0]],
        [[1, 0]],
        [0],
        [],
    )
    calls = iter((student_diagram, teacher_diagram))
    monkeypatch.setattr(
        topology,
        "extract_persistence_diagram",
        lambda *_args, **_kwargs: next(calls),
    )
    monkeypatch.setattr(
        topology,
        "match_signal_diagrams",
        lambda *_args, **_kwargs: (
            np.empty(0, dtype=np.int64),
            np.asarray([[0, 0]], dtype=np.int64),
        ),
    )

    student = torch.tensor([[0.9, 0.6], [0.1, 0.5]], requires_grad=True)
    teacher = torch.tensor([[0.8, 0.5], [0.2, 0.5]])
    parts = topology.topology_consistency_loss(
        student,
        teacher,
        patch_size=2,
        return_parts=True,
    )
    assert isinstance(parts, topology.TopologyLossParts)
    assert torch.allclose(parts.consistency, torch.tensor(0.02))
    assert torch.allclose(parts.removal, torch.tensor(0.01))
    parts.total.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_shape_validation() -> None:
    student = torch.zeros(1, 8, 8)
    teacher = torch.zeros(2, 8, 8)
    with pytest.raises(ValueError, match="same shape"):
        topology.topology_consistency_loss(student, teacher)
