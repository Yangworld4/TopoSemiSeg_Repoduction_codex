"""Noise-aware topological consistency loss used by TopoSemiSeg.

The persistent-homology computation and matching are discrete operations.  They
are intentionally performed on detached NumPy arrays.  Once the critical pixels
have been selected, the loss is assembled by indexing the original student
tensor, so gradients still flow to the student network exactly as described in
Eq. (5) and Eq. (7) of the paper.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from torch import Tensor, nn

try:
    import cripser
except ImportError as exc:  # pragma: no cover - exercised only without extras
    cripser = None
    _CRIPSER_IMPORT_ERROR = exc
else:
    _CRIPSER_IMPORT_ERROR = None

try:
    from gudhi.wasserstein import wasserstein_distance
except ImportError as exc:  # pragma: no cover - exercised only without extras
    wasserstein_distance = None
    _GUDHI_IMPORT_ERROR = exc
else:
    _GUDHI_IMPORT_ERROR = None


@dataclass(frozen=True)
class PersistenceDiagram:
    """A zero-dimensional persistence diagram and its critical pixels."""

    points: np.ndarray
    birth_pixels: np.ndarray
    death_pixels: np.ndarray
    signal_indices: np.ndarray
    noise_indices: np.ndarray

    @property
    def has_points(self) -> bool:
        return self.points.shape[0] > 0


@dataclass(frozen=True)
class TopologyLossParts:
    """Separate terms of the loss, useful for logging and ablations."""

    consistency: Tensor
    removal: Tensor

    @property
    def total(self) -> Tensor:
        return self.consistency + self.removal


def _require_topology_dependencies() -> None:
    if cripser is None:
        raise ImportError(
            "TopoSemiSeg requires `cripser` for persistent homology. "
            "Install the dependencies with `pip install -r requirements.txt`."
        ) from _CRIPSER_IMPORT_ERROR
    if wasserstein_distance is None:
        raise ImportError(
            "TopoSemiSeg requires GUDHI and POT for persistence-diagram "
            "matching. Install the dependencies with "
            "`pip install -r requirements.txt`."
        ) from _GUDHI_IMPORT_ERROR


def extract_persistence_diagram(
    foreground_probability: np.ndarray,
    persistence_threshold: float = 0.7,
) -> PersistenceDiagram:
    """Extract the 0-D diagram used in the paper.

    CubicalRipser implements a sublevel filtration.  The model predicts
    foreground probability (large values are foreground), therefore the
    filtration is ``1 - probability``.
    """

    _require_topology_dependencies()
    probability = np.asarray(foreground_probability, dtype=np.float64)
    if probability.ndim != 2:
        raise ValueError(
            f"foreground_probability must be a 2-D array, got shape {probability.shape}"
        )

    raw = np.asarray(cripser.computePH(1.0 - probability, maxdim=1, location="birth"))
    if raw.size == 0:
        raw_0d = np.empty((0, 9), dtype=np.float64)
    else:
        raw_0d = raw[raw[:, 0] == 0]

    points = np.asarray(raw_0d[:, 1:3], dtype=np.float64).copy()
    if points.size:
        # The essential component has infinite death.  As the filtration is
        # bounded to [0, 1], the official implementation clips it to 1.
        points[:, 1] = np.minimum(points[:, 1], 1.0)
    birth_pixels = np.asarray(raw_0d[:, 3:5], dtype=np.int64)
    death_pixels = np.asarray(raw_0d[:, 6:8], dtype=np.int64)
    persistence = np.abs(points[:, 1] - points[:, 0])
    signal = np.flatnonzero(persistence > persistence_threshold)
    noise = np.flatnonzero(persistence <= persistence_threshold)
    return PersistenceDiagram(
        points=points,
        birth_pixels=birth_pixels,
        death_pixels=death_pixels,
        signal_indices=signal,
        noise_indices=noise,
    )


def match_signal_diagrams(
    student_points: np.ndarray,
    teacher_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return student-to-diagonal indices and student/teacher matched pairs."""

    _require_topology_dependencies()
    student_points = np.asarray(student_points, dtype=np.float64).reshape(-1, 2)
    teacher_points = np.asarray(teacher_points, dtype=np.float64).reshape(-1, 2)

    if student_points.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.int64)
    if teacher_points.shape[0] == 0:
        return (
            np.arange(student_points.shape[0], dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
        )

    _, matching = wasserstein_distance(
        student_points, teacher_points, matching=True, order=2
    )
    matching = np.asarray(matching, dtype=np.int64).reshape(-1, 2)
    to_diagonal = matching[(matching[:, 0] >= 0) & (matching[:, 1] < 0), 0]
    paired = matching[(matching[:, 0] >= 0) & (matching[:, 1] >= 0)]
    return to_diagonal, paired


def _valid_pixel(pixel: Iterable[int], height: int, width: int) -> bool:
    row, col = (int(value) for value in pixel)
    return 0 <= row < height and 0 <= col < width


def _at(image: Tensor, pixel: Iterable[int]) -> Tensor:
    row, col = (int(value) for value in pixel)
    return image[row, col]


def _endpoint_loss(
    student_patch: Tensor,
    teacher_patch: Tensor,
    student_pixel: np.ndarray,
    teacher_pixel: np.ndarray,
) -> Tensor:
    height, width = student_patch.shape
    if not _valid_pixel(student_pixel, height, width):
        return student_patch.sum() * 0.0
    if not _valid_pixel(teacher_pixel, height, width):
        return student_patch.sum() * 0.0
    return (_at(student_patch, student_pixel) - _at(teacher_patch, teacher_pixel)) ** 2


def _patch_topology_loss(
    student_patch: Tensor,
    teacher_patch: Tensor,
    persistence_threshold: float,
) -> TopologyLossParts:
    zero = student_patch.sum() * 0.0
    student_np = student_patch.detach().float().cpu().numpy()
    teacher_np = teacher_patch.detach().float().cpu().numpy()

    # Constant patches carry no useful finite topology and can otherwise add a
    # boundary-dependent essential component.
    if np.ptp(student_np) <= 1e-7:
        return TopologyLossParts(zero, zero)

    student = extract_persistence_diagram(student_np, persistence_threshold)
    teacher = extract_persistence_diagram(teacher_np, persistence_threshold)
    if not student.has_points:
        return TopologyLossParts(zero, zero)

    student_signal = student.points[student.signal_indices]
    teacher_signal = teacher.points[teacher.signal_indices]
    to_diagonal, paired = match_signal_diagrams(student_signal, teacher_signal)

    consistency = zero
    for student_local, teacher_local in paired:
        student_index = int(student.signal_indices[student_local])
        teacher_index = int(teacher.signal_indices[teacher_local])
        consistency = consistency + _endpoint_loss(
            student_patch,
            teacher_patch,
            student.birth_pixels[student_index],
            teacher.birth_pixels[teacher_index],
        )
        consistency = consistency + _endpoint_loss(
            student_patch,
            teacher_patch,
            student.death_pixels[student_index],
            teacher.death_pixels[teacher_index],
        )

    # An unmatched student signal is matched to its orthogonal projection on
    # the diagonal.  In probability coordinates this has the same midpoint.
    height, width = student_patch.shape
    for student_local in to_diagonal:
        student_index = int(student.signal_indices[student_local])
        birth_pixel = student.birth_pixels[student_index]
        death_pixel = student.death_pixels[student_index]
        if _valid_pixel(birth_pixel, height, width) and _valid_pixel(
            death_pixel, height, width
        ):
            birth = _at(student_patch, birth_pixel)
            death = _at(student_patch, death_pixel)
            midpoint = (birth + death).detach() / 2.0
            consistency = consistency + (birth - midpoint) ** 2
            consistency = consistency + (death - midpoint) ** 2

    # Eq. (7): second-order total persistence of all student noise dots.
    removal = zero
    for student_index in student.noise_indices:
        birth_pixel = student.birth_pixels[int(student_index)]
        death_pixel = student.death_pixels[int(student_index)]
        if _valid_pixel(birth_pixel, height, width) and _valid_pixel(
            death_pixel, height, width
        ):
            removal = (
                removal
                + (_at(student_patch, birth_pixel) - _at(student_patch, death_pixel))
                ** 2
            )

    return TopologyLossParts(consistency=consistency, removal=removal)


def topology_consistency_loss(
    student_probability: Tensor,
    teacher_probability: Tensor,
    patch_size: int = 100,
    persistence_threshold: float = 0.7,
    reduction: str = "mean",
    return_parts: bool = False,
) -> Tensor | TopologyLossParts:
    """Compute the paper's noise-aware topological consistency loss.

    Args:
        student_probability: Foreground probabilities with shape ``[H, W]``,
            ``[B, H, W]``, or ``[B, 1, H, W]``.
        teacher_probability: Same shape as ``student_probability``.  It is
            detached internally; gradients only update the student.
        patch_size: Non-overlapping PH window size.  ``0`` uses the full image.
        persistence_threshold: Signal/noise split ``phi`` from the paper.
        reduction: ``"mean"`` (paper's batch averaging) or ``"sum"``.
        return_parts: Return consistency/removal terms separately for logging.
    """

    if student_probability.shape != teacher_probability.shape:
        raise ValueError(
            "student and teacher probability maps must have the same shape, "
            f"got {student_probability.shape} and {teacher_probability.shape}"
        )
    if not student_probability.is_floating_point():
        raise TypeError("student_probability must be a floating-point tensor")
    if patch_size < 0:
        raise ValueError("patch_size must be non-negative")
    if not 0.0 <= persistence_threshold <= 1.0:
        raise ValueError("persistence_threshold must be in [0, 1]")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    student = student_probability
    teacher = teacher_probability.detach().to(
        device=student.device, dtype=student.dtype
    )
    if student.ndim == 2:
        student = student.unsqueeze(0)
        teacher = teacher.unsqueeze(0)
    elif student.ndim == 4 and student.shape[1] == 1:
        student = student[:, 0]
        teacher = teacher[:, 0]
    elif student.ndim != 3:
        raise ValueError(
            f"expected [H,W], [B,H,W], or [B,1,H,W], got {student_probability.shape}"
        )

    consistency = student.sum() * 0.0
    removal = student.sum() * 0.0
    for student_image, teacher_image in zip(student, teacher):
        height, width = student_image.shape
        window = patch_size or max(height, width)
        for top in range(0, height, window):
            for left in range(0, width, window):
                parts = _patch_topology_loss(
                    student_image[top : top + window, left : left + window],
                    teacher_image[top : top + window, left : left + window],
                    persistence_threshold,
                )
                consistency = consistency + parts.consistency
                removal = removal + parts.removal

    if reduction == "mean":
        consistency = consistency / student.shape[0]
        removal = removal / student.shape[0]
    parts = TopologyLossParts(consistency=consistency, removal=removal)
    return parts if return_parts else parts.total


class TopologyConsistencyLoss(nn.Module):
    """``nn.Module`` wrapper around :func:`topology_consistency_loss`."""

    def __init__(
        self,
        patch_size: int = 100,
        persistence_threshold: float = 0.7,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.persistence_threshold = persistence_threshold
        self.reduction = reduction

    def forward(
        self,
        student_probability: Tensor,
        teacher_probability: Tensor,
        return_parts: bool = False,
    ) -> Tensor | TopologyLossParts:
        return topology_consistency_loss(
            student_probability,
            teacher_probability,
            patch_size=self.patch_size,
            persistence_threshold=self.persistence_threshold,
            reduction=self.reduction,
            return_parts=return_parts,
        )


# Backward-compatible names used in the repository's original README.
def getTopoLoss(
    stu_tensor: Tensor,
    tea_tensor: Tensor,
    topo_size: int = 100,
    pd_threshold: float = 0.7,
    loss_mode: str = "mse",
) -> Tensor:
    if loss_mode != "mse":
        raise ValueError("The paper defines only the squared (MSE) topology loss")
    return topology_consistency_loss(
        stu_tensor,
        tea_tensor,
        patch_size=topo_size,
        persistence_threshold=pd_threshold,
        reduction="sum",
    )


def calculate_topo_loss(
    likelihood: Tensor,
    target: Tensor,
    topo_size: int = 100,
    pd_threshold: float = 0.7,
) -> Tensor:
    return topology_consistency_loss(
        likelihood,
        target,
        patch_size=topo_size,
        persistence_threshold=pd_threshold,
        reduction="mean",
    )
