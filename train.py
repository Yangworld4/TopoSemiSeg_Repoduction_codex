"""Two-stage Mean-Teacher training for TopoSemiSeg."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from topo_consistency_loss import TopologyConsistencyLoss, TopologyLossParts
from toposemiseg.data import LabeledDataset, UnlabeledDataset
from toposemiseg.ema import create_ema_teacher, update_ema
from toposemiseg.losses import (
    gaussian_rampup,
    soft_cross_entropy,
    supervised_segmentation_loss,
)
from toposemiseg.metrics import segmentation_metrics
from toposemiseg.model import UNetPlusPlus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument(
        "--stage",
        choices=("all", "pretrain", "finetune"),
        default="all",
        help="Run both paper stages or only one stage",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint to resume; required for standalone finetuning",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("The config root must be a mapping")
    config["_config_path"] = str(config_path)
    return config


def endless(loader: DataLoader) -> Iterator[dict[str, Tensor | list[str]]]:
    while True:
        yield from loader


def unwrap_logits(output: Tensor | list[Tensor]) -> Tensor:
    return output[-1] if isinstance(output, list) else output


def create_grad_scaler(enabled: bool) -> Any:
    """Create a GradScaler without warnings on both PyTorch 2.0 and 2.6+."""

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_loaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    data = config["data"]
    augmentation = config.get("augmentation", {})
    mean = tuple(data.get("mean", [0.5, 0.5, 0.5]))
    std = tuple(data.get("std", [0.5, 0.5, 0.5]))
    crop_size = int(data["crop_size"])
    labeled = LabeledDataset(
        data["labeled_manifest"], crop_size, mean=mean, std=std, training=True
    )
    unlabeled = UnlabeledDataset(
        data["unlabeled_manifest"],
        crop_size,
        mean=mean,
        std=std,
        brightness=float(augmentation.get("brightness", 0.3)),
        contrast=float(augmentation.get("contrast", 0.1)),
        morphology_probability=float(augmentation.get("morphology_probability", 0.5)),
    )
    loader_options = {
        "num_workers": int(data.get("num_workers", 4)),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": int(data.get("num_workers", 4)) > 0,
    }
    labeled_loader = DataLoader(
        labeled,
        batch_size=int(data["labeled_batch_size"]),
        shuffle=True,
        drop_last=False,
        **loader_options,
    )
    unlabeled_loader = DataLoader(
        unlabeled,
        batch_size=int(data["unlabeled_batch_size"]),
        shuffle=True,
        drop_last=False,
        **loader_options,
    )

    validation_loader = None
    if data.get("validation_manifest"):
        validation = LabeledDataset(
            data["validation_manifest"],
            crop_size,
            mean=mean,
            std=std,
            training=False,
        )
        validation_loader = DataLoader(
            validation,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            **loader_options,
        )
    return labeled_loader, unlabeled_loader, validation_loader


def save_checkpoint(
    path: Path,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: dict[str, Any],
    stage: str,
    global_step: int,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "stage": stage,
            "global_step": global_step,
            "epoch": epoch,
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: str | Path,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    student.load_state_dict(checkpoint["student"])
    teacher.load_state_dict(checkpoint["teacher"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("global_step", 0)), int(checkpoint.get("epoch", 0))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    foreground_class: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].numpy()
        logits = unwrap_logits(model(image))
        prediction = (logits.argmax(dim=1) == foreground_class).cpu().numpy()
        for predicted_image, target_image in zip(prediction, mask):
            metrics = segmentation_metrics(
                predicted_image, target_image == foreground_class
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_step(
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    labeled_batch: dict[str, Tensor | list[str]],
    unlabeled_batch: dict[str, Tensor | list[str]],
    topology_loss: TopologyConsistencyLoss,
    device: torch.device,
    config: dict[str, Any],
    global_step: int,
    total_steps: int,
    use_topology: bool,
) -> dict[str, float]:
    training = config["training"]
    foreground_class = int(config["model"].get("foreground_class", 1))
    image = labeled_batch["image"].to(device, non_blocking=True)
    mask = labeled_batch["mask"].to(device, non_blocking=True)
    weak = unlabeled_batch["weak"].to(device, non_blocking=True)
    strong = unlabeled_batch["strong"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        labeled_logits = unwrap_logits(student(image))
        student_logits = unwrap_logits(student(strong))
        with torch.no_grad():
            teacher_logits = unwrap_logits(teacher(weak))
        supervised = supervised_segmentation_loss(
            labeled_logits,
            mask,
            ce_weight=float(training.get("supervised_ce_weight", 0.5)),
            dice_weight=float(training.get("supervised_dice_weight", 0.5)),
            foreground_class=foreground_class,
        )
        pixel = soft_cross_entropy(student_logits, teacher_logits)
        pixel_weight = gaussian_rampup(
            global_step,
            total_steps,
            maximum=float(training.get("pixel_consistency_max_weight", 0.1)),
        )

    # Do not use ``student_logits.sum() * 0`` here. Under AMP, summing a large
    # FP16 logit map can overflow before multiplication and produce inf * 0 =
    # NaN, even during pretraining when the topology term is disabled.
    zero = student_logits.new_zeros(())
    topology_parts = TopologyLossParts(zero, zero)
    topology_interval = int(training.get("topology_interval", 1))
    if use_topology and global_step % topology_interval == 0:
        # PH runs in float32 even under AMP; only the selected student pixels
        # remain in the autograd graph.
        student_probability = torch.softmax(student_logits.float(), dim=1)[
            :, foreground_class
        ]
        teacher_probability = torch.softmax(teacher_logits.float(), dim=1)[
            :, foreground_class
        ]
        topology_parts = topology_loss(
            student_probability, teacher_probability, return_parts=True
        )
        assert isinstance(topology_parts, TopologyLossParts)

    topology_weight = (
        float(training.get("topology_weight", 0.002)) if use_topology else 0.0
    )
    loss = supervised + pixel_weight * pixel + topology_weight * topology_parts.total
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    update_ema(
        teacher,
        student,
        decay=float(training.get("ema_decay", 0.999)),
    )
    return {
        "loss": float(loss.detach()),
        "supervised": float(supervised.detach()),
        "pixel": float(pixel.detach()),
        "pixel_weight": pixel_weight,
        "topology": float(topology_parts.total.detach()),
        "topology_consistency": float(topology_parts.consistency.detach()),
        "topology_removal": float(topology_parts.removal.detach()),
    }


def log_record(path: Path, record: dict[str, Any]) -> None:
    print(json.dumps(record, ensure_ascii=False))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 1337))
    seed_everything(seed)
    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    labeled_loader, unlabeled_loader, validation_loader = build_loaders(config)
    model_config = config["model"]
    student = UNetPlusPlus(
        in_channels=int(model_config.get("in_channels", 3)),
        num_classes=int(model_config.get("num_classes", 2)),
        base_channels=int(model_config.get("base_channels", 32)),
        deep_supervision=bool(model_config.get("deep_supervision", False)),
    ).to(device)
    teacher = create_ema_teacher(student).to(device)
    training = config["training"]
    optimizer = Adam(
        student.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scaler = create_grad_scaler(
        enabled=bool(training.get("amp", True)) and device.type == "cuda"
    )
    topology_loss = TopologyConsistencyLoss(
        patch_size=int(training.get("topology_patch_size", 100)),
        persistence_threshold=float(training.get("persistence_threshold", 0.7)),
        reduction="mean",
    )

    output_dir = Path(config.get("output_dir", "runs/toposemiseg")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    global_step = 0
    start_epoch = 0
    if args.checkpoint:
        global_step, start_epoch = load_checkpoint(
            args.checkpoint,
            student,
            teacher,
            optimizer,
            scaler,
            device,
        )
    elif args.stage == "finetune":
        raise ValueError("--checkpoint is required when --stage=finetune")

    pretrain_steps = int(training["pretrain_iterations"])
    steps_per_epoch = max(len(labeled_loader), len(unlabeled_loader))
    finetune_epochs = int(training["finetune_epochs"])
    total_steps = pretrain_steps + steps_per_epoch * finetune_epochs
    labeled_iterator = endless(labeled_loader)
    unlabeled_iterator = endless(unlabeled_loader)

    if args.stage in {"all", "pretrain"}:
        student.train()
        for local_step in range(global_step, pretrain_steps):
            metrics = train_step(
                student,
                teacher,
                optimizer,
                scaler,
                next(labeled_iterator),
                next(unlabeled_iterator),
                topology_loss,
                device,
                config,
                global_step=local_step,
                total_steps=total_steps,
                use_topology=False,
            )
            global_step = local_step + 1
            if global_step % int(training.get("log_interval", 20)) == 0:
                log_record(
                    log_path,
                    {"stage": "pretrain", "step": global_step, **metrics},
                )
            if global_step % int(training.get("checkpoint_interval", 1000)) == 0:
                save_checkpoint(
                    output_dir / "last.pt",
                    student,
                    teacher,
                    optimizer,
                    scaler,
                    config,
                    "pretrain",
                    global_step,
                    0,
                )
        save_checkpoint(
            output_dir / "pretrained.pt",
            student,
            teacher,
            optimizer,
            scaler,
            config,
            "pretrain",
            global_step,
            0,
        )

    if args.stage in {"all", "finetune"}:
        student.train()
        for epoch in range(start_epoch, finetune_epochs):
            aggregate: dict[str, float] = {}
            for _ in range(steps_per_epoch):
                metrics = train_step(
                    student,
                    teacher,
                    optimizer,
                    scaler,
                    next(labeled_iterator),
                    next(unlabeled_iterator),
                    topology_loss,
                    device,
                    config,
                    global_step=global_step,
                    total_steps=total_steps,
                    use_topology=True,
                )
                global_step += 1
                for key, value in metrics.items():
                    aggregate[key] = aggregate.get(key, 0.0) + value
            record: dict[str, Any] = {
                "stage": "finetune",
                "epoch": epoch + 1,
                "step": global_step,
                **{key: value / steps_per_epoch for key, value in aggregate.items()},
            }
            validation_interval = int(training.get("validation_interval", 1))
            if validation_loader is not None and (epoch + 1) % validation_interval == 0:
                record.update(
                    {
                        f"val_{key}": value
                        for key, value in validate(
                            student,
                            validation_loader,
                            device,
                            int(model_config.get("foreground_class", 1)),
                        ).items()
                    }
                )
            log_record(log_path, record)
            save_checkpoint(
                output_dir / "last.pt",
                student,
                teacher,
                optimizer,
                scaler,
                config,
                "finetune",
                global_step,
                epoch + 1,
            )


if __name__ == "__main__":
    main()
