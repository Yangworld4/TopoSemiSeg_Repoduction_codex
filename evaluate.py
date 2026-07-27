"""Evaluate a TopoSemiSeg checkpoint on a labeled manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from toposemiseg.data import LabeledDataset
from toposemiseg.metrics import segmentation_metrics
from toposemiseg.model import UNetPlusPlus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", help="Defaults to validation_manifest")
    parser.add_argument(
        "--weights",
        choices=("student", "teacher"),
        default="student",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data = config["data"]
    manifest = args.manifest or data.get("validation_manifest")
    if not manifest:
        raise ValueError("Pass --manifest or configure data.validation_manifest")
    dataset = LabeledDataset(
        manifest,
        crop_size=int(data["crop_size"]),
        mean=tuple(data.get("mean", [0.5, 0.5, 0.5])),
        std=tuple(data.get("std", [0.5, 0.5, 0.5])),
        training=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    model_config = config["model"]
    model = UNetPlusPlus(
        in_channels=int(model_config.get("in_channels", 3)),
        num_classes=int(model_config.get("num_classes", 2)),
        base_channels=int(model_config.get("base_channels", 32)),
        deep_supervision=bool(model_config.get("deep_supervision", False)),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint[args.weights])
    model.eval()
    foreground_class = int(model_config.get("foreground_class", 1))
    totals: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"].to(device))
            logits = output[-1] if isinstance(output, list) else output
            prediction = (logits.argmax(dim=1)[0] == foreground_class).cpu().numpy()
            target = batch["mask"][0].numpy() == foreground_class
            for key, value in segmentation_metrics(prediction, target).items():
                totals[key] = totals.get(key, 0.0) + value
    print(
        json.dumps(
            {key: value / len(dataset) for key, value in totals.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
