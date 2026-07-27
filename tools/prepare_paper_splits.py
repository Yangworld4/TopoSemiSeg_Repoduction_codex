"""Create deterministic CRAG/GLaS manifests following the paper protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

PROTOCOLS = {
    "CRAG": {
        "validation": 20,
        "labeled_10": 16,
        "labeled_20": 31,
        "stratify": False,
    },
    "GLaS": {
        "validation": 17,
        "labeled_10": 7,
        "labeled_20": 14,
        "stratify": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def read_records(dataset_root: Path) -> list[dict[str, str]]:
    mapping = dataset_root / "filename_mapping.csv"
    with mapping.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def proportional_allocation(
    groups: dict[str, list[dict[str, str]]],
    count: int,
) -> dict[str, int]:
    total = sum(len(records) for records in groups.values())
    exact = {key: len(records) * count / total for key, records in groups.items()}
    allocation = {key: math.floor(value) for key, value in exact.items()}
    remaining = count - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda key: (exact[key] - allocation[key], len(groups[key]), key),
        reverse=True,
    )
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


def select_records(
    records: list[dict[str, str]],
    count: int,
    seed: int,
    stratify: bool,
) -> list[dict[str, str]]:
    if count > len(records):
        raise ValueError(f"Cannot select {count} records from {len(records)}")
    generator = random.Random(seed)
    if not stratify:
        candidates = sorted(records, key=lambda record: record["standard_name"])
        generator.shuffle(candidates)
        return candidates[:count]

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        groups[record["grade"]].append(record)
    allocation = proportional_allocation(groups, count)
    selected: list[dict[str, str]] = []
    for grade in sorted(groups):
        candidates = sorted(groups[grade], key=lambda record: record["standard_name"])
        generator.shuffle(candidates)
        selected.extend(candidates[: allocation[grade]])
    generator.shuffle(selected)
    return selected


def relative_paths(
    record: dict[str, str],
) -> tuple[str, str, str]:
    split = record["split"]
    name = record["standard_name"]
    return (
        f"../../images/{split}/{name}.png",
        f"../../masks/{split}/{name}.png",
        f"../../instance_masks/{split}/{name}.png",
    )


def verify_record_files(
    dataset_root: Path,
    records: list[dict[str, str]],
) -> None:
    for record in records:
        split = record["split"]
        name = record["standard_name"]
        for directory in ("images", "masks", "instance_masks"):
            path = dataset_root / directory / split / f"{name}.png"
            if not path.is_file():
                raise FileNotFoundError(path)


def write_labeled_manifest(
    path: Path,
    records: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "mask", "instance_mask", "grade"])
        for record in sorted(records, key=lambda item: item["standard_name"]):
            image, mask, instance_mask = relative_paths(record)
            writer.writerow([image, mask, instance_mask, record["grade"]])


def write_unlabeled_manifest(
    path: Path,
    records: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image"])
        for record in sorted(records, key=lambda item: item["standard_name"]):
            image, _, _ = relative_paths(record)
            writer.writerow([image])


def names(records: list[dict[str, str]]) -> list[str]:
    return sorted(record["standard_name"] for record in records)


def grade_counts(records: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record["grade"]] += 1
    return dict(sorted(counts.items()))


def prepare_dataset(dataset_root: Path, seed: int) -> dict[str, Any]:
    dataset = dataset_root.name
    protocol = PROTOCOLS[dataset]
    records = read_records(dataset_root)
    verify_record_files(dataset_root, records)
    official_train = [record for record in records if record["split"] == "train"]
    official_test = [record for record in records if record["split"] == "test"]

    validation = select_records(
        official_train,
        protocol["validation"],
        seed=seed,
        stratify=protocol["stratify"],
    )
    validation_names = set(names(validation))
    train_pool = [
        record
        for record in official_train
        if record["standard_name"] not in validation_names
    ]
    labeled_20 = select_records(
        train_pool,
        protocol["labeled_20"],
        seed=seed + 1,
        stratify=protocol["stratify"],
    )
    labeled_10 = select_records(
        labeled_20,
        protocol["labeled_10"],
        seed=seed + 2,
        stratify=protocol["stratify"],
    )
    labeled_20_names = set(names(labeled_20))
    labeled_10_names = set(names(labeled_10))
    unlabeled_20 = [
        record
        for record in train_pool
        if record["standard_name"] not in labeled_20_names
    ]
    unlabeled_10 = [
        record
        for record in train_pool
        if record["standard_name"] not in labeled_10_names
    ]

    output = dataset_root / "manifests" / f"paper_seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    write_labeled_manifest(output / "labeled_10.csv", labeled_10)
    write_unlabeled_manifest(output / "unlabeled_10.csv", unlabeled_10)
    write_labeled_manifest(output / "labeled_20.csv", labeled_20)
    write_unlabeled_manifest(output / "unlabeled_20.csv", unlabeled_20)
    write_labeled_manifest(output / "validation.csv", validation)
    write_labeled_manifest(output / "test.csv", official_test)
    write_labeled_manifest(output / "train_pool.csv", train_pool)

    summary: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "protocol": {
            "official_train": len(official_train),
            "train_pool": len(train_pool),
            "validation": len(validation),
            "official_test": len(official_test),
            "labeled_10": len(labeled_10),
            "unlabeled_10": len(unlabeled_10),
            "labeled_20": len(labeled_20),
            "unlabeled_20": len(unlabeled_20),
            "stratified_by_grade": protocol["stratify"],
            "nested_labeled_sets": labeled_10_names <= labeled_20_names,
        },
        "grade_counts": {
            "train_pool": grade_counts(train_pool),
            "validation": grade_counts(validation),
            "labeled_10": grade_counts(labeled_10),
            "labeled_20": grade_counts(labeled_20),
            "test": grade_counts(official_test),
        },
        "members": {
            "validation": names(validation),
            "labeled_10": names(labeled_10),
            "labeled_20": names(labeled_20),
        },
    }
    with (output / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    summaries = [
        prepare_dataset(data_root / dataset, args.seed) for dataset in PROTOCOLS
    ]
    for summary in summaries:
        protocol = summary["protocol"]
        print(
            f"{summary['dataset']}: train={protocol['train_pool']}, "
            f"val={protocol['validation']}, test={protocol['official_test']}, "
            f"labeled(10/20)={protocol['labeled_10']}/"
            f"{protocol['labeled_20']}"
        )


if __name__ == "__main__":
    main()
