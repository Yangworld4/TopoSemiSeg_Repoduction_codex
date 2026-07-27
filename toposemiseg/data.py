"""Manifest-based pathology datasets and paper-compatible augmentations."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path | None = None


def read_manifest(path: str | Path, require_mask: bool) -> list[Sample]:
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    samples: list[Sample] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image" not in reader.fieldnames:
            raise ValueError(f"{manifest} must contain an 'image' column")
        if require_mask and "mask" not in reader.fieldnames:
            raise ValueError(f"{manifest} must contain a 'mask' column")
        for row in reader:
            image = Path(row["image"])
            image = image if image.is_absolute() else manifest.parent / image
            mask_value = row.get("mask", "")
            mask = Path(mask_value) if mask_value else None
            if mask is not None and not mask.is_absolute():
                mask = manifest.parent / mask
            samples.append(Sample(image=image, mask=mask))
    if not samples:
        raise ValueError(f"Manifest is empty: {manifest}")
    return samples


def _pad_to_crop(image: Image.Image, crop_size: int, fill: int = 0) -> Image.Image:
    width, height = image.size
    pad_width = max(crop_size - width, 0)
    pad_height = max(crop_size - height, 0)
    if pad_width or pad_height:
        image = ImageOps.expand(
            image,
            border=(
                pad_width // 2,
                pad_height // 2,
                pad_width - pad_width // 2,
                pad_height - pad_height // 2,
            ),
            fill=fill,
        )
    return image


def _shared_geometric_transform(
    image: Image.Image,
    mask: Image.Image | None,
    crop_size: int,
) -> tuple[Image.Image, Image.Image | None]:
    image = _pad_to_crop(image, crop_size)
    if mask is not None:
        mask = _pad_to_crop(mask, crop_size)
    width, height = image.size
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    box = (left, top, left + crop_size, top + crop_size)
    image = image.crop(box)
    if mask is not None:
        mask = mask.crop(box)

    rotation = random.choice((0, 90, 180, 270))
    if rotation:
        image = image.rotate(rotation)
        if mask is not None:
            mask = mask.rotate(rotation)
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
        if mask is not None:
            mask = ImageOps.mirror(mask)
    if random.random() < 0.5:
        image = ImageOps.flip(image)
        if mask is not None:
            mask = ImageOps.flip(mask)
    return image, mask


def _strong_photometric_transform(
    image: Image.Image,
    brightness: float,
    contrast: float,
    morphology_probability: float,
) -> Image.Image:
    brightness_factor = random.uniform(1.0 - brightness, 1.0 + brightness)
    contrast_factor = random.uniform(1.0 - contrast, 1.0 + contrast)
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    # The supplement calls this "morphological shift" without an algorithmic
    # definition.  A random min/max filter gives a one-pixel stain-morphology
    # perturbation while preserving teacher/student spatial correspondence.
    if random.random() < morphology_probability:
        image = image.filter(
            random.choice((ImageFilter.MinFilter(3), ImageFilter.MaxFilter(3)))
        )
    return image


def _image_to_tensor(
    image: Image.Image,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_tensor = tensor.new_tensor(mean)[:, None, None]
    std_tensor = tensor.new_tensor(std)[:, None, None]
    return (tensor - mean_tensor) / std_tensor


def _mask_to_tensor(mask: Image.Image) -> Tensor:
    array = np.asarray(mask, dtype=np.uint8).copy()
    # Supports binary masks encoded as 0/1 or 0/255.
    return torch.from_numpy((array > 0).astype(np.int64))


class LabeledDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        manifest: str | Path,
        crop_size: int,
        mean: tuple[float, ...] = (0.5, 0.5, 0.5),
        std: tuple[float, ...] = (0.5, 0.5, 0.5),
        training: bool = True,
    ) -> None:
        self.samples = read_manifest(manifest, require_mask=True)
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        assert sample.mask is not None
        with Image.open(sample.mask) as source:
            mask = source.convert("L")
        if self.training:
            image, transformed_mask = _shared_geometric_transform(
                image, mask, self.crop_size
            )
            assert transformed_mask is not None
            mask = transformed_mask
        return {
            "image": _image_to_tensor(image, self.mean, self.std),
            "mask": _mask_to_tensor(mask),
            "name": sample.image.stem,
        }


class UnlabeledDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        manifest: str | Path,
        crop_size: int,
        mean: tuple[float, ...] = (0.5, 0.5, 0.5),
        std: tuple[float, ...] = (0.5, 0.5, 0.5),
        brightness: float = 0.3,
        contrast: float = 0.1,
        morphology_probability: float = 0.5,
    ) -> None:
        self.samples = read_manifest(manifest, require_mask=False)
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.brightness = brightness
        self.contrast = contrast
        self.morphology_probability = morphology_probability

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        weak, _ = _shared_geometric_transform(image, None, self.crop_size)
        strong = _strong_photometric_transform(
            weak.copy(),
            brightness=self.brightness,
            contrast=self.contrast,
            morphology_probability=self.morphology_probability,
        )
        return {
            "weak": _image_to_tensor(weak, self.mean, self.std),
            "strong": _image_to_tensor(strong, self.mean, self.std),
            "name": sample.image.stem,
        }
