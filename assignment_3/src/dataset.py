"""Image pipeline: transforms, torch Dataset and backbone input geometry.

CSV loading lives in :mod:`catalog`; this module is the part that needs torch.
"""
from __future__ import annotations

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config

OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


def infer_image_size(model, preprocess) -> int:
    """Square input resolution a backbone expects."""
    size = getattr(getattr(model, "visual", None), "image_size", None)
    if isinstance(size, (list, tuple)) and size:
        return int(size[0])
    if isinstance(size, int):
        return size
    # Fall back to whatever the returned preprocessing pipeline resizes to.
    for t in getattr(preprocess, "transforms", []):
        cand = getattr(t, "size", None)
        if isinstance(cand, int):
            return cand
        if isinstance(cand, (list, tuple)) and cand and isinstance(cand[0], int):
            return int(cand[0])
    return 224


def extract_norm(preprocess) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Channel statistics used by a backbone's preprocessing pipeline."""
    for t in getattr(preprocess, "transforms", []):
        if isinstance(t, transforms.Normalize):
            mean = t.mean if isinstance(t.mean, (list, tuple)) else (t.mean,) * 3
            std = t.std if isinstance(t.std, (list, tuple)) else (t.std,) * 3
            return tuple(float(x) for x in mean), tuple(float(x) for x in std)
    return OPENAI_MEAN, OPENAI_STD


def build_eval_views(size: int, mean, std, views=("center", "full")) -> dict[str, transforms.Compose]:
    """Deterministic test-time-augmentation views.

    ``center``: resize shortest edge to ``size`` + centre crop (stock CLIP).
    ``full``:   resize the whole photo to ``size x size`` - keeps the plate, which
                a centre crop often cuts off.
    ``zoom``:   resize shortest edge to ``1.15 * size`` + centre crop (detail).
    ``flip_*``: horizontal mirror of the above.

    Embeddings of the views are averaged *after* L2 normalisation, so each view
    gets an equal vote. On food photos ``full`` matters most: dishes are often
    tightly framed and a centre crop can remove the food entirely.
    """
    interp = transforms.InterpolationMode.BICUBIC
    out: dict[str, transforms.Compose] = {}
    for view in views:
        base = view.replace("flip_", "")
        if base == "center":
            ops = [transforms.Resize(size, interpolation=interp), transforms.CenterCrop(size)]
        elif base == "full":
            ops = [transforms.Resize((size, size), interpolation=interp)]
        elif base == "zoom":
            ops = [
                transforms.Resize(int(round(size * 1.15)), interpolation=interp),
                transforms.CenterCrop(size),
            ]
        else:
            raise ValueError(f"unknown view {view!r}; choose from {sorted(config.EVAL_VIEWS)}")
        if view.startswith("flip_"):
            ops.append(transforms.RandomHorizontalFlip(p=1.0))
        out[view] = transforms.Compose(
            ops + [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    return out


def build_train_transform(size: int, mean, std, scale=(0.75, 1.0)) -> transforms.Compose:
    """Augmentation for fine-tuning the vision tower.

    Deliberately gentle: strong colour jitter erases the cues (colour, plating)
    that separate the deliberately-confusable categories, and erasing hides small
    objects. Random resized crop + flip, plus a mild jitter.
    """
    interp = transforms.InterpolationMode.BICUBIC
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size, scale=scale, ratio=(0.85, 1.18), interpolation=interp
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class ImagePathDataset(Dataset):
    """``(pixel_values, index)`` for a list of image paths, in order."""

    def __init__(self, paths: list[str], transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, idx


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-8)