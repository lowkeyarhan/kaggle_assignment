"""CLIP/SigLIP embedding extraction with on-disk caching.

One backbone is loaded at a time, encodes images (optionally with TTA views) and
text, and every matrix is cached to ``assignment_3/features`` keyed by backbone
+ split + views. Re-running the pipeline after changing only the fusion weights
then costs seconds instead of a full GPU pass.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import dataset as ds

_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device):
    """fp16 autocast on CUDA, no-op elsewhere (fp16 is unsupported on MPS)."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def load_model(backbone: str, device: torch.device | None = None):
    """Load ``(model, preprocess, tokenizer)`` for a backbone registry key."""
    import open_clip  # imported lazily so --help works without the dep installed

    device = device or pick_device()
    arch, pretrained = config.backbone_spec(backbone)
    key = (arch, pretrained)
    if key not in _MODEL_CACHE:
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=device
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(arch)
        _MODEL_CACHE[key] = (model, preprocess, tokenizer)
    return _MODEL_CACHE[key]


def _ids_fingerprint(ids: list[str]) -> str:
    digest = hashlib.sha1("\n".join(ids).encode()).hexdigest()[:12]
    return digest


def _cache_path(tag: str) -> Path:
    return config.feature_cache_dir() / f"{tag}.npz"


def _save(tag: str, ids: list[str], feats: np.ndarray, extra: dict | None = None) -> None:
    payload = {"ids": np.array(ids, dtype=object), "feats": feats.astype(np.float32)}
    if extra:
        payload.update(extra)
    np.savez_compressed(_cache_path(tag), **payload)


def _load(tag: str, ids: list[str]):
    path = _cache_path(tag)
    if not path.exists():
        return None
    try:
        blob = np.load(path, allow_pickle=True)
    except Exception:
        return None
    cached_ids = list(blob["ids"])
    if cached_ids != ids:
        return None  # different gallery: never reuse a stale cache
    return {k: blob[k] for k in blob.files if k != "ids"}, cached_ids


@torch.no_grad()
def encode_images(
    backbone: str,
    image_ids: list[str],
    paths: list[str],
    views=("center", "full"),
    batch_size: int = 64,
    num_workers: int = 8,
    device: torch.device | None = None,
    use_cache: bool = True,
    progress: bool = True,
) -> tuple[np.ndarray, dict]:
    """Encode images into L2-normalised float32 features, averaged over views.

    Returns ``(features [N, D], info)``. Embeddings of different views are
    averaged *after* normalisation, which keeps each view's vote comparable.
    """
    device = device or pick_device()
    view_tag = "-".join(sorted(views))
    tag = config.cache_key("img", backbone, view_tag, _ids_fingerprint(image_ids))
    if use_cache:
        cached = _load(tag, image_ids)
        if cached is not None:
            return cached[0]["feats"], {"cached": True, "backbone": backbone, "views": list(views)}

    model, preprocess, _ = load_model(backbone, device)
    size = ds.infer_image_size(model, preprocess)
    mean, std = ds.extract_norm(preprocess)
    view_transforms = ds.build_eval_views(size, mean, std, views)

    per_view = []
    for view, tf in view_transforms.items():
        loader = DataLoader(
            ds.ImagePathDataset(paths, tf),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        chunks = []
        iterator = loader
        if progress:
            from tqdm import tqdm

            iterator = tqdm(loader, desc=f"{backbone}:{view}", leave=False)
        for pixels, _ in iterator:
            pixels = pixels.to(device, non_blocking=True)
            with autocast_ctx(device):
                feats = model.encode_image(pixels)
            feats = feats.float()
            chunks.append(ds.l2_normalize(feats).cpu().numpy())
        per_view.append(np.concatenate(chunks, axis=0))

    stacked = np.stack(per_view, axis=0)  # [V, N, D]
    feats = stacked.mean(axis=0)
    feats /= np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8, None)
    feats = feats.astype(np.float32)

    info = {
        "cached": False,
        "backbone": backbone,
        "views": list(views),
        "image_size": size,
        "dim": int(feats.shape[1]),
    }
    if use_cache:
        _save(tag, image_ids, feats, {"info": np.array([str(info)])})
    return feats, info


@torch.no_grad()
def encode_texts(
    backbone: str,
    texts: list[str],
    batch_size: int = 256,
    device: torch.device | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Encode a list of strings into L2-normalised float32 features."""
    device = device or pick_device()
    tag = config.cache_key("txt", backbone, _ids_fingerprint(texts))
    if use_cache:
        cached = _load(tag, texts)
        if cached is not None:
            return cached[0]["feats"]

    model, _, tokenizer = load_model(backbone, device)
    chunks = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[start : start + batch_size]).to(device)
        with autocast_ctx(device):
            feats = model.encode_text(tokens)
        chunks.append(ds.l2_normalize(feats.float()).cpu().numpy())
    feats = np.concatenate(chunks, axis=0).astype(np.float32)
    if use_cache:
        _save(tag, texts, feats)
    return feats


def free_backbone() -> None:
    """Drop cached models so several backbones can be encoded in one process."""
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def concat_fused(feats_by_backbone: list[np.ndarray], weights=None) -> np.ndarray:
    """L2-normalise each matrix, scale, then concatenate.

    Concatenating (rather than averaging) similarity matrices keeps the
    backbones' spaces separate and simply sums their cosine scores, which is the
    standard way to ensemble retrieval embeddings. Weights must be positive.
    """
    weights = np.ones(len(feats_by_backbone)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(weights) != len(feats_by_backbone):
        raise ValueError("weights and feature lists must have the same length")
    parts = []
    for w, feats in zip(weights, feats_by_backbone):
        f = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8, None)
        parts.append((w * f).astype(np.float32))
    return np.concatenate(parts, axis=1)