"""CLIP/SigLIP embedding extraction with on-disk caching.

One backbone is loaded at a time, encodes images (optionally with TTA views) and
text, and every matrix is cached to ``assignment_3/features`` keyed by backbone
+ split + views. Re-running the pipeline after changing only the fusion weights
then costs seconds instead of a full GPU pass.
"""
from __future__ import annotations

import hashlib
import sys
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


def autocast_dtype(device: torch.device, precision: str = "auto"):
    """Pick an autocast dtype.

    ``auto`` uses fp16 on CUDA and bf16 on Apple silicon (fp16 is unreliable in
    some MPS kernels, bf16 is supported since macOS 14); CPU and explicit
    ``fp32`` run without autocast.
    """
    if precision == "fp32" or device.type == "cpu":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps" and torch.backends.mps.is_available():
        return torch.bfloat16
    return None


def autocast_ctx(device: torch.device, precision: str = "auto"):
    dtype = autocast_dtype(device, precision)
    if dtype is None:
        return torch.autocast("cpu", enabled=False)
    return torch.autocast(device.type, dtype=dtype)


def _mps_expand_hotfix(model: torch.nn.Module) -> int:
    """Work around an MPS backend bug with stride-0 expanded tensors.

    timm's attention-pool head (`AttentionPoolLatent`) feeds
    ``self.latent.expand(B, -1, -1)`` straight into a Linear layer. On MPS that
    stride-0 view aborts the process for any batch > 1
    (``[MPSNDArray ...] buffer is not large enough``) while batch 1 silently
    works. Cloning the expansion to a contiguous tensor in a forward pre-hook
    keeps everything on the GPU and costs one cheap copy.
    """
    patched = 0
    try:
        from timm.layers import AttentionPoolLatent
    except ImportError:
        return 0
    for module in model.modules():
        if isinstance(module, AttentionPoolLatent) and not getattr(module, "_mps_hotfixed", False):
            def _declone(_mod, args, _kwargs=None):
                data = args[0]
                if data.stride(0) == 0:
                    data = data.contiguous()
                return ((data,) + args[1:], _kwargs or {})

            module.q.register_forward_pre_hook(_declone, with_kwargs=True)
            module._mps_hotfixed = True
            patched += 1
    return patched


def load_model(backbone: str, device: torch.device | None = None, precision: str = "auto"):
    """Load ``(model, preprocess, tokenizer)`` for a backbone registry key.

    The model is built on the CPU and cast to the autocast dtype *before* it
    moves to the GPU: shipping fp32 weights to the device doubles the transfer
    (4.5 GB for SO400M) and the transfer is a real cost, not a rounding error.
    """
    import open_clip  # imported lazily so --help works without the dep installed

    device = device or pick_device()
    arch, pretrained = config.backbone_spec(backbone)
    dtype = autocast_dtype(device, precision)
    key = (arch, pretrained, str(dtype))
    if key not in _MODEL_CACHE:
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device="cpu"
        )
        if dtype is not None:
            model = model.to(dtype=dtype)
        model = model.eval().to(device)
        if device.type == "mps":
            n = _mps_expand_hotfix(model)
            if n:
                print(f"[features] applied MPS expand hotfix to {n} attention-pool head(s)")
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
    feats = blob["feats"]
    if not np.isfinite(feats).all():
        # A cache written by an older, unguarded run can hold NaNs/inf; treat it
        # as a miss rather than poisoning everything downstream.
        return None
    return {k: blob[k] for k in blob.files if k != "ids"}, cached_ids


def _encode_paths(
    model,
    view_transforms: dict,
    paths: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    dtype,
    progress: bool,
    tag: str,
) -> np.ndarray:
    """Encode ``paths`` with every view and return the averaged, normalised matrix."""
    per_view = []
    for view, tf in view_transforms.items():
        loader = DataLoader(
            ds.ImagePathDataset(paths, tf),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        iterator = loader
        if progress:
            from tqdm import tqdm

            iterator = tqdm(loader, desc=f"{tag}:{view}", leave=False)
        chunks = []
        for pixels, _ in iterator:
            pixels = pixels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype) if dtype else _nullcontext():
                feats = model.encode_image(pixels)
            chunks.append(ds.l2_normalize(feats.float()).cpu().numpy())
        per_view.append(np.concatenate(chunks, axis=0))
    return per_view


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


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
    progress: bool | None = None,
    precision: str = "auto",
) -> tuple[np.ndarray, dict]:
    """Encode images into L2-normalised float32 features, averaged over views.

    Returns ``(features [N, D], info)``. Embeddings of different views are
    averaged *after* normalisation, which keeps each view's vote comparable.
    """
    device = device or pick_device()
    progress = sys.stdout.isatty() if progress is None else progress
    view_tag = "-".join(sorted(views))
    tag = config.cache_key("img", backbone, view_tag, precision, _ids_fingerprint(image_ids))
    if use_cache:
        cached = _load(tag, image_ids)
        if cached is not None:
            return cached[0]["feats"], {"cached": True, "backbone": backbone, "views": list(views)}

    model, preprocess, _ = load_model(backbone, device, precision)
    size = ds.infer_image_size(model, preprocess)
    mean, std = ds.extract_norm(preprocess)
    view_transforms = ds.build_eval_views(size, mean, std, views)

    per_view = _encode_paths(
        model,
        view_transforms,
        paths,
        device,
        batch_size,
        num_workers,
        autocast_dtype(device, precision),
        progress,
        f"{backbone}",
    )

    stacked = np.stack(per_view, axis=0)  # [V, N, D]
    feats = stacked.mean(axis=0)

    # Low-precision autocast can overflow on a handful of images. Re-encode those
    # in fp32 rather than letting one NaN poison the whole similarity matrix.
    bad = ~np.isfinite(feats).all(axis=1)
    if bad.any():
        bad_idx = np.flatnonzero(bad)
        print(
            f"[features] {len(bad_idx)} non-finite embeddings for {backbone}; "
            "retrying those images in fp32"
        )
        retry = _encode_paths(
            model,
            view_transforms,
            [paths[i] for i in bad_idx],
            device,
            batch_size,
            num_workers,
            None,  # fp32
            progress,
            f"{backbone}-fp32-retry",
        )
        fixed = np.stack(retry, axis=0).mean(axis=0)
        still_bad = ~np.isfinite(fixed).all(axis=1)
        if still_bad.any():
            print(f"[features] {int(still_bad.sum())} images stay unusable; zeroing them")
            fixed[still_bad] = 0.0
        feats[bad_idx] = fixed

    feats /= np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8, None)
    feats = feats.astype(np.float32)

    info = {
        "cached": False,
        "backbone": backbone,
        "views": list(views),
        "image_size": size,
        "precision": precision,
        "zero_rows": int((np.linalg.norm(feats, axis=1) <= 1e-8).sum()),
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
    precision: str = "auto",
) -> np.ndarray:
    """Encode a list of strings into L2-normalised float32 features."""
    device = device or pick_device()
    tag = config.cache_key("txt", backbone, precision, _ids_fingerprint(texts))
    if use_cache:
        cached = _load(tag, texts)
        if cached is not None:
            return cached[0]["feats"]

    model, _, tokenizer = load_model(backbone, device, precision)
    chunks = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[start : start + batch_size]).to(device)
        with autocast_ctx(device, precision):
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