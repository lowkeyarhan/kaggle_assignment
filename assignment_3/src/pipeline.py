"""One pipeline, two entry points: validate on the validation split, submit on test.

``run_pipeline`` does the whole thing and returns everything a report needs:
the rankings, the query->category decisions with their margins, the raw
image->category score matrices, and the diagnostics that explain *why* a score
is what it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import catalog
import config
import features
import imagemap
import prompts
import retrieval
import textmap


@dataclass
class PipelineConfig:
    """Everything the pipeline can be tuned with (defaults = my recommended run)."""

    backbones: tuple[str, ...] = (config.DEFAULT_BACKBONE,)
    views: tuple[str, ...] = config.DEFAULT_VIEWS
    text_backbone: str | None = None  # defaults to the first vision backbone
    backbone_weights: tuple[float, ...] | None = None
    weight_lexical: float = 0.75
    balance_queries: bool = True
    blend_query_sim: float = 0.0
    top_k: int = config.DEFAULT_TOP_K
    batch_size: int = 64
    num_workers: int = 8
    device: str | None = None
    precision: str = "auto"  # "auto" = fp16 on CUDA, bf16 on Apple silicon, fp32 on CPU
    use_cache: bool = True
    image_cfg: imagemap.ImageMapConfig = field(default_factory=imagemap.ImageMapConfig)


@dataclass
class PipelineResult:
    split: str
    data_dir: str
    categories: pd.DataFrame
    gallery: pd.DataFrame
    queries: pd.DataFrame
    image_ids: list[str]
    rankings: dict[str, list[str]]
    query_assignments: pd.DataFrame
    image_scores: dict[str, np.ndarray]
    query_sims: np.ndarray | None
    diagnostics: dict


def _encode_roles(
    cfg: PipelineConfig,
    cats: pd.DataFrame,
    gallery: pd.DataFrame,
    queries: pd.DataFrame,
    train: pd.DataFrame,
):
    """Encode every role (gallery / train / query / category text) per backbone."""
    device = features.pick_device(cfg.device)
    weights = cfg.backbone_weights or (1.0,) * len(cfg.backbones)
    gallery_parts, train_parts, query_parts, proto_parts, text_parts = [], [], [], [], []
    query_texts = prompts.query_texts(queries)

    for backbone, weight in zip(cfg.backbones, weights):
        gal, _ = features.encode_images(
            backbone,
            gallery["image_id"].tolist(),
            gallery["path"].tolist(),
            views=cfg.views,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            device=device,
            use_cache=cfg.use_cache,
            precision=cfg.precision,
        )
        trn, _ = features.encode_images(
            backbone,
            train["image_id"].tolist(),
            train["path"].tolist(),
            views=cfg.views,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            device=device,
            use_cache=cfg.use_cache,
            precision=cfg.precision,
        )
        proto, _ = prompts.category_prototype_matrix(
            backbone, cats, device=device, use_cache=cfg.use_cache, precision=cfg.precision
        )
        text_matrix, _ = prompts.category_text_matrix(
            backbone, cats, device=device, use_cache=cfg.use_cache, precision=cfg.precision
        )
        qemb = features.encode_texts(
            backbone, query_texts, device=device, use_cache=cfg.use_cache, precision=cfg.precision
        )

        gallery_parts.append(gal)
        train_parts.append(trn)
        query_parts.append(qemb)
        proto_parts.append(proto)
        text_parts.append(text_matrix)
        features.free_backbone()

    return (
        _fuse(gallery_parts, weights),
        _fuse(train_parts, weights),
        _fuse(query_parts, weights),
        _fuse(proto_parts, weights),
        _fuse(text_parts, weights),
    )


def _fuse(parts: list[np.ndarray], weights) -> np.ndarray:
    return features.concat_fused(parts, weights)


def query_label_accuracy(assignments: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Fraction of queries mapped to the right category (validation only)."""
    truth = dict(zip(ground_truth["query_id"], ground_truth["category_id"]))
    hits = [
        1 for row in assignments.itertuples() if truth.get(row.query_id) == row.category_id
    ]
    return float(len(hits)) / max(len(assignments), 1)


def run_pipeline(
    data_dir: str | None = None,
    split: str = "validation",
    cfg: PipelineConfig | None = None,
    ground_truth: pd.DataFrame | None = None,
) -> PipelineResult:
    """Run query mapping + image mapping + ranking for one split.

    ``ground_truth`` is optional and only used for diagnostics: pass
    ``catalog.load_ground_truth(...)`` on the validation split to get query-side
    and image-side accuracy numbers alongside the rankings.
    """
    cfg = cfg or PipelineConfig()
    root = config.resolve_data_dir(data_dir)

    cats = catalog.load_categories(root)
    gallery = catalog.load_images(root, split)
    queries = catalog.load_queries(root, split)
    train = catalog.load_images(root, "train")

    gallery_emb, train_emb, query_emb, proto, cat_text = _encode_roles(
        cfg, cats, gallery, queries, train
    )

    # --- text side: which dish does each query talk about? ---
    clip_text_scores = query_emb @ cat_text.T
    assignments, text_scores = textmap.resolve(
        queries,
        cats,
        clip_scores=clip_text_scores,
        weight_lexical=cfg.weight_lexical,
        balanced=cfg.balance_queries,
    )

    # --- image side: which dish is each gallery image? ---
    train_labels = imagemap.labels_from_frame(train, cats["category_id"].tolist())
    image_scores = imagemap.score_images(
        gallery_emb, proto, train_emb, train_labels, cfg.image_cfg
    )

    # --- ranking ---
    query_sims = None
    if cfg.blend_query_sim > 0:
        query_sims = retrieval.query_image_similarity(query_emb, gallery_emb)
    rankings = retrieval.rank_queries(
        assignments,
        image_scores["scores"],
        gallery["image_id"].tolist(),
        query_sims=query_sims,
        top_k=cfg.top_k,
        blend=cfg.blend_query_sim,
    )

    diagnostics: dict = {
        "n_queries": int(len(queries)),
        "n_images": int(len(gallery)),
        "n_categories": int(len(cats)),
        "backbones": list(cfg.backbones),
        "views": list(cfg.views),
        "exact_text_match": float(assignments["exact_match"].mean()),
        "min_margin": float(assignments["margin"].min()),
        "ambiguous_queries": int((assignments["margin"] < 0.5).sum()),
    }

    if ground_truth is not None:
        image_labels = catalog.validation_image_labels(gallery, ground_truth, cats)
        known = image_labels >= 0
        diagnostics["query_category_accuracy"] = query_label_accuracy(assignments, ground_truth)
        diagnostics["image_category_accuracy"] = {
            name: imagemap.classification_accuracy(
                mat[known], image_labels[known]
            )
            for name, mat in image_scores.items()
        }
        # Ceiling: R-precision with a perfect query mapper and these image scores.
        diagnostics["image_top1"] = diagnostics["image_category_accuracy"]["scores"]["top1"]

    return PipelineResult(
        split=split,
        data_dir=str(root),
        categories=cats,
        gallery=gallery,
        queries=queries,
        image_ids=gallery["image_id"].tolist(),
        rankings=rankings,
        query_assignments=assignments,
        image_scores=image_scores,
        query_sims=query_sims,
        diagnostics=diagnostics,
    )