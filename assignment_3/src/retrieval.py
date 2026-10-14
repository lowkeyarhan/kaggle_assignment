"""Turning (query -> category) + (image -> category) into a top-20 ranking.

The metric is what makes this simple: with R = number of gallery photos of the
queried dish and exactly 20 ids submitted, R-precision is 1.0 as soon as the 20
submitted ids *are* the dish's photos. Order inside the top 20 does not change
it. So the ranking is

    for each query: take the 20 gallery images with the highest score for the
    query's predicted category

with two safety nets: images of the query's own category always come first, and
a direct query-image similarity can break ties for the report metrics (mAP@20,
MRR@20) without hurting R-precision when its weight is small.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

import config


def rank_for_category(
    category_scores: np.ndarray,
    category_index: int,
    image_ids: Sequence[str],
    top_k: int = config.DEFAULT_TOP_K,
    query_sim: np.ndarray | None = None,
    blend: float = 0.0,
) -> list[str]:
    """Best ``top_k`` image ids for one query.

    Args:
        category_scores: ``[N, K]`` image->category ranking scores.
        category_index: category predicted for this query.
        image_ids: gallery ids, aligned with the rows of ``category_scores``.
        top_k: number of ids to return (20).
        query_sim: optional ``[N]`` direct query-image similarity.
        blend: in ``[0, 1]``; weight of ``query_sim`` inside the category's
            candidate list. ``0`` maximises R-precision, larger values trade a
            little R-precision for better mAP/MRR.
    """
    own = np.asarray(category_scores[:, category_index], dtype=np.float64)
    if query_sim is not None and blend > 0:
        sim = np.asarray(query_sim, dtype=np.float64)
        own = own + blend * (sim - sim.mean()) / max(sim.std(), 1e-6)
    take = min(top_k, len(image_ids))
    order = np.argsort(-own)[:take]
    picked = [image_ids[i] for i in order]

    if len(picked) < top_k:  # only possible if a category is over-subscribed
        mask = np.ones(len(image_ids), dtype=bool)
        mask[order] = False
        for i in np.argsort(-np.where(mask, category_scores.max(axis=1), -np.inf)):
            picked.append(image_ids[i])
            if len(picked) == top_k:
                break
    return picked[:top_k]


def rank_queries(
    query_assignments: pd.DataFrame,
    category_scores: np.ndarray,
    image_ids: Sequence[str],
    query_sims: np.ndarray | None = None,
    top_k: int = config.DEFAULT_TOP_K,
    blend: float = 0.0,
) -> dict[str, list[str]]:
    """Rank the gallery for every query in ``query_assignments``.

    ``query_sims`` is optional ``[N_queries, N_images]``; only its row for the
    query at hand is used.
    """
    rankings: dict[str, list[str]] = {}
    for row_idx, row in enumerate(query_assignments.itertuples()):
        sim = None
        if query_sims is not None and blend > 0:
            sim = query_sims[row_idx]
        rankings[row.query_id] = rank_for_category(
            category_scores,
            int(row.category_index),
            image_ids,
            top_k=top_k,
            query_sim=sim,
            blend=blend,
        )
    return rankings


def query_image_similarity(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
) -> np.ndarray:
    """``[N_queries, N_images]`` cosine similarity (used only for blending)."""
    return query_emb @ gallery_emb.T


def write_submission(
    rankings: dict[str, list[str]],
    query_ids: Sequence[str],
    out_path: str,
    top_k: int = config.DEFAULT_TOP_K,
) -> str:
    """Write ``submission.csv`` (``query_id,image_ids``), space separated."""
    lines = ["query_id,image_ids"]
    for query_id in query_ids:
        ids = list(rankings[query_id])[:top_k]
        if len(ids) != top_k:
            raise ValueError(f"query {query_id} has {len(ids)} ids, expected {top_k}")
        if len(set(ids)) != top_k:
            raise ValueError(f"query {query_id} contains duplicate image ids")
        lines.append(f"{query_id},{' '.join(ids)}")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_path


def read_submission(path: str) -> dict[str, list[str]]:
    """Read a submission back into ``query_id -> [image_id]``."""
    import csv

    out: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["query_id"]] = str(row["image_ids"]).split()
    return out


def rankings_frame(rankings: dict[str, list[str]]) -> pd.DataFrame:
    """Long-format ranking table, handy for eyeballing in a notebook."""
    rows = [
        {"query_id": qid, "rank": rank, "image_id": iid}
        for qid, ids in rankings.items()
        for rank, iid in enumerate(ids, start=1)
    ]
    return pd.DataFrame(rows)