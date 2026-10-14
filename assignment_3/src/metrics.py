"""Retrieval metrics: the leaderboard R-precision plus the report metrics.

The competition scores a query with

    R-precision = |top-R predictions intersect relevant| / R,   R = |relevant|

and averages that over queries. Note that R equals the number of gallery images
of the queried dish (10 in validation, 19-20 in test), so with exactly 20
submitted ids the metric only cares about *which 20 images* you pick - the order
inside the first R never changes it. That single observation drives the whole
solution design (see README).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

TOP_K = 20


def r_precision(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """Fraction of the dish's R photos that appear in the first R predictions."""
    relevant = set(relevant)
    if not relevant:
        return float("nan")
    r = len(relevant)
    return len(set(list(ranked)[:r]) & relevant) / r


def average_precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = TOP_K) -> float:
    """AP@k: rewards pushing correct images earlier (report metric)."""
    relevant = set(relevant)
    if not relevant:
        return float("nan")
    hits, total = 0, 0.0
    for rank, image_id in enumerate(list(ranked)[:k], start=1):
        if image_id in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = 5) -> float:
    relevant = set(relevant)
    if not relevant:
        return float("nan")
    return sum(1 for i in list(ranked)[:k] if i in relevant) / k


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return float("nan")
    return float(any(i in relevant for i in list(ranked)[:k]))


def mrr_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = TOP_K) -> float:
    relevant = set(relevant)
    if not relevant:
        return float("nan")
    for rank, image_id in enumerate(list(ranked)[:k], start=1):
        if image_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    relevant: Mapping[str, Iterable[str]],
) -> dict[str, float]:
    """All metrics for a full split, plus the number of queries scored."""
    scores = {name: [] for name in ("r_precision", "mAP@20", "P@5", "Hit@1", "Hit@5", "MRR@20")}
    for query_id, ranked in rankings.items():
        rel = relevant.get(query_id)
        if not rel:
            continue
        scores["r_precision"].append(r_precision(ranked, rel))
        scores["mAP@20"].append(average_precision_at_k(ranked, rel))
        scores["P@5"].append(precision_at_k(ranked, rel))
        scores["Hit@1"].append(hit_at_k(ranked, rel, 1))
        scores["Hit@5"].append(hit_at_k(ranked, rel, 5))
        scores["MRR@20"].append(mrr_at_k(ranked, rel))
    out = {k: (sum(v) / len(v) if v else float("nan")) for k, v in scores.items()}
    out["n_queries"] = float(len(scores["r_precision"]))
    return out


def safe_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")