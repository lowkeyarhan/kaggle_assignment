"""Assignment helpers: standardisation plus the two balanced-assignment tricks.

Both splits of this competition have a rigid structure that a plain argmax
throws away:

* the test gallery holds 2000 images for 101 categories (19 or 20 each), so the
  101 image->category columns should receive ~19.8 images each;
* the test set has 303 queries = exactly 3 per category, and validation has 202
  = exactly 2 per category, so queries must be spread evenly over categories.

Imposing those marginals is worth far more than any single modelling tweak,
because R-precision is 1.0 as soon as the top-20 list of a query is exactly the
images of its (correctly predicted) dish.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


def row_zscore(scores: np.ndarray) -> np.ndarray:
    """Standardise every row, so scores from different sources are comparable.

    Cosine similarities, kNN votes and classifier logits all live on different
    scales; z-scoring per row (per image, over categories) puts them on a common
    footing before fusing.
    """
    scores = np.asarray(scores, dtype=np.float64)
    mu = scores.mean(axis=1, keepdims=True)
    sd = scores.std(axis=1, keepdims=True)
    return (scores - mu) / np.clip(sd, 1e-6, None)


def row_softmax(scores: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64) / temp
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def sinkhorn_log_plan(
    scores: np.ndarray,
    temp: float = 0.25,
    n_iter: int = 100,
    col_marginal: np.ndarray | None = None,
) -> np.ndarray:
    """Entropic optimal transport of images onto categories, in log space.

    Solves ``diag(exp(log_u)) K diag(exp(log_v)) -> row_marginal, col_marginal``
    with ``K = exp(scores / temp)``. With uniform marginals every category is
    forced to attract ``N / K`` images, which is exactly the competition's
    gallery structure; a wrong-but-confident image gets pushed out of an
    over-full category instead of eating a slot of the right one.

    Returns the **log** transport plan (use it directly for ranking; taking log
    of the plan itself loses all precision).
    """
    scores = np.asarray(scores, dtype=np.float64)
    n, k = scores.shape
    log_row = np.full(n, -np.log(n))
    if col_marginal is None:
        log_col = np.full(k, -np.log(k))
    else:
        col = np.asarray(col_marginal, dtype=np.float64)
        col = np.clip(col, 1e-12, None)
        log_col = np.log(col / col.sum())
    kernel = scores / temp
    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(k, dtype=np.float64)
    for _ in range(n_iter):
        log_u = log_row - logsumexp(kernel + log_v[None, :], axis=1)
        log_v = log_col - logsumexp(kernel + log_u[:, None], axis=0)
    return kernel + log_u[:, None] + log_v[None, :]


def balanced_assign(scores: np.ndarray, capacity: int | None = None) -> np.ndarray:
    """Exact capacity-constrained assignment (Hungarian) of rows to columns.

    Every category column is duplicated ``capacity`` times, so at most
    ``capacity`` rows can be assigned to it. Rows beyond ``capacity * K`` (which
    cannot happen for a well-formed split) are filled greedily.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n, k = scores.shape
    capacity = n // k if capacity is None else int(capacity)
    cols = np.repeat(np.arange(k), capacity)
    if cols.size < n:  # capacity too small for this split: pad with round-robin
        extra = np.tile(np.arange(k), int(np.ceil(n / max(cols.size, 1))))
        cols = np.concatenate([cols, extra])[: max(cols.size, n)]
    rows, col_idx = linear_sum_assignment(-scores[:, cols])
    assign = np.full(n, -1, dtype=np.int64)
    assign[rows] = cols[col_idx]
    leftovers = np.where(assign < 0)[0]
    for row in leftovers:
        assign[row] = int(np.argmax(scores[row]))
    return assign


def greedy_capacity_assign(scores: np.ndarray, capacities: list[int]) -> np.ndarray:
    """Greedy assignment respecting an exact per-category capacity.

    Cheaper and more aggressive than Sinkhorn: process the globally most
    confident (row, column) pairs first. Useful as a diagnostic upper bound on
    what perfect balancing could give.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n, k = scores.shape
    remaining = np.asarray(capacities, dtype=np.int64).copy()
    flat = np.dstack(np.unravel_index(np.argsort(-scores, axis=None), scores.shape))[0]
    assign = np.full(n, -1, dtype=np.int64)
    for row, col in flat:
        if assign[row] >= 0 or remaining[col] <= 0:
            continue
        assign[row] = col
        remaining[col] -= 1
        if (assign >= 0).all():
            break
    for row in np.where(assign < 0)[0]:
        order = np.argsort(-remaining)
        col = int(order[0]) if remaining.max() > 0 else int(np.argmax(scores[row]))
        assign[row] = col
        remaining[col] = max(int(remaining[col]) - 1, 0)
    return assign


def split_capacity(n_rows: int, n_cols: int) -> list[int]:
    """Column capacities for a split with ``n_rows`` rows over ``n_cols`` classes.

    Spreads the remainder one-by-one across the first categories, mirroring how
    the organisers built the galleries (81 categories with 20 images, 20 with 19).
    """
    base, rem = divmod(n_rows, n_cols)
    caps = [base] * n_cols
    for i in range(rem):
        caps[i] += 1
    return caps