"""Image -> category scoring (the visual half of the problem).

Five complementary views of the same question, fused and then re-balanced
against the known gallery structure:

``zero-shot``   cosine(image, category prompt ensemble) - no training at all,
                surprisingly strong on food, and completely independent of the
                (noisy) training labels.
``linear``       multinomial logistic regression on the frozen embeddings using
                the 2000 labelled training images - adapts the space to these
                exact 101 dishes.
``knn``          similarity-weighted vote from the training images - a local,
                non-parametric view that survives noisy labels better than a
                global linear head.
``label-prop``   diffusion of those scores over the gallery's own k-NN graph:
                the gallery clusters by dish, so a decision gets reinforced by
                its visual neighbours and isolated mistakes average out.
``sinkhorn``     entropic optimal transport onto the 101 categories, forcing each
                category to attract its ~19.8 gallery images.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import sparse

import balance


@dataclass
class ImageMapConfig:
    """Fusion weights and switches. Tuned on the validation split, not the LB."""

    weight_zero_shot: float = 0.30
    weight_linear: float = 0.40
    weight_knn: float = 0.20
    knn_k: int = 25
    knn_temp: float = 0.05
    logreg_C: float = 10.0
    prop_k: int = 15
    prop_alpha: float = 0.5
    prop_iters: int = 1
    prop_temp: float = 0.07
    balance: str = "sinkhorn"  # "sinkhorn" | "none"
    sinkhorn_temp: float = 0.35
    sinkhorn_iters: int = 100
    standardize: bool = True
    head_paths: tuple[str, ...] = ()  # trained probe heads from train.py (mode=probe)
    extra: dict = field(default_factory=dict)


def head_scores(
    feats: np.ndarray,
    path: str,
    n_classes: int,
    scale: float | None = None,
) -> np.ndarray:
    """Class log-probs from a probe head saved by ``train.py --mode probe``.

    The head was trained as ``logits = scale * normalize(x) @ normalize(W).T + b``
    on frozen features, so it is reproduced exactly here. The backbone used at
    inference must match the one used for training, otherwise the feature
    dimension check below fires.
    """
    blob = np.load(path, allow_pickle=False)
    weight = blob["weight"]
    bias = blob["bias"] if "bias" in blob.files else np.zeros(weight.shape[0])
    if weight.shape[0] != feats.shape[1]:
        raise ValueError(
            f"{path} expects {weight.shape[0]}-d features but the gallery has "
            f"{feats.shape[1]} - train the probe with the same backbone used here"
        )
    if weight.shape[0] and weight.shape[1] != n_classes:
        raise ValueError(f"{path} has {weight.shape[1]} classes, expected {n_classes}")
    step = float(blob["scale"]) if "scale" in blob.files else float(scale or 100.0)
    x = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8, None)
    w = weight / np.clip(np.linalg.norm(weight, axis=1, keepdims=True), 1e-8, None)
    logits = step * (x @ w.T) + bias
    logits = logits - logits.max(axis=1, keepdims=True)
    return logits - np.log(np.clip(np.exp(logits).sum(axis=1, keepdims=True), 1e-12, None))


def head_ensemble_scores(
    feats: np.ndarray,
    paths: tuple[str, ...] | list[str],
    n_classes: int,
) -> np.ndarray:
    """Average the log-probabilities of several fold heads.

    Fold ensembling is close to free here (the heads are tiny matrices) and
    reliably removes the variance of any single fold.
    """
    mats = [head_scores(feats, path, n_classes) for path in paths]
    return np.mean(mats, axis=0)


def zero_shot_scores(gallery_emb: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Cosine similarity of every gallery image against every category prototype."""
    return gallery_emb @ prototypes.T


def knn_scores(
    query_emb: np.ndarray,
    ref_emb: np.ndarray,
    ref_labels: np.ndarray,
    n_classes: int,
    k: int = 25,
    temp: float = 0.05,
) -> np.ndarray:
    """Similarity-weighted k-NN vote distribution over classes."""
    q = torch.from_numpy(np.ascontiguousarray(query_emb)).float()
    r = torch.from_numpy(np.ascontiguousarray(ref_emb)).float()
    labels = torch.from_numpy(np.asarray(ref_labels, dtype=np.int64))
    k = int(min(k, r.shape[0]))

    scores = torch.zeros(q.shape[0], n_classes, dtype=torch.float32)
    block = max(1, int(2**24 // max(r.shape[0], 1)))  # keep the similarity matrix small
    for start in range(0, q.shape[0], block):
        sims = q[start : start + block] @ r.T
        values, indices = sims.topk(k, dim=1)
        weights = torch.softmax(values / temp, dim=1)
        chunk = torch.zeros(values.shape[0], n_classes, dtype=torch.float32)
        chunk.scatter_add_(1, labels[indices], weights)
        scores[start : start + block] = chunk
    return scores.numpy()


def linear_scores(
    gallery_emb: np.ndarray,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    n_classes: int,
    C: float = 10.0,
    max_iter: int = 2000,
) -> np.ndarray:
    """Multinomial logistic regression on frozen features -> class log-probs.

    On 2000 images / 101 classes with strong pretrained features this linear
    probe usually beats both zero-shot and k-NN, and unlike k-NN it spreads
    probability mass sensibly over the confusable categories.
    """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        n_jobs=-1,
        # No multi_class/solver arguments: scikit-learn >=1.5 removed the
        # deprecated `multi_class` switch and solves multinomial with lbfgs by
        # default, which is what we want here.
    )
    clf.fit(train_emb, train_labels)
    log_proba = clf.predict_log_proba(gallery_emb)
    # predict_log_proba columns follow clf.classes_; re-index to 0..n_classes-1.
    full = np.full((log_proba.shape[0], n_classes), -30.0, dtype=np.float64)
    full[:, clf.classes_.astype(int)] = log_proba
    return full


def _knn_graph(emb: np.ndarray, k: int, temp: float) -> sparse.csr_matrix:
    """Row-stochastic, symmetrically normalised k-NN graph over the gallery."""
    x = torch.from_numpy(np.ascontiguousarray(emb)).float()
    n = x.shape[0]
    k = int(min(k, max(n - 1, 1)))
    rows, cols, vals = [], [], []
    block = max(1, int(2**24 // max(n, 1)))
    for start in range(0, n, block):
        sims = x[start : start + block] @ x.T
        rows_idx = torch.arange(start, min(start + block, n)).unsqueeze(1).expand(-1, k)
        vals_blk, cols_blk = sims.topk(k, dim=1)
        rows.append(rows_idx.flatten())
        cols.append(cols_blk.flatten())
        vals.append(torch.softmax(vals_blk / temp, dim=1).flatten())
    rows_t = torch.cat(rows).numpy()
    cols_t = torch.cat(cols).numpy()
    vals_t = torch.cat(vals).numpy()
    a = sparse.coo_matrix((vals_t, (rows_t, cols_t)), shape=(n, n)).tocsr()
    # D^-1/2 A D^-1/2 keeps the diffusion stable regardless of degree.
    deg = np.asarray(a.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-8, None))
    d = sparse.diags(dinv)
    return (d @ a @ d).tocsr()


def label_propagation(
    scores: np.ndarray,
    emb: np.ndarray,
    k: int = 15,
    alpha: float = 0.5,
    iters: int = 1,
    temp: float = 0.07,
) -> np.ndarray:
    """Diffuse class scores over the gallery's k-NN graph and re-inject them."""
    if alpha <= 0 or iters <= 0:
        return scores
    graph = _knn_graph(emb, k=k, temp=temp)
    out = np.asarray(scores, dtype=np.float64)
    for _ in range(iters):
        out = (1.0 - alpha) * scores + alpha * (graph @ out)
    return out


def fuse(matrices: list[np.ndarray], weights: list[float], standardize: bool = True) -> np.ndarray:
    """Weighted fusion of score matrices after per-row standardisation."""
    fused = np.zeros_like(np.asarray(matrices[0], dtype=np.float64))
    used = 0.0
    for mat, weight in zip(matrices, weights):
        if weight == 0:
            continue
        mat = np.asarray(mat, dtype=np.float64)
        fused += weight * (balance.row_zscore(mat) if standardize else mat)
        used += abs(weight)
    return fused / max(used, 1e-8)


def apply_balance(
    scores: np.ndarray,
    mode: str = "sinkhorn",
    temp: float = 0.35,
    n_iter: int = 100,
) -> np.ndarray:
    """Impose the gallery's known per-category image budget on the scores."""
    if mode == "none":
        return scores
    if mode == "sinkhorn":
        return balance.sinkhorn_log_plan(scores, temp=temp, n_iter=n_iter)
    raise ValueError(f"unknown balance mode {mode!r}")


def score_images(
    gallery_emb: np.ndarray,
    prototypes: np.ndarray | None,
    train_emb: np.ndarray | None = None,
    train_labels: np.ndarray | None = None,
    cfg: ImageMapConfig | None = None,
) -> dict[str, np.ndarray]:
    """Run the whole image-side pipeline.

    Returns a dict with the individual score matrices (``zero_shot``, ``linear``,
    ``knn``, ``fused_raw``) plus the final ranking matrix ``scores``, which is
    what :mod:`retrieval` uses to order a category's images.
    """
    cfg = cfg or ImageMapConfig()
    n_classes = prototypes.shape[0] if prototypes is not None else int(train_labels.max()) + 1
    out: dict[str, np.ndarray] = {}

    mats: list[np.ndarray] = []
    weights: list[float] = []

    if prototypes is not None and cfg.weight_zero_shot:
        zs = zero_shot_scores(gallery_emb, prototypes)
        out["zero_shot"] = zs
        mats.append(zs)
        weights.append(cfg.weight_zero_shot)

    have_train = train_emb is not None and train_labels is not None
    if cfg.head_paths:
        lin = head_ensemble_scores(gallery_emb, cfg.head_paths, n_classes)
        out["linear"] = lin
        mats.append(lin)
        weights.append(cfg.weight_linear or 1.0)
    elif have_train and cfg.weight_linear:
        lin = linear_scores(gallery_emb, train_emb, train_labels, n_classes, C=cfg.logreg_C)
        out["linear"] = lin
        mats.append(lin)
        weights.append(cfg.weight_linear)
    if have_train and cfg.weight_knn:
        knn = knn_scores(
            gallery_emb,
            train_emb,
            train_labels,
            n_classes,
            k=cfg.knn_k,
            temp=cfg.knn_temp,
        )
        out["knn"] = knn
        mats.append(knn)
        weights.append(cfg.weight_knn)

    if not mats:
        raise ValueError("score_images needs prototypes and/or labelled train features")

    fused = fuse(mats, weights, standardize=cfg.standardize)
    out["fused_raw"] = fused

    propagated = label_propagation(
        fused, gallery_emb, k=cfg.prop_k, alpha=cfg.prop_alpha, iters=cfg.prop_iters, temp=cfg.prop_temp
    )
    out["propagated"] = propagated
    out["scores"] = apply_balance(
        propagated, mode=cfg.balance, temp=cfg.sinkhorn_temp, n_iter=cfg.sinkhorn_iters
    )
    return out


def labels_from_frame(frame, category_ids: list[str]) -> np.ndarray:
    """Map a dataframe with a ``category_id`` column to class indices.

    Images whose category is unknown (possible in the test gallery, where labels
    are hidden) are dropped from the diagnostics rather than guessed.
    """
    index = {cid: i for i, cid in enumerate(category_ids)}
    mask = frame["category_id"].isin(index)
    return frame.loc[mask, "category_id"].map(index).to_numpy(dtype=np.int64)


def classification_accuracy(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Top-1 / top-5 image->category accuracy - the retrieval upper bound.

    R-precision for a query is 1.0 exactly when the top-20 list of that query
    equals the true images of its dish, so this number is the ceiling the
    image-side model can reach for correctly-mapped queries.
    """
    if labels.size == 0:
        return {"top1": float("nan"), "top5": float("nan"), "n": 0}
    order = np.argsort(-scores, axis=1)
    top1 = float((order[:, 0] == labels).mean())
    top5 = float(np.mean([labels[i] in order[i, :5] for i in range(len(labels))]))
    return {"top1": top1, "top5": top5, "n": int(len(labels))}