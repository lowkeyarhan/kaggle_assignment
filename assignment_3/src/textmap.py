"""Query -> category mapping (the text half of the problem).

Every test query names its dish, usually verbatim ("A photograph of a croque
madame.") and sometimes through an alternative name ("Show me pictures of
potstickers." for gyoza). So the text half of the problem is close to string
matching against ``category_metadata.csv`` - which makes it nearly solvable -
and any residual disagreement between queries is fixed by the known "3 queries
per category" constraint.

The design has three layers, from most to least trusted:

1. **Lexical match** on normalised text (category name + alternative names),
   scored with IDF-weighted token coverage and a bonus for a contiguous phrase
   match.
2. **CLIP text-text similarity** between the query and each category's metadata
   text, which catches phrasings the string matcher misses.
3. **Balanced assignment** (``balance.balanced_assign``), which resolves the
   remaining conflicts globally under the exact query-per-category budget.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

import balance
import catalog

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Words that carry no discriminative signal in food descriptions.
STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "with", "in", "on", "it", "is", "this",
    "that", "picture", "photo", "photograph", "image", "images", "show", "me",
    "see", "want", "find", "looking", "for", "some", "one", "dish", "type",
    "food", "which", "is", "called", "known", "as", "served", "made", "contains",
    "please", "give", "display", "retrieval", "retrieve",
}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return _NON_ALNUM.sub(" ", str(text).lower()).strip()


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def _singular(token: str) -> str:
    """Crude singulariser - enough to match 'waffles' with 'waffle'."""
    if len(token) > 3 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _token_variants(tokens: list[str]) -> set[str]:
    out: set[str] = set()
    for token in tokens:
        out.add(token)
        out.add(_singular(token))
    return out


def build_idf(cats: pd.DataFrame) -> dict[str, float]:
    """IDF over category names + alternative names + descriptions.

    Rare words ("waffles", "ceviche") are the ones that identify a dish, so they
    must count for much more than "photo" or "dish".
    """
    docs: list[set[str]] = []
    for row in cats.itertuples():
        forms = catalog.surface_forms(pd.Series(row._asdict()))
        bag = set()
        for form in forms:
            bag |= set(tokenize(form))
        bag |= set(tokenize(row.description))
        docs.append(bag or {""})
    n_docs = len(docs)
    df = Counter(t for doc in docs for t in doc)
    return {t: math.log((1 + n_docs) / (1 + c)) + 1.0 for t, c in df.items()}


def _contiguous(haystack: list[str], needle: list[str]) -> bool:
    """True if ``needle`` appears as a consecutive run inside ``haystack``."""
    if not needle:
        return False
    n, m = len(haystack), len(needle)
    return any(haystack[i : i + m] == needle for i in range(n - m + 1))


def lexical_scores(
    queries: pd.DataFrame,
    cats: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Graded string-match score of every query against every category.

    Returns ``(scores [N, K], info)`` where ``info`` records, per query, the best
    matching surface form and whether the match was a full phrase hit - the two
    things worth eyeballing when a query is mapped to the wrong dish.
    """
    idf = build_idf(cats)
    default_idf = math.log(len(cats) + 1) + 1.0

    # Pre-tokenise the categories once.
    cat_forms: list[list[tuple[str, list[str], set[str], float]]] = []
    for row in cats.itertuples():
        forms = catalog.surface_forms(pd.Series(row._asdict()))
        parsed = []
        for form in forms:
            tokens = tokenize(form)
            if not tokens:
                continue
            variants = _token_variants(tokens)
            weight = sum(idf.get(t, default_idf) for t in variants)
            parsed.append((form, tokens, variants, max(weight, 1e-6)))
        # Longest (most informative) forms first.
        parsed.sort(key=lambda item: -item[3])
        cat_forms.append(parsed)

    query_tokens = [tokenize(q) for q in queries["description"]]
    query_variants = [_token_variants(toks) for toks in query_tokens]

    n, k = len(queries), len(cats)
    scores = np.zeros((n, k), dtype=np.float64)
    best_form = np.empty((n, k), dtype=object)
    best_exact = np.zeros((n, k), dtype=bool)

    for i, q_variants in enumerate(query_variants):
        for j, forms in enumerate(cat_forms):
            best = 0.0
            best_name = ""
            exact = False
            for form, form_tokens, form_variants, weight in forms:
                covered = sum(
                    idf.get(t, default_idf) for t in form_variants if t in q_variants
                )
                if covered <= 0:
                    continue
                coverage = covered / weight  # in [0, 1]
                phrase = 1.0 if _contiguous(query_tokens[i], form_tokens) else 0.0
                value = 0.6 * coverage + 0.9 * phrase
                if value > best:
                    best, best_name, exact = value, form, bool(phrase)
            scores[i, j] = best
            best_form[i, j] = best_name
            best_exact[i, j] = exact

    info = pd.DataFrame(
        {
            "query_id": queries["query_id"].values,
            "best_form": [best_form[i, int(np.argmax(scores[i]))] for i in range(n)],
            "best_score": [float(scores[i].max()) for i in range(n)],
            "exact_match": [bool(best_exact[i, int(np.argmax(scores[i]))]) for i in range(n)],
            "n_tied": [int(np.sum(scores[i] >= scores[i].max() - 1e-9)) for i in range(n)],
        }
    )
    return scores, info


def combine_scores(
    lexical: np.ndarray,
    clip_scores: np.ndarray | None,
    weight_lexical: float = 0.75,
) -> np.ndarray:
    """Fuse lexical and CLIP text scores after per-query standardisation."""
    fused = weight_lexical * balance.row_zscore(lexical)
    if clip_scores is not None:
        fused = fused + (1.0 - weight_lexical) * balance.row_zscore(clip_scores)
    return fused


def assign_categories(
    scores: np.ndarray,
    capacity: int | None = None,
    balanced: bool = True,
) -> np.ndarray:
    """Argmax, or a capacity-constrained assignment when ``balanced``."""
    if not balanced:
        return np.argmax(scores, axis=1)
    return balance.balanced_assign(scores, capacity=capacity)


def resolve(
    queries: pd.DataFrame,
    cats: pd.DataFrame,
    clip_scores: np.ndarray | None = None,
    weight_lexical: float = 0.75,
    balanced: bool = True,
    capacity: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Full text-side mapping.

    Returns ``(assignments DataFrame, score matrix used)``. The DataFrame has
    one row per query with the chosen ``category_id``/``category_name``, the
    margin to the runner-up category (low margin = worth inspecting) and the
    surface form that drove the decision.
    """
    lexical, info = lexical_scores(queries, cats)
    fused = combine_scores(lexical, clip_scores, weight_lexical=weight_lexical)
    assign = assign_categories(fused, capacity=capacity, balanced=balanced)

    rows = fused[np.arange(len(queries)), assign]
    # Margin over the best *other* category (margin is 0 when the assignment is
    # forced by the balanced solver).
    masked = fused.copy()
    masked[np.arange(len(queries)), assign] = -np.inf
    rows_alt = masked.max(axis=1)
    margin = rows - rows_alt

    out = pd.DataFrame(
        {
            "query_id": queries["query_id"].values,
            "description": queries["description"].values,
            "category_index": assign,
            "category_id": cats["category_id"].values[assign],
            "category_name": cats["category_name"].values[assign],
            "score": rows,
            "margin": margin,
        }
    ).merge(info[["query_id", "best_form", "exact_match", "n_tied"]], on="query_id")
    return out, fused