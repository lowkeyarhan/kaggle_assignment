"""Turning categories and queries into the text strings that CLIP/SigLIP sees.

Two different jobs, kept separate on purpose:

* :func:`category_prototype_texts` builds the *label* side - the text that
  stands for a category when scoring gallery images.
* :func:`query_texts` builds the *query* side - the raw competition description,
  which must be encoded as written (its wording never appears in training).
"""
from __future__ import annotations

import pandas as pd

import catalog
import config

# Number of surface forms per category that enter the prompt ensemble. Short
# lists keep text encoding cheap while still covering name + synonyms.
MAX_SURFACE_FORMS = 4


def category_prototype_texts(cats: pd.DataFrame, max_surface_forms: int = MAX_SURFACE_FORMS):
    """``list[list[str]]`` - the prompt ensemble for every category.

    Each category gets, for every prompt template and every written name of the
    dish (name, alternative names), one sentence. Alternative names are
    averaged in as *extra* prompts rather than replacing the name: a synonym is
    a weaker but still useful view of the same dish.
    """
    prototypes: list[list[str]] = []
    for row in cats.itertuples():
        forms = catalog.surface_forms(pd.Series(row._asdict()))[:max_surface_forms]
        description = str(row.description).strip().rstrip(".")
        texts: list[str] = []
        for form in forms:
            for template in config.PROMPT_TEMPLATES:
                texts.append(template.format(name=form))
        if description:
            for template in config.DESCRIPTION_TEMPLATES:
                texts.append(template.format(name=forms[0], description=description))
        prototypes.append(texts)
    return prototypes


def category_prototype_matrix(
    backbone: str, cats: pd.DataFrame, device=None, use_cache: bool = True, precision: str = "auto"
):
    """Embed the prompt ensemble and average it into one prototype per category.

    Returns ``(matrix [K, D] L2-normalised, n_prompts_per_category [K])``.

    Averaging the *normalised* prompt embeddings and re-normalising is the
    standard prompt-ensembling trick: it behaves like a soft "OR" over phrasings
    and is consistently better than any single template.
    """
    import numpy as np

    from features import encode_texts

    prototypes = category_prototype_texts(cats)
    flat = [text for group in prototypes for text in group]
    embedded = encode_texts(backbone, flat, device=device, use_cache=use_cache, precision=precision)
    counts = np.array([len(group) for group in prototypes], dtype=np.int64)
    matrix = np.zeros((len(prototypes), embedded.shape[1]), dtype=np.float64)
    offset = 0
    for i, count in enumerate(counts):
        block = embedded[offset : offset + count]
        matrix[i] = block.mean(axis=0)
        offset += int(count)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8, None)
    return matrix.astype(np.float32), counts


def query_texts(queries: pd.DataFrame) -> list[str]:
    """Encode the competition descriptions exactly as given."""
    return [str(d).strip() for d in queries["description"]]


def category_text_matrix(
    backbone: str, cats: pd.DataFrame, device=None, use_cache: bool = True, precision: str = "auto"
):
    """Embed a compact "category identity" matrix used by the text-side mapper.

    The query mapper must judge "which category does this sentence talk about",
    which is a text-text problem, not a captioning problem. This matrix holds one
    entry per category built from ``category_metadata`` only (name,
    alternative names, description, train paraphrases) so the *query* wording is
    never seen.
    """
    import numpy as np

    from features import encode_texts

    groups: list[list[str]] = []
    for row in cats.itertuples():
        forms = catalog.surface_forms(pd.Series(row._asdict()))
        description = str(row.description).strip().rstrip(".")
        texts: list[str] = []
        for form in forms:
            texts += [tpl.format(name=form) for tpl in config.QUERY_CATEGORY_TEMPLATES]
        if description:
            texts.append(f"{forms[0]}, {description}")
        texts += list(row.paraphrases)  # competition-provided paraphrases of the label
        groups.append(texts)

    flat = [t for g in groups for t in g]
    embedded = encode_texts(backbone, flat, device=device, use_cache=use_cache, precision=precision)
    counts = np.array([len(g) for g in groups], dtype=np.int64)
    matrix = np.zeros((len(groups), embedded.shape[1]), dtype=np.float64)
    offset = 0
    for i, count in enumerate(counts):
        matrix[i] = embedded[offset : offset + int(count)].mean(axis=0)
        offset += int(count)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8, None)
    return matrix.astype(np.float32), counts