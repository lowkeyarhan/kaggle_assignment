"""Competition data catalog: pure pandas/numpy, no torch.

Everything that reads the competition CSVs lives here so the text-side logic
(query -> category mapping, metrics bookkeeping) can run without importing
torch. :mod:`dataset` adds the image pipeline on top.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_FILES = {
    "train": ("train.csv", "train_gallery"),
    "validation": ("validation_gallery.csv", "validation_gallery"),
    "test": ("test_gallery.csv", "test_gallery"),
}

SPLIT_QUERY_FILES = {
    "train": "train_queries.csv",
    "validation": "validation_queries.csv",
    "test": "test_queries.csv",
}


def split_list(value) -> list[str]:
    """Parse a pipe-separated metadata cell into a clean list of strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def load_categories(data_dir: str | Path) -> pd.DataFrame:
    """The 101 categories: name, description, alternative names, paraphrases."""
    df = pd.read_csv(Path(data_dir) / "category_metadata.csv")
    df["alternative_names"] = df["alternative_names"].map(split_list)
    df["paraphrases"] = df["train_paraphrases"].map(split_list)
    return df[
        ["category_id", "category_name", "description", "alternative_names", "paraphrases"]
    ]


def surface_forms(row: pd.Series) -> list[str]:
    """Every written name of a category: the name first, then alternative names.

    Alternative names matter because a query may name the dish by a synonym
    ("potstickers" for gyoza, "lasagne" for lasagna).
    """
    forms = [str(row["category_name"]).strip()]
    forms += [str(a).strip() for a in row["alternative_names"] if str(a).strip()]
    seen, out = set(), []
    for form in forms:
        if form.lower() not in seen:
            seen.add(form.lower())
            out.append(form)
    return out


def load_images(data_dir: str | Path, split: str) -> pd.DataFrame:
    """Gallery for a split with resolved image paths.

    Columns: ``image_id, path`` plus ``category_id``/``category_name`` for the
    labelled ``train`` split.
    """
    data_dir = Path(data_dir)
    csv_name, folder = SPLIT_FILES[split]
    df = pd.read_csv(data_dir / csv_name, dtype=str)
    df["path"] = [str(data_dir / folder / f"{iid}.jpg") for iid in df["image_id"]]
    head = df["path"].head(50)
    missing = [p for p in head if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"image not found, e.g. {missing[0]} (wrong --data?)")
    return df


def load_queries(data_dir: str | Path, split: str) -> pd.DataFrame:
    """Queries for a split (``query_id, description`` [+ labels for train])."""
    df = pd.read_csv(Path(data_dir) / SPLIT_QUERY_FILES[split], dtype=str)
    cols = ["query_id", "description"]
    if "category_id" in df.columns:
        cols += ["category_id", "category_name"]
    return df[cols]


def load_ground_truth(data_dir: str | Path) -> pd.DataFrame:
    """Validation ground truth with ``relevant_image_ids`` as a python list."""
    df = pd.read_csv(Path(data_dir) / "validation_ground_truth.csv", dtype=str)
    df["relevant_image_ids"] = df["relevant_image_ids"].map(lambda s: str(s).split())
    return df


def relevant_map(ground_truth: pd.DataFrame) -> dict[str, set[str]]:
    """``query_id -> set of relevant image ids``."""
    return {row.query_id: set(row.relevant_image_ids) for row in ground_truth.itertuples()}


def validation_image_labels(
    gallery: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cats: pd.DataFrame,
) -> np.ndarray:
    """Class index per validation gallery image (``-1`` = never relevant).

    Reconstructed from the ground truth the organisers ship, so diagnostics use
    nothing a participant is not given.
    """
    cat_index = {cid: i for i, cid in enumerate(cats["category_id"])}
    labels = np.full(len(gallery), -1, dtype=np.int64)
    image_pos = {iid: i for i, iid in enumerate(gallery["image_id"])}
    for row in ground_truth.itertuples():
        target = cat_index[row.category_id]
        for iid in row.relevant_image_ids:
            pos = image_pos.get(iid)
            if pos is not None:
                labels[pos] = target
    return labels