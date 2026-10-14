"""Check a submission file before uploading it.

    python3 scripts/check_submission.py submission.csv --data-dir match-the-dish-to-its-rizz

Same rules as the organisers' ``validate_submission.py``: header
``query_id,image_ids``, one row per test query, exactly 20 distinct ids from
``test_gallery.csv`` per row, space separated, best first. This copy lives in the
repo so you can validate without touching the data folder, but run the official
one too when submitting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

COLUMNS = ["query_id", "image_ids"]
TOP_K = 20


def check(submission, data_dir=".") -> int:
    """Raise ValueError describing every problem, or return the number of rows."""
    data_dir = Path(data_dir)
    queries = set(pd.read_csv(data_dir / "test_queries.csv", dtype=str).query_id)
    gallery = set(pd.read_csv(data_dir / "test_gallery.csv", dtype=str).image_id)

    sub = pd.read_csv(submission, dtype=str, keep_default_na=False)
    if [c.strip() for c in sub.columns] != COLUMNS:
        raise ValueError(f"Header must be exactly: {','.join(COLUMNS)}")
    sub.columns = COLUMNS
    sub = sub.apply(lambda s: s.str.strip())
    lists = sub.image_ids.str.split()

    errors: list[str] = []
    if sub.query_id.duplicated().any():
        dupes = sub.query_id[sub.query_id.duplicated()].tolist()[:5]
        errors.append(f"duplicated query_id: {dupes}")
    if queries - set(sub.query_id):
        errors.append(f"{len(queries - set(sub.query_id))} test queries are missing")
    if set(sub.query_id) - queries:
        errors.append(f"unknown query_id: {sorted(set(sub.query_id) - queries)[:5]}")
    if (lists.map(len) != TOP_K).any():
        errors.append(f"every row must contain exactly {TOP_K} image ids")
    bad = sorted({v for ids in lists for v in ids if v not in gallery})
    if bad:
        errors.append(f"{len(bad)} image ids not in test_gallery.csv, e.g. {bad[:5]}")
    if lists.map(lambda ids: len(set(ids)) != len(ids)).any():
        errors.append("image ids within a row must be distinct")
    if errors:
        raise ValueError("Invalid submission:\n - " + "\n - ".join(errors))
    return len(sub)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("submission")
    p.add_argument("--data-dir", default=".")
    args = p.parse_args()
    try:
        rows = check(args.submission, args.data_dir)
    except ValueError as exc:
        print(exc)
        sys.exit(1)
    print(f"OK: {rows} rows")


if __name__ == "__main__":
    main()