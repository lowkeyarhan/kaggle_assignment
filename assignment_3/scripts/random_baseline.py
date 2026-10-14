"""Write a random valid submission - a format smoke test, not a model.

    python3 scripts/random_baseline.py --out random_submission.csv
    python3 scripts/check_submission.py random_submission.csv --data-dir match-the-dish-to-its-rizz

Random ranking scores ~0.01 R-precision. It exists so the submission writer,
the validator and the Kaggle upload path are proven before any GPU time is
spent.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="match-the-dish-to-its-rizz")
    p.add_argument("--out", default="random_submission.csv")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    queries = pd.read_csv(data_dir / "test_queries.csv", dtype=str).query_id.tolist()
    gallery = pd.read_csv(data_dir / "test_gallery.csv", dtype=str).image_id.tolist()
    rng = random.Random(args.seed)

    rows = [
        f"{qid},{' '.join(rng.sample(gallery, args.top_k))}" for qid in queries
    ]
    Path(args.out).write_text("query_id,image_ids\n" + "\n".join(rows) + "\n")
    print(f"wrote {args.out}: {len(queries)} rows (random baseline, ~0.01 R-precision)")


if __name__ == "__main__":
    main()