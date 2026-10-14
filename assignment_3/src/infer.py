"""Build ``submission.csv`` for the test split.

Run from the assignment root::

    python3 src/infer.py --backbone siglip2-so400m-384 --out submission.csv
    python3 src/infer.py --backbone dfn5b-h14-378 siglip2-gopt-384 --out submission.csv

Everything except the split and the output is identical to ``evaluate.py``, so
whatever scored best on validation is exactly what produces the submission.
Always check the result with the organisers' own validator::

    python match-the-dish-to-its-rizz/validate_submission.py submission.csv \
        --data-dir match-the-dish-to-its-rizz
"""
from __future__ import annotations

import argparse
import os

import config
import evaluate as ev
import pipeline
import retrieval


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    ev.add_common_args(p)
    p.add_argument("--out", default="submission.csv")
    return p.parse_args()


def main() -> None:
    args = parse()
    cfg = ev.build_config(args)
    root = config.resolve_data_dir(args.data)

    result = pipeline.run_pipeline(root, "test", cfg)
    retrieval.write_submission(
        result.rankings,
        result.queries["query_id"].tolist(),
        args.out,
        top_k=cfg.top_k,
    )

    diag = result.diagnostics
    print(f"wrote {args.out}: {len(result.rankings)} queries x {cfg.top_k} image ids")
    print(
        f"backbone(s): {', '.join(cfg.backbones)} | views: {','.join(cfg.views)} | "
        f"image balancing: {cfg.image_cfg.balance}"
    )
    print(
        f"exact text matches: {diag['exact_text_match']:.3f} | "
        f"ambiguous queries: {diag['ambiguous_queries']}"
    )
    print(
        "validate with: python "
        f"{os.path.join(root, 'validate_submission.py')} {args.out} --data-dir {root}"
    )


if __name__ == "__main__":
    main()