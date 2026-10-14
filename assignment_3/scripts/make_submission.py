"""Validate *and* submit in one process, so the backbone is loaded once.

    python3 scripts/make_submission.py --backbone siglip2-so400m-384 \
        --weight-linear 0.2 --prop-alpha 0.3 --use-validation-labels --out submission.csv

Encoding dominates the runtime (and, on a laptop GPU, so does copying the
weights to the device). Doing validation and the test split in one process means
one model load and one device transfer instead of two, and it guarantees the
number you validate is produced by exactly the code that writes the submission.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import catalog  # noqa: E402
import config  # noqa: E402
import evaluate as ev  # noqa: E402
import metrics  # noqa: E402
import pipeline  # noqa: E402
import retrieval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ev.add_common_args(parser)
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    cfg = ev.build_config(args)
    root = config.resolve_data_dir(args.data)

    # --- validation: the number that decides whether this config is worth submitting
    ground_truth = catalog.load_ground_truth(root)
    val = pipeline.run_pipeline(root, "validation", cfg, ground_truth=ground_truth)
    scores = metrics.evaluate_rankings(val.rankings, catalog.relevant_map(ground_truth))
    diag = val.diagnostics
    print(
        f"[validation] R-precision {scores['r_precision']:.4f} | mAP@20 {scores['mAP@20']:.4f} | "
        f"P@5 {scores['P@5']:.4f} | Hit@1 {scores['Hit@1']:.4f} | MRR@20 {scores['MRR@20']:.4f}"
    )
    print(
        f"[validation] query->category {diag.get('query_category_accuracy'):.4f} | "
        f"image->category top1 {diag.get('image_top1'):.4f}"
    )

    # --- test: same code path, plus the labelled validation gallery as extra supervision
    test_cfg = pipeline.PipelineConfig(**{**cfg.__dict__})
    test = pipeline.run_pipeline(root, "test", test_cfg)
    retrieval.write_submission(
        test.rankings, test.queries["query_id"].tolist(), args.out, top_k=test_cfg.top_k
    )
    print(f"[test] wrote {args.out} ({len(test.rankings)} queries x {test_cfg.top_k} ids)")

    report = {
        "config": {
            "backbones": list(cfg.backbones),
            "views": list(cfg.views),
            "weight_zero_shot": cfg.image_cfg.weight_zero_shot,
            "weight_linear": cfg.image_cfg.weight_linear,
            "weight_knn": cfg.image_cfg.weight_knn,
            "prop_alpha": cfg.image_cfg.prop_alpha,
            "sinkhorn_temp": cfg.image_cfg.sinkhorn_temp,
            "add_validation_labels": cfg.add_validation_labels,
        },
        "validation_metrics": scores,
        "validation_diagnostics": diag,
    }
    out_json = Path("results") / f"run_{'_'.join(cfg.backbones)}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=float))
    print("[report]", out_json)
    print(
        "[validate] python3 scripts/check_submission.py "
        f"{args.out} --data-dir {root}"
    )


if __name__ == "__main__":
    main()