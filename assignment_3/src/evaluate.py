"""Validate the pipeline on the validation split and print every metric.

Run from the assignment root::

    python3 src/evaluate.py --backbone siglip2-so400m-384
    python3 src/evaluate.py --backbone dfn5b-h14-378 siglip2-gopt-384 --weight 1 1

The report metrics (mAP@20, P@5, Hit@1/5, MRR@20) come from the same rankings as
the leaderboard R-precision, and the diagnostics explain where the score is lost:
a query mapped to the wrong category, or a gallery photo pushed into the wrong
category's top-20.
"""
from __future__ import annotations

import argparse
import json

import catalog
import config
import imagemap
import metrics
import pipeline


def add_common_args(p: argparse.ArgumentParser) -> None:
    """CLI flags shared by evaluate.py and infer.py."""
    p.add_argument("--data", default=None, help="competition data dir (default: auto-detect)")
    p.add_argument(
        "--backbone",
        nargs="+",
        default=[config.DEFAULT_BACKBONE],
        choices=sorted(config.BACKBONES),
        help="one or more backbones; several are concatenated (ensemble)",
    )
    p.add_argument("--backbone-weight", type=float, nargs="*", default=None)
    p.add_argument(
        "--views",
        nargs="+",
        default=list(config.DEFAULT_VIEWS),
        choices=list(config.EVAL_VIEWS),
        help="TTA views averaged at embedding time",
    )
    p.add_argument("--weight-lexical", type=float, default=0.75,
                   help="weight of string matching vs CLIP text similarity in query mapping")
    p.add_argument("--no-balance-queries", action="store_true",
                   help="disable the 'N queries per category' constraint on the text side")
    p.add_argument("--balance-images", choices=["sinkhorn", "none"], default="sinkhorn")
    p.add_argument("--sinkhorn-temp", type=float, default=0.35)
    p.add_argument("--weight-zero-shot", type=float, default=0.30)
    p.add_argument("--weight-linear", type=float, default=0.40)
    p.add_argument("--weight-knn", type=float, default=0.0,
                   help="weight of the k-NN vote (validation search says 0.0)")
    p.add_argument("--prop-alpha", type=float, default=0.0,
                   help="gallery label-propagation strength (validation search says 0.0)")
    p.add_argument("--prop-k", type=int, default=15)
    p.add_argument("--head", nargs="*", default=None,
                   help="probe heads from train.py --mode probe (glob the fold files); "
                        "when given, they replace the sklearn linear probe")
    p.add_argument("--blend-query-sim", type=float, default=0.0,
                   help="weight of direct query-image similarity inside a category (report metrics)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default="auto",
                   help="autocast dtype for encoding (auto: fp16 CUDA, bf16 Apple silicon)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--use-validation-labels", action="store_true",
                   help="also train the image-side head on the labelled validation gallery "
                        "(only sensible for the test submission)")
    p.add_argument("--dump-queries", default=None, help="optional csv of the query->category decisions")
    p.add_argument("--json", action="store_true", help="print metrics as json")


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    return p.parse_args(argv)


def build_config(args) -> pipeline.PipelineConfig:
    image_cfg = imagemap.ImageMapConfig(
        weight_zero_shot=args.weight_zero_shot,
        weight_linear=args.weight_linear,
        weight_knn=args.weight_knn,
        prop_alpha=args.prop_alpha,
        prop_k=args.prop_k,
        balance=args.balance_images,
        sinkhorn_temp=args.sinkhorn_temp,
        head_paths=tuple(sorted(args.head)) if args.head else (),
    )
    return pipeline.PipelineConfig(
        backbones=tuple(args.backbone),
        views=tuple(args.views),
        backbone_weights=tuple(args.backbone_weight) if args.backbone_weight else None,
        weight_lexical=args.weight_lexical,
        balance_queries=not args.no_balance_queries,
        blend_query_sim=args.blend_query_sim,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        precision=args.precision,
        use_cache=not args.no_cache,
        add_validation_labels=args.use_validation_labels,
        image_cfg=image_cfg,
    )


def main() -> None:
    args = parse()
    cfg = build_config(args)
    root = config.resolve_data_dir(args.data)

    ground_truth = catalog.load_ground_truth(root)
    result = pipeline.run_pipeline(root, "validation", cfg, ground_truth=ground_truth)
    scores = metrics.evaluate_rankings(result.rankings, catalog.relevant_map(ground_truth))

    report = {"metrics": scores, "diagnostics": result.diagnostics}
    if args.json:
        print(json.dumps(report, indent=2, default=float))
    else:
        print("backbones :", ", ".join(cfg.backbones), "| views:", ",".join(cfg.views))
        print(
            f"R-precision {scores['r_precision']:.4f} | mAP@20 {scores['mAP@20']:.4f} | "
            f"P@5 {scores['P@5']:.4f} | Hit@1 {scores['Hit@1']:.4f} | "
            f"Hit@5 {scores['Hit@5']:.4f} | MRR@20 {scores['MRR@20']:.4f}"
        )
        diag = result.diagnostics
        print(f"queries -> right category : {diag.get('query_category_accuracy', float('nan')):.4f}")
        print(f"images  -> right category : top1 {diag.get('image_top1', float('nan')):.4f}")
        for name, acc in diag.get("image_category_accuracy", {}).items():
            print(f"    {name:<12} top1 {acc['top1']:.4f}  top5 {acc['top5']:.4f}")
        print(f"exact text matches: {diag['exact_text_match']:.3f} | ambiguous queries: {diag['ambiguous_queries']}")
        if scores["r_precision"] < diag.get("image_top1", 1.0):
            print(
                "note: R-precision is capped by the image->category accuracy; "
                "improve the image side before touching the text side."
            )

    if args.dump_queries:
        result.query_assignments.to_csv(args.dump_queries, index=False)
        print("query decisions ->", args.dump_queries)


if __name__ == "__main__":
    main()