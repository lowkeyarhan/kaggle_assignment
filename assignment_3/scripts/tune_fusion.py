"""Tune the image-side fusion on the validation split, never on the leaderboard.

    python3 scripts/tune_fusion.py --backbone siglip2-b16-384

Computes the score components once (zero-shot, linear probe, kNN, gallery label
propagation), then grid-searches the fusion weights, the propagation strength
and the Sinkhorn temperature against validation R-precision. Because every
component is derived from cached embeddings, a few hundred configurations take
seconds.

Writes the winning flags, ready to paste into ``src/infer.py``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import balance  # noqa: E402
import catalog  # noqa: E402
import config  # noqa: E402
import features  # noqa: E402
import imagemap  # noqa: E402
import metrics  # noqa: E402
import pipeline  # noqa: E402
import retrieval  # noqa: E402
import textmap  # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=None)
    p.add_argument("--backbone", nargs="+", default=[config.DEFAULT_BACKBONE])
    p.add_argument("--views", nargs="+", default=list(config.DEFAULT_VIEWS))
    p.add_argument("--top", type=int, default=10, help="how many configs to print")
    p.add_argument("--out", default="results/fusion_search.json")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse()
    root = config.resolve_data_dir(args.data)

    cats = catalog.load_categories(root)
    gallery = catalog.load_images(root, "validation")
    queries = catalog.load_queries(root, "validation")
    train = catalog.load_images(root, "train")
    gt = catalog.load_ground_truth(root)
    relevant = catalog.relevant_map(gt)

    cfg = pipeline.PipelineConfig(
        backbones=tuple(args.backbone),
        views=tuple(args.views),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    gallery_emb, train_emb, query_emb, proto, cat_text = pipeline._encode_roles(
        cfg, cats, gallery, queries, train
    )

    # Text side is already exact, so it is fixed rather than searched.
    assignments, _ = textmap.resolve(
        queries, cats, clip_scores=query_emb @ cat_text.T, balanced=True
    )
    image_ids = gallery["image_id"].tolist()

    n_classes = len(cats)
    labels = imagemap.labels_from_frame(train, cats["category_id"].tolist())
    print("computing score components ...")
    zs = imagemap.zero_shot_scores(gallery_emb, proto)
    lin = imagemap.linear_scores(gallery_emb, train_emb, labels, n_classes)
    knn = imagemap.knn_scores(gallery_emb, train_emb, labels, n_classes, k=25, temp=0.05)
    graph = imagemap._knn_graph(gallery_emb, k=15, temp=0.07)

    image_labels = catalog.validation_image_labels(gallery, gt, cats)
    known = image_labels >= 0

    def score_config(w_zs, w_lin, w_knn, alpha, temp):
        fused = imagemap.fuse([zs, lin, knn], [w_zs, w_lin, w_knn])
        prop = (1 - alpha) * fused + alpha * (graph @ fused)
        balanced = balance.sinkhorn_log_plan(prop, temp=temp, n_iter=100)
        rankings = retrieval.rank_queries(assignments, balanced, image_ids, top_k=20)
        result = metrics.evaluate_rankings(rankings, relevant)
        acc = imagemap.classification_accuracy(balanced[known], image_labels[known])
        return result["r_precision"], result["mAP@20"], acc["top1"]

    grid = {
        "w_zs": [0.0, 0.2, 0.3, 0.4],
        "w_lin": [0.2, 0.4, 0.6],
        "w_knn": [0.0, 0.2, 0.4],
        "alpha": [0.0, 0.3, 0.5, 0.7],
        "temp": [0.15, 0.35, 0.6],
    }
    keys = list(grid)
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg_values = dict(zip(keys, combo))
        if cfg_values["w_knn"] == 0 and cfg_values["w_lin"] == 0:
            continue  # nothing but zero-shot would be a silly configuration
        r, m, top1 = score_config(**cfg_values)
        results.append(
            {
                "w_zero_shot": cfg_values["w_zs"],
                "w_linear": cfg_values["w_lin"],
                "w_knn": cfg_values["w_knn"],
                "prop_alpha": cfg_values["alpha"],
                "sinkhorn_temp": cfg_values["temp"],
                "r_precision": r,
                "mAP@20": m,
                "image_top1": top1,
            }
        )
        print(
            f"zs={cfg_values['w_zs']:.1f} lin={cfg_values['w_lin']:.1f} "
            f"knn={cfg_values['w_knn']:.1f} a={cfg_values['alpha']:.1f} "
            f"T={cfg_values['temp']:.2f} -> R {r:.4f} (img top1 {top1:.4f})",
            flush=True,
        )

    results.sort(key=lambda row: -row["r_precision"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Save the whole grid, not just the winners: the spread across the grid is
    # what tells you whether a setting is a real effect or validation noise.
    out.write_text(json.dumps({"backbone": args.backbone, "results": results}, indent=2))

    print(f"\ntop {args.top} configurations by validation R-precision:")
    for row in results[: args.top]:
        print(
            f"  R {row['r_precision']:.4f} mAP@20 {row['mAP@20']:.4f} | "
            f"--weight-zero-shot {row['w_zero_shot']} --weight-linear {row['w_linear']} "
            f"--weight-knn {row['w_knn']} --prop-alpha {row['prop_alpha']} "
            f"--sinkhorn-temp {row['sinkhorn_temp']}"
        )
    print(f"\nfull results -> {out}")


if __name__ == "__main__":
    main()