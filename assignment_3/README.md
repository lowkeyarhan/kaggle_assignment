# Text-to-Image Food Retrieval — Kaggle Solution

Rank 2,000 food photos for each of 303 dish descriptions. Scored by **R-precision**.
Generic pretrained vision-language models only; no training on these food categories from outside data.

## The key observation

For a query about dish *d*, the number of relevant gallery images is `R = |gallery ∩ d|` (19 or 20 on
test, exactly 10 on validation), and the submission contains exactly 20 ids. So

```
R-precision = |top-20 ∩ relevant| / R
```

is **1.0 whenever the 20 submitted ids are exactly the photos of the queried dish**, and the order
inside the top 20 changes nothing. The task therefore decomposes into two independent classification
problems plus one structural prior:

| Component | Question | Expected difficulty |
| --- | --- | --- |
| Text side | which dish does this sentence talk about? | easy — every query names its dish |
| Image side | which dish is this photo? | the real work |
| Structure | every dish owns ~19.8 gallery images; every dish gets 3 queries | free, if exploited |

Score ≈ `query_category_accuracy × image_top1`. Both are printed by `src/evaluate.py` so you always
know which side is losing points.

## Approach

**Text side** (`src/textmap.py`) — every query names its dish, so the primary signal is string
matching against `category_metadata.csv` (category name + the 33 categories that carry
`alternative_names`), scored with IDF-weighted token coverage plus a bonus for a contiguous phrase
hit. CLIP text-text similarity to the category metadata catches phrasings the matcher misses, and a
**capacity-constrained Hungarian assignment** (3 queries per category, known exactly) resolves the
rest globally — including cases where one dish name is a substring of another (`beef` inside
`beef carpaccio`).

**Image side** (`src/imagemap.py`) — five views fused, each with a different failure mode:

- **zero-shot**: cosine(image, prompt-ensembled category prototype) — no training, immune to the
  noisy training labels;
- **linear probe** (or the fold ensemble from `src/train.py`) on frozen features — adapts the space
  to these 101 dishes;
- **k-NN vote** from the 2000 training images — local, robust to label noise;
- **label propagation** over the gallery's own k-NN graph — the gallery clusters by dish, so
  isolated mistakes average out;
- **Sinkhorn / optimal transport** onto the 101 categories with uniform marginals, forcing each
  category to attract its ~19.8 images. This is the single biggest structural win.

**Ranking** (`src/retrieval.py`) — for each query, the 20 images of its predicted category, ordered
by image score. A direct query-image similarity term is available (`--blend-query-sim`) to improve
mAP@20/MRR@20 at a small R-precision cost.

## Repository Structure

```
src/           Pipeline (config, dataset, features, textmap, imagemap, balance, retrieval, metrics)
src/train.py   Optional training: prompt-initialised probe (default) or LoRA fine-tune
scripts/       Submission checker and a random-baseline format smoke test
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. prove the submission path works before touching a GPU
python3 scripts/random_baseline.py --out random_submission.csv
python3 scripts/check_submission.py random_submission.csv --data-dir match-the-dish-to-its-rizz

# 2. validate the real pipeline on the validation split
python3 src/evaluate.py --backbone siglip2-so400m-384

# 3. optional: train the probe heads, then use them
python3 src/train.py --mode probe --backbone siglip2-so400m-384 --folds 5 --epochs 20
python3 src/evaluate.py --backbone siglip2-so400m-384 --head models/probe/probe_fold*.npz

# 4. write the test submission with whatever validated best
python3 src/infer.py --backbone siglip2-so400m-384 --head models/probe/probe_fold*.npz \
    --out submission.csv
python3 scripts/check_submission.py submission.csv --data-dir match-the-dish-to-its-rizz
```

Embeddings are cached in `features/`, so re-running with different fusion weights costs seconds.

## Data

Download the competition data and place it at `assignment_3/match-the-dish-to-its-rizz/`
(or pass `--data`). Expected layout:

```
train.csv, train_queries.csv, category_metadata.csv
validation_gallery.csv, validation_queries.csv, validation_ground_truth.csv
test_gallery.csv, test_queries.csv, validate_submission.py
train_gallery/<image_id>.jpg, validation_gallery/..., test_gallery/...
```

See `dataset_description.md` for details and `kaggle_run.md` for a Kaggle notebook runbook.