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
- **Sinkhorn / optimal transport** onto the 101 categories with uniform marginals, forcing each
  category to attract its ~19.8 images. This is the single biggest structural win (+0.064).

Two further views (k-NN vote from the training images, gallery label propagation) are implemented and
reachable by flag, but validation says they *hurt* on this data, so they default to off.

**Ranking** (`src/retrieval.py`) — for each query, the 20 images of its predicted category, ordered
by image score. A direct query-image similarity term is available (`--blend-query-sim`) to improve
mAP@20/MRR@20 at a small R-precision cost.

## Repository Structure

```
src/           Pipeline (config, dataset, features, textmap, imagemap, balance, retrieval, metrics)
src/train.py   Optional training: prompt-initialised probe (default) or LoRA fine-tune
scripts/       Submission checker and a random-baseline format smoke test
```

## Results (validation split, 1,010 images / 202 queries)

| Backbone | image top-1 | R-precision | mAP@20 | P@5 |
| --- | --- | --- | --- | --- |
| `clip-b32-laion` (smoke test) | 0.803 | 0.806 | 0.840 | 0.945 |
| `clip-b32-laion` + 5-fold probe heads | 0.803 | 0.806 | 0.844 | 0.947 |
| `siglip2-b16-384` (tuned fusion) | 0.932 | **0.938** | 0.966 | 0.992 |
| `siglip2-b16-384` + `clip-b32-laion` (equal weights) | 0.912 | 0.911 | 0.946 | 0.988 |

Query -> category accuracy is **1.000** in every run (202/202), so the entire
score is the image side. Measured relationship on this data:

```
R-precision == image->category top-1  (exactly, for every configuration tried)
```

The backbone swap alone moved the score by **+0.13**; every transductive trick
combined was worth less than a tenth of that. Ensembling a much weaker model in
*hurts* (0.938 -> 0.911): weight the members, or do not ensemble.

### What moved the needle, and what did not

| Change | Effect on R-precision |
| --- | --- |
| Sinkhorn balancing of the gallery (structure prior) | **+0.064** (0.742 -> 0.806) |
| Query -> category via string matching | already 1.000, no gain left |
| k-NN vote from training images | **-0.02** (hurts) |
| Gallery label propagation | **-0.02** (hurts: smears confusable clusters) |
| Prototype rectification with train means | -0.03 (hurts) |
| Label propagation seeded from train labels | -0.10 (much worse) |
| EM self-training on the gallery | +0.00 (converges immediately) |

Conclusion: with a weak backbone every transductive trick saturates at ~0.81.
The remaining ~0.17 is **image recognition quality**, so the next moves are, in
order: a stronger backbone, fine-tuning on the labelled images (train + the
labelled validation gallery = 3,010 images), and an ensemble of two backbones
from independent pretraining corpora. `kaggle_run.md` has the exact commands and
the Kaggle notebook automates them.

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