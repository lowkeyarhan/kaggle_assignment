# Kaggle Run Guide

## Setup (new Kaggle notebook, GPU on)

1. Add the competition data as a notebook input (it appears under `/kaggle/input/<slug>/`; the
   pipeline auto-detects the folder that contains `train.csv`).
2. Internet ON for the one-time backbone download, cached afterwards.

```bash
pip install -q -r requirements.txt
```

Everything below runs from the notebook working directory with the repo (or just `src/`, `scripts/`)
uploaded as a second input.

## Validate first, submit second

```bash
python3 src/evaluate.py --backbone siglip2-so400m-384
```

Read the output top to bottom:

- `queries -> right category` — the text side. Should be ~1.0. If it is not, inspect the ambiguous
  queries printed by `--dump-queries query_decisions.csv`; a single wrong category costs that query
  its entire score.
- `images -> right category` — the image side. This is the number to optimise, and it is bounded by
  the gallery accuracy of whatever backbone you used.
- `R-precision` vs the image top-1 — R-precision cannot exceed image top-1 by much, so that gap
  tells you immediately whether the text side is still costing you.

## Iterate on the image side

```bash
# stronger backbone
python3 src/evaluate.py --backbone dfn5b-h14-378

# ensemble two independent pretraining corpora (concatenated embeddings)
python3 src/evaluate.py --backbone dfn5b-h14-378 siglip2-gopt-384

# trained probe heads (5-fold, prompt-initialised)
python3 src/train.py --mode probe --backbone siglip2-so400m-384 --folds 5 --epochs 20
python3 src/evaluate.py --backbone siglip2-so400m-384 --head models/probe/probe_fold*.npz

# only if the probe is clearly stuck: low-rank fine-tuning of the vision tower
python3 src/train.py --mode lora --backbone siglip2-so400m-384 --lr 1e-4 --epochs 10 \
    --lora-rank 16 --lora-alpha 32
```

Ablations worth running once for the report (each is a single flag):

| Flag | Question it answers |
| --- | --- |
| `--balance-images none` | how much does Sinkhorn balancing buy? |
| `--no-balance-queries` | how much does the 3-queries-per-category constraint buy? |
| `--weight-lexical 1.0` | how far does CLIP text similarity matter vs. string matching? |
| `--views center` | how much do the extra TTA views buy? |
| `--prop-alpha 0` | how much does gallery label propagation buy? |

## Submit

```bash
python3 src/infer.py --backbone siglip2-so400m-384 --head models/probe/probe_fold*.npz \
    --out submission.csv
python3 scripts/check_submission.py submission.csv --data-dir /kaggle/input/<data-slug>
```

Upload `submission.csv`. Validate on the validation split, not the public leaderboard: the public
board is 50% of the test queries and a single bad category decision there is worth chasing noise.

## If validation R-precision stalls below the image top-1

- The gallery is 19/20 images per dish: a single image pushed into a neighbouring category's top-20
  costs `1/20` of one query. Look at `imagemap.classification_accuracy` for `propagated` vs
  `fused_raw` and make sure label propagation is not smearing the confusable clusters together.
- Several dishes are deliberately near-identical (different pasta, cakes, rice dishes, noodle soups,
  cuts of beef). Add a second backbone from a different pretraining corpus; disagreement between
  LAION-trained and WebLI-trained models is exactly where those categories live.
- Finally, `--balance-images none` tells you whether the balancing prior is now *hurting* (it will,
  once the features are good enough that argmax is already balanced).