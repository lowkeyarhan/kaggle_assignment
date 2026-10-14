# Kaggle Run Guide

## Setup (new Kaggle notebook, GPU on)

1. Add this repo (or upload `src/`) + the competition data as notebook inputs.
2. Internet ON (one-time backbone weight download, cached after).

```bash
pip install -q -r requirements.txt
```

## Train (5-fold GroupKFold, frozen ResNet50 + attention head)

```bash
python3 src/train.py --data /kaggle/input/<data-path> --out models --epochs 25
```

Expect ~0.86–0.88 mean fold AUC. If a fold stalls below 0.80, it overfit —
lower `--lr` to 1e-4 or raise dropout via code default.

## Infer (fold ensemble + TTA → submission.csv)

```bash
python3 src/infer.py --data /kaggle/input/<data-path> --models models --out submission.csv
python3 scripts/check_submission.py --submission submission.csv \
  --sample /kaggle/input/<data-path>/sample_submission.csv
```

Upload `submission.csv`. TTA (4 crops × flip × 5 folds = 40 passes) is where
the last +0.02 AUC comes from — don't skip it before concluding the model
is weak.

## If public LB < val by >0.05

That gap is domain shift, not noise: freeze more (already frozen), strengthen
aug (ColorJitter/blur), or add backbone B (train a second run with
`--backbone resnet18`) and average the two submissions 50/50.
