# Dashcam Crash Detection — Kaggle Competition Solution

Binary classification of dashcam video clips (30 frames each) to predict whether a traffic crash occurs. Ranked by ROC-AUC on a hidden test set from a different camera/scene distribution.

## Approach

- **Backbone:** Frozen ResNet50 pretrained on ImageNet
- **Head:** Attention-based frame aggregation
- **Validation:** 5-fold GroupKFold (clips from the same source video share a `group_id`)
- **Inference:** Fold ensemble + TTA (4 crops x flip x 5 folds = 40 passes)
- **Expected val AUC:** ~0.86-0.88

## Repository Structure

```
src/           Training and inference code
scripts/       Submission validation
notebooks/     Kaggle notebooks (EDA, solution, submission)
```

## Quick Start

```bash
pip install -r requirements.txt
python3 src/train.py --data /path/to/data --out models --epochs 25
python3 src/infer.py --data /path/to/data --models models --out submission.csv
python3 scripts/check_submission.py --submission submission.csv --sample /path/to/data/sample_submission.csv
```

## Data

Download from the [Kaggle competition page](https://www.kaggle.com/competitions). Expected layout:

```
train/<clip_id>/frame_000.jpg ... frame_029.jpg
test/<clip_id>/frame_000.jpg ... frame_029.jpg
train_labels.csv
sample_submission.csv
```

See `dataset_description.md` for full details and `kaggle_run.md` for a step-by-step Kaggle notebook guide.
