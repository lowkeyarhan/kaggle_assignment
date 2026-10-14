# Dashcam Crash Detection — Competition Overview

## Problem Statement

Can you tell, from a few seconds of dashcam footage, whether a crash happened?

Systems that watch the road (dashcams, fleet-safety tools, driver-assistance software) need to recognize a real crash quickly and reliably, on roads, in weather and with cameras they were never tuned for.

In this competition you are given short front-facing dashcam clips and must predict the probability that a crash occurs in each clip. You train on one collection of clips and are scored on clips from a different collection: different cameras, regions, weather and accident types. Models that only memorize the training footage will not survive the move.

## Goal

For every test clip, output a crash probability between 0 and 1. Submissions are ranked by **ROC-AUC**.

## What Makes It Hard

- Train and test come from different sources, so the leaderboard rewards generalization, not fitting.
- The share of crash clips differs between train and test.
- "No crash" looks different in the two collections, so a model that just detects "danger-looking" scenes will get fooled.

## The Task

Each sample is a clip of 30 consecutive dashcam frames. Your job is binary: does a traffic crash occur somewhere inside this clip? You predict a probability for every test clip.

## The Data

All clips, in both train and test, share one format so that you never have to guess about preprocessing:

- 30 frames per clip, in chronological order, named `frame_000.jpg` … `frame_029.jpg`
- JPEG, 384 x 224 pixels, identical for train and test
- Clip IDs are random and carry no information, and the rows of every CSV are shuffled

| Path                                              | Contents                 |
| ------------------------------------------------- | ------------------------ |
| `train/<clip_id>/frame_000.jpg ... frame_029.jpg` | 3,000 labeled clips      |
| `test/<clip_id>/frame_000.jpg ... frame_029.jpg`  | 1,750 unlabeled clips    |
| `train_labels.csv`                                | `clip_id,label,group_id` |
| `sample_submission.csv`                           | `clip_id,label`          |
| `clip_spec.yaml`                                  | the clip format above    |

## Labels

`label` is 1 if a crash occurs inside the clip, 0 otherwise. The training set is exactly balanced: 1,500 crash clips and 1,500 no-crash clips.

## Use `group_id` for Validation

Training clips come in pairs cut from the same source video: a no-crash clip and a crash clip whose frames partly overlap in time. Both carry the same `group_id`. Build your validation folds with grouped splits (for example `GroupKFold`). If you split randomly, a clip and its overlapping twin land in different folds, and your validation score will be far better than your leaderboard score.

## Train and Test Are Different on Purpose

The test clips come from a different collection than the training clips. Expect these shifts:

- **Cameras and places:** different dashcams, roads, regions, lighting and weather.
- **Class balance:** about half of the training clips contain a crash; the test set has a lower share, closer to one in three. This is why the metric is ROC-AUC, which does not depend on a decision threshold or on the class ratio.
- **What "no crash" means:** the training no-crash clips are the seconds before a crash in videos that later crash. The test no-crash clips are ordinary driving with no crash in the video.
- **Image quality:** clips were resized, cropped of black borders, and re-encoded to a common format, but differences in compression and camera overlays (for example a date/time stamp) can remain.

Treat the leaderboard gap between your validation score and your test score as information about how well your model generalizes.

## Rules of Thumb

- Pretrained image or video backbones are allowed (see the Rules for exactly what is and is not allowed).
- The test clips are unlabeled. Labeling them by hand, or trying to find the original datasets they came from, is not allowed, and the Rules say why.
- Split your validation by `group_id`, and prefer methods that are robust to camera and scene changes over methods that fit the training scenes.

## Evaluation

Submissions are evaluated on the area under the ROC curve between the predicted probability and the observed label (1 = a crash occurs in the clip, 0 = no crash).

ROC-AUC only depends on how your predictions rank the clips, not on a threshold, so it is not affected by the different crash rates in train and test. A perfect ranking scores 1.0 and random guessing scores 0.5.

### Public and Private Leaderboard

The test clips are split at random into two fixed parts. The public leaderboard is computed on 30% of them and is visible while the competition runs. The private leaderboard is computed on the remaining 70% and is revealed only after the competition ends. Your final rank uses the private score of your selected Final Submissions (see the Rules).

## Submission File

For each `clip_id` in the test set, predict the probability that the clip contains a crash. The file must contain a header, exactly the two columns below, and one row for every clip listed in `sample_submission.csv`:

```
clip_id,label
c100001,0.93
c100002,0.04
c100003,0.51
etc.
```

- `label` must be numeric and between 0 and 1 (a probability; raw scores work for ranking, but keep them in range).
- Do not repeat a `clip_id`, add extra columns, or leave a value empty. Malformed files are rejected.

You can check your file before submitting with the validator in the project repository:

```bash
python scripts/check_submission.py --submission submission.csv --sample sample_submission.csv
```

## Final Marks Formula Used for Grading

Let:

- **S** = Student's ROC-AUC score
- **T** = Average ROC-AUC score of the top 10% participants
- **B** = 0.5 = Random performance baseline
- Maximum marks = 100

Where the result is capped between 0 and 100.

### Example: If the top 10% average ROC-AUC is 0.90

| Student ROC-AUC | Marks (square root scale) |
| --------------- | ------------------------- |
| 0.50            | 0                         |
| 0.60            | 58                        |
| 0.65            | 67                        |
| 0.70            | 75                        |
| 0.75            | 82                        |
| 0.80            | 89                        |
| 0.85            | 94                        |
| 0.90            | 100                       |

This formula rewards improvements over random performance.
