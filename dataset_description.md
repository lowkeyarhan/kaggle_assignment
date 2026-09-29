# Dataset Description

Every sample is a clip: 30 consecutive dashcam frames stored as 30 JPEG images in one folder. You predict, for each test clip, the probability that a crash occurs somewhere inside it.

## Files

| File / Directory        | Description                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------- |
| `train/`                | 3,000 labeled training clips. One sub-folder per clip, named by its `clip_id`.     |
| `test/`                 | 1,750 unlabeled test clips, in the same layout as `train/`.                        |
| `train_labels.csv`      | The label of every training clip, plus a `group_id` used for validation splits.    |
| `sample_submission.csv` | A sample submission file in the correct format (every clip predicted as 0.5).      |
| `clip_spec.yaml`        | The shared clip format (number of frames, image size, JPEG quality) for reference. |

### Clip folder layout

```
train/<clip_id>/frame_000.jpg ... frame_029.jpg
test/<clip_id>/frame_000.jpg  ... frame_029.jpg
```

## Clip Format

- Each clip folder contains exactly 30 images, `frame_000.jpg` to `frame_029.jpg`.
- Frames are in chronological order: `frame_000.jpg` is the earliest, `frame_029.jpg` the latest.
- Every image is a color JPEG of **384 x 224 pixels** (width x height). Train and test use the identical format, so no resizing is needed unless your model requires it.
- The clips are front-facing dashcam views.
- Clip IDs are random strings such as `c123456`. They carry no information, and the rows of the CSV files are shuffled.

## Columns

### `train_labels.csv`

| Column     | Description                                                                                                                                                                                                                                                   |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clip_id`  | The clip's identifier, and the name of its folder inside `train/`.                                                                                                                                                                                            |
| `label`    | 1 if a crash occurs inside the clip, 0 if not. This is what you predict for the test clips.                                                                                                                                                                   |
| `group_id` | Clips that were cut from the same underlying video share a `group_id`. Such clips can overlap in time, so keep all clips with the same `group_id` in the same validation fold (for example with `GroupKFold`). This column does not exist for the test clips. |

### `sample_submission.csv` and your submission file

| Column    | Description                                                                  |
| --------- | ---------------------------------------------------------------------------- |
| `clip_id` | The identifier of a test clip (a folder name inside `test/`).                |
| `label`   | Your predicted probability, between 0 and 1, that the clip contains a crash. |

The submission must contain one row for every clip listed in `sample_submission.csv`, with exactly these two columns. See the Evaluation page for the full rules.

## What Am I Predicting?

For each test clip, the probability that a crash occurs somewhere within its 30 frames.

- **Training set:** 3,000 clips (1,500 labeled 1, 1,500 labeled 0).
- **Test set:** 1,750 clips, no labels. The share of crash clips in the test set is lower than in the training set.
- The test clips come from different cameras, places and conditions than the training clips.

## Acronyms and Terms

| Term                         | Definition                                                                                                                                                      |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Clip                         | One 30-frame sample, stored as a folder of 30 images.                                                                                                           |
| Crash                        | A traffic collision that happens inside the clip (label 1).                                                                                                     |
| ROC-AUC                      | Area under the receiver operating characteristic curve, the competition metric. It measures how well your probabilities rank crash clips above non-crash clips. |
| JPEG                         | The image file format of every frame.                                                                                                                           |
| Public / private leaderboard | The two fixed parts of the test set used for scoring (30% / 70%). See the Evaluation page.                                                                      |
