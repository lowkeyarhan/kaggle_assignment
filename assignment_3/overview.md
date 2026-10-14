# Text-to-Image Food Retrieval — Competition Overview

## Problem Statement

You get a short description of a dish — "A picture of ravioli: Italian pasta parcels with a filling sealed between two layers of thin pasta dough." — and a gallery of 2,000 food photos. Your job: find every photo of that dish and rank them first.

This is text-to-image retrieval over 100 food categories. Each description refers to one dish; every gallery image of that dish counts as a correct answer (about 20 per query).

Use whatever approach you like: CNNs and transfer learning, learned joint text–image embeddings, contrastive training, pretrained vision–language models, re-ranking, ensembles. No specific model is required. What matters is how well you retrieve, how carefully you validate, and whether you can explain what works and what doesn't.

## Goal

For each test query, output your 20 best-matching image IDs from the test gallery, best first. Submissions are ranked by **R-precision** (see Evaluation).

## What You Get

- 2,000 labelled training images with example descriptions
- A validation gallery (1,010 images) with queries and answers, for model selection
- The test gallery (2,000 images) and 303 test descriptions, without answers
- A starter notebook that loads the data, scores itself on validation and writes a valid submission

## What You Submit

For each test query, your 20 best-matching image IDs, best first. See the Evaluation tab.

## What is Text-to-Image Retrieval?

Text-to-image retrieval means searching a collection of images with a sentence. You type "A photo of ramen" and the system returns the photos from its collection that best match, ordered from most to least relevant.

Under the hood, most retrieval systems turn both the text query `q` and every gallery image `x` into something comparable, then score each pair with a relevance function `s(q, x)` and sort the gallery by that score:

```
s(q, x)
```

That score can come from many places: the cosine similarity between a text embedding and an image embedding in a shared space,

```
cos(text_emb(q), image_emb(x))
```

or the probability that an image classifier assigns to the dish the query is about, or any mix of the two. This is the machinery behind photo search in phone galleries, product search in online shops, and the retrieval step of many multimodal assistants. Building a ranking that puts all the right photos first, given nothing but a short description, is the whole task in this competition.

## About the Dataset

The images are real-world food photos in food categories (dishes such as ramen, tiramisu, paella or gyoza), taken by many different people with different cameras, lighting and plating. Every image belongs to exactly one category.

The data comes in three parts:

- **Training set:** 2,000 images (about 20 per category) with their category labels, plus 5 example text descriptions per category.
- **Validation set:** a separate gallery of 1,010 images (10 per category), 202 queries, and their answers. Use it to compare your approaches.
- **Test set:** a gallery of 2,000 images and 303 text queries (3 per category). The answers are hidden.

Each query describes one dish, for example:

> A picture of ravioli: Italian pasta parcels with a filling sealed between two layers of thin pasta dough.
>
> An image that contains red velvet cake.
>
> Show me pictures of potstickers.

Descriptions are about the dish, never a specific photo. Every query names its dish, sometimes through an alternative name (potstickers for gyoza, burger for hamburger), and `category_metadata.csv` lists these names. Test queries use sentence wordings that never appear in the training or validation queries.

Relevance is defined at the category level: for a query about ramen, every ramen photo in the test gallery is a correct answer. Each test query therefore has `R = 19` or `20` correct images among the 2,000 (validation queries have `R = 10`).

All image, query and category IDs (`img_…`, `q_…`, `c_…`) are random. They carry no information about the content, so don't read anything into them. No image appears in more than one split, and exact and near-duplicate photos across splits were removed. Several categories are deliberately hard to tell apart (different pasta dishes, cakes, rice dishes, noodle soups, cuts of beef), so expect the interesting mistakes to happen there.

## Evaluation Metric, Step by Step

We do not just check whether one correct photo shows up near the top. With about 20 correct photos per query, finding one is easy. We measure how many of the correct photos your ranking puts first.

For each test query, let `R` be the number of gallery images of that dish (19 or 20). Look at the first `R` images of your submitted ranking and count how many of them are correct:

```
R-precision(q) = |{ top-R predictions of q } ∩ relevant(q)| / R
```

Then average over all `N` test queries for your final score:

```
score = (1 / N) * Σ_q R-precision(q)
```

Higher is better. The score ranges from 0 to 1:

- **1.0:** for every query, your first `R` images are exactly the `R` photos of that dish.
- **0.5:** on average, half of your first `R` images are correct.
- **About 0.01:** what a random ranking scores.

A worked example: a tiramisu query has `R = 20` tiramisu photos in the gallery. Your first 20 predictions contain 14 tiramisu photos and 6 others, so this query scores `14 / 20 = 0.70`. Positions 21 and beyond would not count. The order within the first `R` does not change R-precision either.

The public leaderboard is computed on about 50% of the test queries. The final ranking uses the other 50% and is revealed after the deadline. Choose your model on the validation set: chasing the public leaderboard tends to overfit it.

### Secondary Metrics (for your report, not the leaderboard)

The starter notebook also computes, on the validation set:

- **mAP@20:** average precision over your 20 predictions. Unlike R-precision, it rewards putting correct images earlier.
- **Precision@5:** the fraction of your first 5 images that are correct.
- **Hit@1 and Hit@5:** whether a correct image appears at rank 1, or anywhere in the top 5.
- **MRR@20:** the mean of `1 / (rank of your first correct image)`.

Report them alongside R-precision. They show how your ranking succeeds or fails.

## Submission Format

Submit a single `submission.csv` with exactly these columns:

```
query_id,image_ids
```

`image_ids` holds exactly 20 distinct image IDs from `test_gallery.csv`, separated by single spaces, best match first:

```
query_id,image_ids
q_006e83c1,img_aa14d78b57 img_10600424b5 img_a7bc05b34d ... (20 ids in total)
q_034492a4,img_71736b0896 img_d88d5f08f6 img_0607944642 ... (20 ids in total)
```

Include one row for every `query_id` in `test_queries.csv` (303 rows), with no duplicates and no missing rows. Run `python validate_submission.py submission.csv --data-dir <data folder>` before submitting. It checks every rule above without needing the answers.

## Allowed Methods

Any approach that produces a ranking counts, including but not limited to:

- **Classification-based retrieval:** map the query to its dish (e.g. by matching the dish name), then rank gallery images by a CNN classifier's confidence for that dish, using transfer learning from ImageNet-pretrained backbones (ResNet, EfficientNet, ConvNeXt, ViT).
- **Embedding-based retrieval:** learn a joint text–image embedding with contrastive training on the training images and descriptions, or use pretrained vision–language models, then rank by similarity.
- **Re-ranking and ensembles:** combine classifier scores and embedding similarities, use query expansion or nearest-neighbour re-ranking on the gallery, or test-time augmentation.
- **Any hybrid of the above.**

There is no restriction on architecture beyond producing a valid `submission.csv` under the Competition Rules. In particular, you may not use external copies of the competition images or labels, match test images against outside image collections, or use checkpoints trained to classify these specific food categories. Generic pretrained models are fine, as long as you name them in your report.

## Starter Notebook

The starter notebook loads all files, implements the metric exactly as above for the validation set, and builds a simple baseline: an ImageNet ResNet-50 with a linear classifier trained on the 2,000 training images, plus dish-name matching for the text. It scores an R-precision of about 0.5 on the leaderboard. It then writes a valid submission. Treat it as a starting point, not a template you have to follow.

## Evaluation (Summary)

Submissions are scored by R-precision, averaged over all test queries. For a query whose dish has `R` images in the test gallery (`R = 19` or `20`):

```
R-precision = (number of correct images among your first R predictions) / R
```

A score of 1.0 means your first `R` images are exactly the `R` photos of that dish. Images at positions `R+1` to 20 do not change this score.

The leaderboard shows R-precision only. The starter notebook also computes mAP@20, Precision@5, Hit@1, Hit@5 and MRR@20 on the validation set; report these in your write-up.

The public leaderboard uses about 50% of the test queries; the final ranking uses the other 50%. Select your model on the validation set, not on the public leaderboard.