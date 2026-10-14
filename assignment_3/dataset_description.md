# Text-to-Image Food Retrieval — Dataset Description

Images are real-world food photos, re-encoded as JPEG (max 384 px) and renamed with random IDs. Text descriptions were generated from the category labels with templates — they describe the dish, never a particular photo. Every description names its dish (sometimes via an alternative name, e.g. potstickers for gyoza).

Use of the images is limited to this course (see `DATA_TERMS.txt`).

## Files

| File | Description |
| ---- | ----------- |
| `train.csv` | `image_id, category_id, category_name` — 2,000 training images (~20 per category) |
| `train_gallery/` | training images, `<image_id>.jpg` |
| `train_queries.csv` | `query_id, description, category_id, category_name` — 5 example descriptions per category |
| `category_metadata.csv` | `category_id, category_name, description, alternative_names, train_paraphrases` |
| `validation_gallery.csv`, `validation_gallery/` | 1,010 validation images (10 per category) |
| `validation_queries.csv` | `query_id, description` — 2 per category |
| `validation_ground_truth.csv` | `query_id, category_id, relevant_image_ids` (space-separated) — use for model selection |
| `test_gallery.csv`, `test_gallery/` | the 2,000 test images to rank |
| `test_queries.csv` | `query_id, description` — 303 test queries (3 per category), no answers |
| `sample_submission.csv` | a valid (random) submission in the required format |
| `validate_submission.py` | format checker for your submission |
| `DATA_TERMS.txt` | terms of use for the images |

## Notes

- No image appears in more than one split; exact and near-duplicate images across splits were removed.
- Training labels contain some noise; test labels were manually checked.
- Validation queries have `R = 10` relevant images, test queries `R = 19–20`, so validation and test scores are not directly comparable.

## Dataset Summary

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

## Data Totals

- **Files:** 5,021
- **Size:** 185.9 MB
- **Type:** jpg, csv, py + 1 other
- **License:** Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

## Data Terms of Use (`DATA_TERMS.txt`)

The images in this competition were collected from a public food-photo sharing website and remain the property of their original photographers. They are provided solely for this course, for teaching and scientific fair use.

- Use the data only for this competition and course.
- Do not redistribute, publish or share the images or labels outside the competition.
- Any use beyond scientific fair use requires permission from the respective picture owners.