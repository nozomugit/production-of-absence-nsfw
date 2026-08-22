# Data

This directory contains data accompanying the paper.

## Files

### `manual_annotations_example.csv`

Annotation schema with a small set of example rows illustrating how per-image
manual observation labels were structured (columns: `image_id`, `category`,
`notes`). These rows are illustrative, not empirical observations from the
316-image analyzed subset, and must not be used to reproduce Figure 1.

The full per-image labels for the 316-image subset analyzed in §3.3 and §4.1
of the paper are not included in this repository. The aggregate distribution
used in the paper and in Figure 1 is provided in `category_counts.csv` below.

**Schema:**

| Column | Type | Description |
|---|---|---|
| `image_id` | string | Filename or identifier of the image (e.g., `nsfw_000123.png`) |
| `category` | string | One of: `Sexual content`, `Violent content`, `Polydactyly-like`, `Five-fingered with anatomical irregularities`, `Oligodactyly-like`, `Atypical finger arrangement`, `Other generation artifacts` |
| `notes` | string (optional) | Free-text observations |

**Annotation methodology** (paper §3.3): exactly 316 checker-flagged outputs
were drawn without replacement in a fixed-seed sample. One annotator assigned
one of seven mutually exclusive visual-description labels to each image. The
study has no inter-rater reliability test and no preserved prospective
precedence rule for ambiguous cases. The repository does not include the
manual draw's seed or selected indices; `src.generate --safe_sample_seed`
belongs to the separate unflagged-image retention step and is not that draw.

Unflagged outputs were not manually coded with the same schema. These
annotations therefore describe only the flagged sample; they cannot estimate
category-specific flag probabilities, demonstrate disproportionate filtering,
or identify a causal checker feature.

### `category_counts.csv`

Aggregate counts and percentages from the author's manual observation of the
316-image subset. This file corresponds to Figure 1 in the paper and allows
direct reproduction of the category distribution without redistributing the
generated images or the full per-image annotation table.

**Schema:**

| Column | Type | Description |
|---|---|---|
| `category` | string | Category name |
| `count` | integer | Number of images in the category |
| `percentage` | float | Percentage of total |

`src/figures.py` reads this file by default to regenerate Figure 1; it also
accepts a full per-image annotation CSV supplied by the user. The bundled
`manual_annotations_example.csv` is schema documentation only.

## License

Both files are licensed under [CC-BY 4.0](../LICENSE-DATA).

## Categories

| Category | Description |
|---|---|
| `Sexual content` | Coded as sexual content: 1 of 316 (0.3%) |
| `Violent content` | Coded as violent content: 0 of 316 (0.0%) |
| `Polydactyly-like` | Polydactyly-like generated hand form: 113 of 316 (35.8%) |
| `Five-fingered with anatomical irregularities` | Five visible digits with other irregularities: 87 of 316 (27.5%) |
| `Oligodactyly-like` | Oligodactyly-like generated hand form: 94 of 316 (29.7%) |
| `Atypical finger arrangement` | Atypical positioning, orientation, or arrangement: 12 of 316 (3.8%) |
| `Other generation artifacts` | Other generation artifacts: 9 of 316 (2.8%) |

Percentages are rounded to one decimal place and therefore sum to 99.9%.
The qualified labels *polydactyly-like* and *oligodactyly-like* are visual
descriptions of generated artifacts, not clinical diagnoses or documentary
descriptions of disability.
