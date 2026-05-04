# Data

This directory contains data accompanying the paper.

## Files

### `manual_annotations.csv`

Annotation schema with a small set of example rows illustrating how per-image
manual observation labels were structured (columns: `image_id`, `category`,
`notes`).

The full per-image labels for the ~316-image subset analyzed in §3.3 and §4.1
of the paper are not included in this repository. They are retained by the
author and may be released in future work pending appropriate ethical review
(see paper §8). The aggregate distribution used in the paper and in Figure 1
is provided in `category_counts.csv` below.

**Schema:**

| Column | Type | Description |
|---|---|---|
| `image_id` | string | Filename or identifier of the image (e.g., `nsfw_000123.png`) |
| `category` | string | One of: `Sexual`, `Violent`, `Polydactyly`, `Five-fingered`, `Oligodactyly`, `Atypical Arrangement`, `Misclassified`, `Other` |
| `notes` | string (optional) | Free-text observations |

**Annotation methodology** (paper §3.3): the observation is exploratory and
conducted by a single annotator; it does not include inter-rater reliability
or formal annotation protocol. Within the present study, manual observation
is positioned as preliminary description for qualitatively grasping the
filter's behavioral tendencies.

### `category_counts.csv`

Aggregate counts and percentages from the author's manual observation of the
~316-image subset. This file corresponds to Figure 1 in the paper and allows
direct reproduction of the category distribution without redistributing the
generated images or the full per-image annotation table.

**Schema:**

| Column | Type | Description |
|---|---|---|
| `category` | string | Category name |
| `count` | integer | Number of images in the category |
| `percentage` | float | Percentage of total |

`src/figures.py` reads this file by default to regenerate Figure 1; it also
accepts a per-image `manual_annotations.csv` if such a file is present.

## License

Both files are licensed under [CC-BY 4.0](../LICENSE-DATA).

## Categories

| Category | Description |
|---|---|
| `Sexual` | Image contains sexual content (paper §4.1: 0.3% of sample) |
| `Violent` | Image contains violent content (paper §4.1: 0.0% of sample) |
| `Polydactyly` | Hand depicted with more than five fingers (35.9%) |
| `Five-fingered` | Standard five-fingered hand with other anatomical irregularities (27.6%) |
| `Oligodactyly` | Hand depicted with fewer than five fingers (29.8%) |
| `Atypical Arrangement` | Atypical positioning, orientation, or arrangement of fingers (3.8%) |
| `Misclassified` | Image clearly does not depict a hand or anything plausibly NSFW (0.0%) |
| `Other` | Other generation artifacts (2.9%) |

The category labels in this dataset preserve the medical-anatomical terminology
used in the paper (polydactyly, oligodactyly). These terms describe physical
characteristics of human bodies; their use here is descriptive, not pathologizing.
The political point of the paper is precisely that such bodily variations should
not be filtered out of AI-generated visual fields as if they were "inappropriate."
