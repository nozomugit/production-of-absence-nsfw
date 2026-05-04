# Data

This directory contains the manual annotation data accompanying the paper.

## Files

### `manual_annotations.csv`

Per-image manual observation labels for the ~316 NSFW-flagged hand sketch
images analyzed in §3.3 and §4.1 of the paper.

>Note: This file documents the annotation schema with a small set of example rows, illustrating how per-image labels would be structured. The aggregate distribution used in the paper (and in Figure 1) is provided in category_counts.csv. Per-image labels for the full ~316-image subset are retained by the author and may be released in future work pending appropriate ethical review (see paper §8).

**Schema:**

| Column | Type | Description |
|---|---|---|
| `image_id` | string | Filename or identifier of the image (e.g., `nsfw_000123.png`) |
| `category` | string | One of: `Sexual`, `Violent`, `Polydactyly`, `Five-fingered`, `Oligodactyly`, `Atypical Arrangement`, `Misclassified`, `Other` |
| `notes` | string (optional) | Free-text observations |

**Annotation methodology** (§3.3): "exploratory and conducted by a single
annotator; it does not include inter-rater reliability or formal annotation
protocol. Within the present study, manual observation is positioned as
preliminary description for qualitatively grasping the filter's behavioral
tendencies."

### `category_counts.csv`

Aggregate counts and percentages of the manual annotation categories,
corresponding to the data shown in Figure 1.

**Schema:**

| Column | Type | Description |
|---|---|---|
| `category` | string | Category name |
| `count` | integer | Number of images in the category |
| `percentage` | float | Percentage of total |

These aggregate counts are derived from `manual_annotations.csv`. Both files
are provided for convenience: the per-image labels enable detailed analysis,
while the aggregate counts allow direct reproduction of Figure 1 without
processing the full label file.

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
