# Production of Absence: Reading Stable Diffusion's NSFW Filter through Code and Typology

Code and data accompanying the paper:

> Kubota, Nozomu. 2026. "Production of Absence: Reading Stable Diffusion's NSFW Filter through Code and Typology." Accepted for publication in the proceedings of *Expanded 2026: Conference on Animation and Interactive Art*; forthcoming.

Paper DOI: [10.1145/3840416.3840454](https://doi.org/10.1145/3840416.3840454)

## Overview

This repository documents a historically specific Stable Diffusion v1.5
prompt–model–checker run. With the prompt `"hand sketch, pencil drawing"`, the
run stopped after exactly 42,158 generation attempts when 10,000 outputs had
been flagged by the checker (23.7203%; 32,158 outputs were unflagged).

The **methodological core** is a single line of code that replaces the
pipeline-internal checker with a pass-through function, preserving decoded
images for an audit view:

```python
pipe.safety_checker = dummy_safety_checker
```

See §5.1 of the paper for a critical reading of this intervention.

The empirical record is summarized as follows:

- A fixed-seed sample of 316 flagged outputs was drawn without replacement and
  coded by one annotator using mutually exclusive visual-description labels.
  The repository does not include that draw's seed or selected indices.
- One sampled image (0.3%) was coded as sexual content and none as violent.
  The other counts were: polydactyly-like 113 (35.8%), oligodactyly-like 94
  (29.7%), five-fingered with anatomical irregularities 87 (27.5%), atypical
  finger arrangement 12 (3.8%), and other generation artifacts 9 (2.8%).
- Unflagged outputs were not manually coded with the same schema. The study
  therefore does not estimate category-specific flag probabilities and does
  not establish that any anatomical category caused or was disproportionately
  likely to receive a flag.
- A separate CLIP ViT-B/32 model was used only for exploratory post hoc feature
  plots. Its 512-dimensional features are not the checker's internal decision
  features and do not identify a causal mechanism.

## Repository Structure

```
production-of-absence-nsfw/
├── README.md
├── LICENSE                     # MIT License (code)
├── LICENSE-DATA                # CC-BY 4.0 (annotation materials)
├── CITATION.cff                # Citation metadata
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── safety_bypass.py        # The dummy_safety_checker (§5.1)
│   ├── generate.py             # Image generation pipeline (§3.1, §3.2)
│   ├── extract_clip.py         # Auxiliary CLIP embedding extraction (§4.2–4.3)
│   ├── compose_grid.py         # Typological grid composition (§6.1, Figure 8)
│   └── figures.py              # Generate updated descriptive figures
└── data/
    ├── README.md
    ├── manual_annotations_example.csv  # Non-empirical schema examples
    └── category_counts.csv     # Aggregate counts used for Figure 1
```

## Methodology

The methodology corresponds to §3 of the paper:

### §3.1 NSFW Filter Architecture

Stable Diffusion's pipeline (`runwayml/stable-diffusion-v1-5`) integrates a
CLIP-based safety checker. The corresponding standard pipeline and this audit
workflow should be distinguished:

1. **Corresponding standard pipeline:** one internal `safety_checker` pass
   follows image decoding. A flagged output is replaced with a black image
   before delivery.
2. **Audit workflow used here:** the internal replacement is bypassed, and a
   standalone `StableDiffusionSafetyChecker` is applied once to the preserved
   decoded image so its classification can be recorded.

The audit position is not a second pass in ordinary use. It is an experimental
position constructed to inspect outputs that the corresponding standard
pipeline would be expected to replace.

The checker uses its own CLIP ViT-L/14-derived vision representation, fixed
concept and special-care embeddings, and threshold weights. This internal
representation is distinct from the separate OpenAI CLIP ViT-B/32 model used
after collection for the exploratory plots. See the
[CompVis model card](https://huggingface.co/CompVis/stable-diffusion-safety-checker)
and the
[Diffusers implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py).

### §3.2 Generation Parameters

| Parameter | Value |
|---|---|
| Generator model | `runwayml/stable-diffusion-v1-5` |
| Standalone checker | `CompVis/stable-diffusion-safety-checker` |
| Auxiliary analysis model | `openai/clip-vit-base-patch32` |
| Prompt | `"hand sketch, pencil drawing"` |
| `num_inference_steps` | 20 |
| `guidance_scale` | 7.5 |
| Resolution | 512 × 512 |
| Random-number generator | Initialized once with seed 42 and advanced across trials |
| Unflagged-retention RNG | Independent NumPy RNG, seed 42; unrelated to the 316-image manual draw |
| Historical stopping rule | Stop after collecting 10,000 checker-flagged outputs |
| Generation attempts | 42,158 |
| Checker-flagged outputs | 10,000 |
| Unflagged outputs | 32,158 |
| Observed flagged share | 23.7203% |

### §3.3 Manual Observation

A fixed-seed sample of exactly 316 checker-flagged images was drawn without
replacement and observed by the author. The seven mutually exclusive labels
were polydactyly-like, five-fingered with anatomical irregularities,
oligodactyly-like, atypical finger arrangement, other generation artifacts,
sexual content, and violent content. These are visual-description labels for
generated artifacts, not clinical diagnoses or documentary descriptions of
disability.

The observation is exploratory and was conducted by a single annotator. It has
no inter-rater reliability test or preserved prospective precedence rule for
ambiguous cases. Most importantly, no unflagged comparison sample was manually
coded with the same schema. The counts describe the flagged sample only; they
do not establish category-specific flag rates, bias, or causation.

The repository does not preserve the manual draw's seed or selected indices.
The `--safe_sample_seed` option in `src.generate` controls only the separate,
approximately 1% retention of unflagged images for auxiliary comparison.

## Reproducibility

The repository documents the workflow, identifiers, parameters, annotation
schema, aggregate counts, and figure scripts. It omits the full-resolution
images, full per-image labels, manual-draw seed and indices, historical
generation log, and embeddings.
Consequently, the archived paper plots that depend on those historical files
cannot be reconstructed exactly from this repository alone. The current plot
script uses within-group proportions and accessible styling for new runs; it
does not claim to recreate the paper's archived raw-count plots pixel for
pixel.

The 10,000-image corpus is **not redistributed** to reduce content-sensitive
and decontextualized reuse. This is an ethical dissemination decision, not a
claim that the model license grants the licensor ownership of generated
outputs.

The published scripts support workflow reconstruction, not exact replication
of the historical corpus. Model, checker, CLIP, Python-package, hardware, and
runtime revisions were not all pinned, so rerunning the same nominal settings
may produce different images, labels, timings, and aggregate counts.
Each generation run must use a new or empty `--output_dir`; the script rejects
a non-empty directory so that images from an earlier run cannot be silently
mixed with a newly written log.
The figure CLI likewise requires a new or empty output directory so omitted
inputs cannot leave stale plots from an earlier invocation in place.

### Reproduction Steps

```bash
# 1. Clone the repository
git clone https://github.com/nozomugit/production-of-absence-nsfw.git
cd production-of-absence-nsfw

# 2. Install dependencies (Python 3.10+ recommended; a CUDA GPU is strongly
#    recommended for a full run)
pip install -r requirements.txt

# 3a. Reconstruct the historical stopping rule.
#     This is not expected to recreate the historical corpus exactly; the
#     larger ceiling allows a changed environment to reach 10,000 flags.
python -m src.generate --target_flagged 10000 --max_attempts 100000 \
                       --seed 42 --safe_sample_seed 42 \
                       --output_dir output

# 3b. Optional strict check against the recorded historical attempt count.
#     This intentionally fails if flag 10,000 is not reached at attempt 42,158.
python -m src.generate --target_flagged 10000 --max_attempts 42158 \
                       --expected_attempts 42158 \
                       --seed 42 --safe_sample_seed 42 \
                       --output_dir output_strict

# 4. Extract auxiliary CLIP embeddings for checker-flagged and sampled
#    unflagged images (the directories retain their historical names)
python -m src.extract_clip --input_dir output/nsfw_images \
                            --output output/embeddings_nsfw.npz

python -m src.extract_clip --input_dir output/safe_sample \
                            --output output/embeddings_safe.npz

# 5. Generate updated descriptive figures (Figures 1–7 workflow).
#    Figure 1 uses the archived n=316 aggregate counts; Figures 2–7 use the
#    newly generated run supplied above. This is therefore not a single
#    historical-run reproduction. Comparison histograms use within-group
#    proportions because the groups differ in size.
python -m src.figures --records output/generation_records.json \
                      --embeddings_nsfw output/embeddings_nsfw.npz \
                      --embeddings_safe output/embeddings_safe.npz \
                      --annotations data/category_counts.csv \
                      --output_dir figures

# 6a. Compose the full 4 m × 1 m installation layout
python -m src.compose_grid --input_dir output/nsfw_images \
                            --output figures/figure8_typology_grid.jpg \
                            --rows 200 --columns 50 --no_grayscale

# 6b. Compose a 10 × 10 Figure 8-style excerpt. Without the withheld
#     historical corpus, this does not recreate the archived excerpt exactly.
python -m src.compose_grid --input_dir output/nsfw_images \
                            --output figures/figure8_excerpt.jpg \
                            --grid_size 10 --no_grayscale
```

### Hardware Notes

- A full 42,158-attempt run is computationally intensive; a CUDA-capable GPU is
  practically necessary. Runtime varies with hardware and software versions.
- For testing the methodology without the full generation, use
  `--num_attempts 100 --output_dir output_test` (the legacy name
  `--num_trials` remains an alias).
- The standalone `StableDiffusionSafetyChecker` and `dummy_safety_checker` bypass work on CPU as well, but generation will be impractically slow.

## License

This repository uses two licenses:

- **Source code** (everything in `src/`): [MIT License](LICENSE)
- **Annotation schema examples and aggregate manual-observation counts**
  (`data/manual_annotations_example.csv` and `data/category_counts.csv`):
  [CC-BY 4.0](LICENSE-DATA)

The [CreativeML OpenRAIL-M license](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE)
states that the licensor claims no rights in generated outputs while users
remain accountable for their use. As noted above, non-distribution of the
corpus here is based on content sensitivity and the risk of decontextualized
reuse, not an asserted output-ownership restriction.

## Citation

If you use this code, the annotation schema, or the aggregate manual-observation counts in your research, please cite:

```bibtex
@inproceedings{kubota2026production,
  author = {Kubota, Nozomu},
  title  = {Production of Absence: Reading Stable Diffusion's {NSFW} Filter through Code and Typology},
  booktitle = {Expanded 2026: Conference on Animation and Interactive Art},
  year   = {2026},
  doi    = {10.1145/3840416.3840454},
  note   = {Accepted for publication in the conference proceedings; forthcoming}
}
```

## Acknowledgments

This research was conducted at Creator's NEXT.

## Author

**Nozomu Kubota**
Creator's NEXT
nozomu.k@cnxt.jp
