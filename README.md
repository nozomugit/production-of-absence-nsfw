# Production of Absence: Reading Stable Diffusion's NSFW Filter through Code and Typology

Code and data accompanying the paper:

> Kubota, Nozomu. 2026. "Production of Absence: Reading Stable Diffusion's NSFW Filter through Code and Typology." *Expanded Conference 2026*.

## Overview

This research investigates how Stable Diffusion's NSFW filter systematically removes hand sketches containing anatomical variations—polydactyly, oligodactyly, and atypical finger arrangements—rather than the sexual or violent content it ostensibly targets.

The **methodological core** is a single line of code that bypasses the pipeline-internal first-stage NSFW filter, allowing observation of images that would otherwise be invisible to ordinary users:

```python
pipe.safety_checker = dummy_safety_checker
```

See §5.1 of the paper for a critical reading of this intervention.

The empirical findings are summarized as follows:

- Over 42,000 generation trials with the prompt `"hand sketch, pencil drawing"` produced approximately 10,000 NSFW-flagged images.
- The NSFW classification rate converged to approximately 23.7%.
- Manual observation of approximately 316 NSFW-flagged images found that less than 0.3% contained sexual or violent content; the overwhelming majority depicted hands with anatomical variation.
- Approximately one in four images depicting standard five-fingered hands also received an NSFW classification, indicating a structural bias against hand sketches as such.

## Repository Structure

```
production-of-absence-nsfw/
├── README.md
├── LICENSE                     # MIT License (code)
├── LICENSE-DATA                # CC-BY 4.0 (manual annotations)
├── CITATION.cff                # Citation metadata
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── safety_bypass.py        # The dummy_safety_checker (§5.1)
│   ├── generate.py             # Image generation pipeline (§3.1, §3.2)
│   ├── extract_clip.py         # CLIP embedding extraction (§5.3)
│   ├── compose_grid.py         # Typological grid composition (§6.1, Figure 8)
│   └── figures.py              # Regenerate paper figures (Figures 1–7)
├── data/
│   ├── README.md
│   ├── manual_annotations.csv  # Per-image labels (template)
│   └── category_counts.csv     # Aggregate counts (Figure 1 source data)
└── notebooks/
    └── (regeneration notebooks)
```

## Methodology

The methodology corresponds to §3 of the paper:

### §3.1 NSFW Filter Architecture

Stable Diffusion (`runwayml/stable-diffusion-v1-5`) implements a two-stage CLIP-based NSFW classifier:

1. **First stage** (pipeline-internal `safety_checker`): classifies images immediately after generation; if `has_nsfw_concepts=True`, the image is replaced with a black image. **This stage is bypassed in our methodology.**
2. **Second stage** (standalone `StableDiffusionSafetyChecker`): performs independent NSFW judgment, used here to flag generated images while preserving them for analysis.

### §3.2 Generation Parameters

| Parameter | Value |
|---|---|
| Model | `runwayml/stable-diffusion-v1-5` |
| Prompt | `"hand sketch, pencil drawing"` |
| `num_inference_steps` | 20 |
| `guidance_scale` | 7.5 |
| Resolution | 512 × 512 |
| Random seed | 42 |
| Number of trials | 42,000+ |

### §3.3 Manual Observation

A subset of approximately 316 NSFW-flagged images was manually observed by the author. Categories: polydactyly-like, five-fingered with anatomical irregularities, oligodactyly-like, atypical finger arrangement, other generation artifacts, sexual content, violent content.

The observation is *exploratory and conducted by a single annotator*; it does not include inter-rater reliability or formal annotation protocol. See §3.3 of the paper for a full discussion of methodological limitations.

## Reproducibility

The complete experimental dataset (~10,000 NSFW-flagged hand sketch images) is **not redistributed** due to:

- Model licensing constraints (CreativeML OpenRAIL-M)
- Content sensitivity considerations

However, all generation parameters are fully documented, allowing reproduction of the dataset by running the published code with the same model version and random seed.

### Reproduction Steps

```bash
# 1. Clone the repository
git clone https://github.com/<username>/production-of-absence-nsfw.git
cd production-of-absence-nsfw

# 2. Install dependencies (Python 3.10+ recommended, CUDA-capable GPU required)
pip install -r requirements.txt

# 3. Generate the dataset
#    Note: 42,000 trials on a single A100 takes approximately 12-14 hours.
#    A reduced count (e.g., NUM_TRIALS=1000) is recommended for initial testing.
python -m src.generate --num_trials 42000 --output_dir output

# 4. Extract CLIP embeddings for collected images
python -m src.extract_clip --input_dir output/nsfw_images --output output/embeddings.npz

# 5. Regenerate paper figures (Figures 1, 5, 6, 7)
python -m src.figures --records output/generation_records.json \
                       --embeddings output/embeddings.npz \
                       --annotations data/manual_annotations.csv \
                       --output_dir figures

# 6. Compose typological grid (Figure 8)
python -m src.compose_grid --input_dir output/nsfw_images \
                            --output figures/figure8_typology_grid.jpg \
                            --grid_size 100
```

### Hardware Notes

- The full 42,000-trial generation requires a CUDA-capable GPU. Tested on Google Colab with NVIDIA A100.
- For testing the methodology without the full generation, use `--num_trials 100`.
- The standalone `StableDiffusionSafetyChecker` and `dummy_safety_checker` bypass work on CPU as well, but generation will be impractically slow.

## License

This repository uses two licenses:

- **Source code** (everything in `src/` and notebooks): [MIT License](LICENSE)
- **Manual annotations** (`data/manual_annotations.csv` and `data/category_counts.csv`): [CC-BY 4.0](LICENSE-DATA)

Generated images are subject to the [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license) license inherited from the underlying Stable Diffusion model.

## Citation

If you use this code or the manual annotations in your research, please cite:

```bibtex
@inproceedings{kubota2026production,
  author    = {Kubota, Nozomu},
  title     = {Production of Absence: Reading Stable Diffusion's {NSFW} Filter through Code and Typology},
  booktitle = {Proceedings of the Expanded Conference 2026},
  year      = {2026}
}
```

## Acknowledgments

This research was conducted at Creator's NEXT.

## Author

**Nozomu Kubota**
Creator's NEXT
nozomu.k@cnxt.jp
