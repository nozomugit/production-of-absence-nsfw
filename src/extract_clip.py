"""
extract_clip.py
===============

Extract auxiliary CLIP ViT-B/32 embeddings from collected images for the
descriptive dimension analysis in §4.2 (Figures 2-4) and mean-vector difference
analysis in §4.3 (Figure 5). This corresponds to the implementation line:

    features = model.get_image_features(**inputs)

The auxiliary model here is ``openai/clip-vit-base-patch32`` (ViT-B/32). It is
not the safety checker's internal CLIP vision encoder, which is ViT-L/14 in
``CompVis/stable-diffusion-safety-checker``. Consequently, differences observed
in these auxiliary embeddings are descriptive associations between groups.
They do not identify the checker's decision features and, by themselves, do not
establish a causal mechanism or prove training-data or model bias.

Usage
-----
    python -m src.extract_clip --input_dir output/nsfw_images \\
                                --output output/embeddings_nsfw.npz

Reference: Kubota (2026), §4.2, §4.3, §5.3.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel


CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def load_clip(device: str):
    """Load the auxiliary CLIP ViT-B/32 model and its image processor."""
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    model.eval()
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def extract_embeddings(
    image_paths: list,
    model: CLIPModel,
    processor: CLIPImageProcessor,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract 512-dimensional auxiliary CLIP image embeddings.

    These dimensions support descriptive comparisons between checker-flagged
    and checker-unflagged samples. Because this ViT-B/32 model is separate from
    the checker's internal ViT-L/14 encoder, the embeddings are not a trace of
    the checker's causal decision process or evidence of bias on their own.
    """
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CLIP"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            # This is the actual Transformers CLIP image-feature API. These
            # features come from the auxiliary ViT-B/32, not the checker.
            features = model.get_image_features(**inputs)

        embeddings.append(features.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract auxiliary CLIP ViT-B/32 embeddings for descriptive "
            "group comparisons; this is not the safety checker's encoder."
        )
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing PNG/JPG images")
    parser.add_argument("--output", required=True,
                        help="Output .npz file path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--extensions", default="png,jpg,jpeg",
                        help="Comma-separated image extensions to process")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    extensions = [f".{ext.strip().lower()}" for ext in args.extensions.split(",")]
    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in extensions
    ])

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"Found {len(image_paths)} images in {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, processor = load_clip(device)

    embeddings = extract_embeddings(
        image_paths, model, processor, device, args.batch_size
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        filenames=np.array([p.name for p in image_paths]),
        model_id=np.array(CLIP_MODEL_ID),
        embedding_role=np.array(
            "auxiliary descriptive features; separate from checker ViT-L/14"
        ),
    )

    print(f"Saved {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
