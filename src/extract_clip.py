"""
extract_clip.py
===============

Extract CLIP (ViT-B/32) embeddings from collected images for the dimension
analysis in §4.2 (Figures 2-4) and mean vector difference analysis in §4.3
(Figure 5).

This corresponds to the line:

    clip_feat = clip_model.encode_image(clip_input)

discussed critically in §5.3 of the paper. The CLIP embedding space itself,
and the dataset on which it was trained, are revealed to be structurally
biased through the dimension analysis enabled by these embeddings.

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
    """Load CLIP ViT-B/32 model and image processor."""
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
    """Extract 512-dimensional CLIP image embeddings.

    The 512 dimensions are the basis of the §4.2 dimension analysis. Specific
    dimensions (321, 178, 166) show pronounced separation between NSFW and SAFE
    classified groups, suggesting structural bias in the embedding space.
    """
    embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CLIP"):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            # The line discussed in §5.3
            features = model.get_image_features(**inputs)

        embeddings.append(features.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP ViT-B/32 embeddings (§4.2, §5.3)."
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
    )

    print(f"Saved {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
