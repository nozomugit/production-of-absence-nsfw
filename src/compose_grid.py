"""
compose_grid.py
===============

Compose the typological grid of NSFW-flagged images (Figure 8 in the paper,
The Invisible Hand of AI installation, §6.1).

Arranges collected images into an N×N grid, rendered in grayscale, to
materialize the "production of absence" as a forensic typology.

Reference: Kubota (2026), §6.1 "The 100×100 Grid".

Usage
-----
    # Full installation (100×100 = 10,000 images)
    python -m src.compose_grid --input_dir output/nsfw_images \\
                                --output figures/figure8_typology_grid.jpg \\
                                --grid_size 100

    # Excerpt for paper figure (10×10 = 100 images)
    python -m src.compose_grid --input_dir output/nsfw_images \\
                                --output figures/figure8_excerpt.jpg \\
                                --grid_size 10
"""

import argparse
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def compose_grid(
    image_paths: list,
    grid_size: int,
    cell_size: int = 200,
    grayscale: bool = True,
    seed: int = 42,
) -> Image.Image:
    """Compose a square grid of images.

    Parameters
    ----------
    image_paths : list of Path
        Source images. If more than grid_size**2 are provided, a random
        sample is taken (deterministic given `seed`).
    grid_size : int
        Number of cells per side (e.g., 100 for a 100×100 = 10,000 grid).
    cell_size : int
        Pixel size of each cell (output image is cell_size * grid_size pixels
        per side).
    grayscale : bool
        Render images in grayscale (paper uses grayscale for the installation).
    seed : int
        Random seed for sampling, when applicable.

    Returns
    -------
    grid : PIL.Image.Image
        The composed grid image.
    """
    n_cells = grid_size * grid_size

    if len(image_paths) < n_cells:
        raise ValueError(
            f"Need at least {n_cells} images for a {grid_size}x{grid_size} grid, "
            f"but only {len(image_paths)} provided."
        )

    rng = random.Random(seed)
    selected = rng.sample(image_paths, n_cells) if len(image_paths) > n_cells else list(image_paths)

    canvas_size = cell_size * grid_size
    mode = "L" if grayscale else "RGB"
    bg_color = 0 if grayscale else (0, 0, 0)
    canvas = Image.new(mode, (canvas_size, canvas_size), bg_color)

    for idx, path in enumerate(tqdm(selected, desc=f"Composing {grid_size}x{grid_size} grid")):
        row = idx // grid_size
        col = idx % grid_size

        img = Image.open(path)
        if grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        img = img.resize((cell_size, cell_size), Image.LANCZOS)
        canvas.paste(img, (col * cell_size, row * cell_size))

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Compose typological grid (Figure 8 / installation §6.1)."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing source images")
    parser.add_argument("--output", required=True,
                        help="Output image path (.jpg recommended)")
    parser.add_argument("--grid_size", type=int, default=100,
                        help="Cells per side (100 for full installation)")
    parser.add_argument("--cell_size", type=int, default=200,
                        help="Pixel size of each cell")
    parser.add_argument("--no_grayscale", action="store_true",
                        help="Keep RGB instead of converting to grayscale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (paper uses 42)")
    parser.add_argument("--extensions", default="png,jpg,jpeg",
                        help="Comma-separated image extensions to include")
    parser.add_argument("--quality", type=int, default=90,
                        help="JPEG quality (1-100)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    extensions = [f".{ext.strip().lower()}" for ext in args.extensions.split(",")]
    image_paths = sorted([
        p for p in input_dir.iterdir() if p.suffix.lower() in extensions
    ])

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"Found {len(image_paths)} images. Composing {args.grid_size}x{args.grid_size} grid...")

    grid = compose_grid(
        image_paths=image_paths,
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        grayscale=not args.no_grayscale,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {}
    if output_path.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs["quality"] = args.quality
        save_kwargs["optimize"] = True

    grid.save(output_path, **save_kwargs)

    canvas_size = args.cell_size * args.grid_size
    print(f"Saved: {output_path} ({canvas_size}x{canvas_size} pixels, "
          f"{args.grid_size**2} cells)")


if __name__ == "__main__":
    main()
