"""
compose_grid.py
===============

Compose the typological grid of checker-flagged images (Figure 8 in the paper,
The Invisible Hand of AI installation, §6.1).

Arranges collected images into a rectangular or square grid. This utility
defaults to grayscale, matching the archived paper excerpt.

Reference: Kubota (2026), §6.1 "The 4 m × 1 m Grid".

Usage
-----
    # Full 4:1 installation layout (200×50 = 10,000 images)
    python -m src.compose_grid --input_dir output/nsfw_images \\
                                --output figures/figure8_typology_grid.jpg \\
                                --rows 200 --columns 50

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
    grid_size: int = None,
    cell_size: int = 200,
    grayscale: bool = True,
    seed: int = 42,
    grid_rows: int = None,
    grid_columns: int = None,
) -> Image.Image:
    """Compose a rectangular or square grid of images.

    Parameters
    ----------
    image_paths : list of Path
        Source images. If more than the requested cell count are provided, a
        random sample is taken (deterministic given `seed`).
    grid_size : int
        Number of cells per side for a square grid. Retained for compatibility;
        do not combine with `grid_rows` or `grid_columns`.
    cell_size : int
        Pixel size of each square cell. Output dimensions are
        ``cell_size * columns`` by ``cell_size * rows`` pixels.
    grayscale : bool
        Render images in grayscale. Set this to False to retain source color.
    seed : int
        Random seed for sampling, when applicable.
    grid_rows, grid_columns : int
        Rectangular grid dimensions. Both must be supplied together and
        cannot be combined with `grid_size`.

    Returns
    -------
    grid : PIL.Image.Image
        The composed grid image.
    """
    if grid_rows is None and grid_columns is None:
        if grid_size is None:
            raise ValueError("Provide grid_size or both grid_rows and grid_columns.")
        rows = columns = grid_size
    elif grid_rows is None or grid_columns is None:
        raise ValueError("grid_rows and grid_columns must be provided together.")
    elif grid_size is not None:
        raise ValueError("Use grid_size or grid_rows/grid_columns, not both.")
    else:
        rows, columns = grid_rows, grid_columns

    if rows <= 0 or columns <= 0 or cell_size <= 0:
        raise ValueError("Grid dimensions and cell_size must be positive integers.")

    n_cells = rows * columns

    if len(image_paths) < n_cells:
        raise ValueError(
            f"Need at least {n_cells} images for a {rows}x{columns} grid, "
            f"but only {len(image_paths)} provided."
        )

    rng = random.Random(seed)
    selected = (
        rng.sample(image_paths, n_cells)
        if len(image_paths) > n_cells
        else list(image_paths)
    )

    canvas_width = cell_size * columns
    canvas_height = cell_size * rows
    mode = "L" if grayscale else "RGB"
    bg_color = 0 if grayscale else (0, 0, 0)
    canvas = Image.new(mode, (canvas_width, canvas_height), bg_color)

    for idx, path in enumerate(
        tqdm(selected, desc=f"Composing {rows}x{columns} grid")
    ):
        row = idx // columns
        col = idx % columns

        with Image.open(path) as source:
            img = source.convert("L" if grayscale else "RGB")
            img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
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
    parser.add_argument(
        "--grid_size",
        type=int,
        help="Cells per side for a square grid (e.g., 10 for the paper excerpt)",
    )
    parser.add_argument("--rows", type=int,
                        help="Rows for a rectangular grid (default: 200)")
    parser.add_argument("--columns", type=int,
                        help="Columns for a rectangular grid (default: 50)")
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

    if args.grid_size is not None and (
        args.rows is not None or args.columns is not None
    ):
        parser.error("use --grid_size or --rows/--columns, not both")
    if (args.rows is None) != (args.columns is None):
        parser.error("--rows and --columns must be provided together")

    if args.grid_size is not None:
        grid_size = args.grid_size
        grid_rows = grid_columns = None
    else:
        grid_size = None
        grid_rows = args.rows if args.rows is not None else 200
        grid_columns = args.columns if args.columns is not None else 50

    input_dir = Path(args.input_dir)
    extensions = [f".{ext.strip().lower()}" for ext in args.extensions.split(",")]
    image_paths = sorted([
        p for p in input_dir.iterdir() if p.suffix.lower() in extensions
    ])

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    rows = grid_size if grid_size is not None else grid_rows
    columns = grid_size if grid_size is not None else grid_columns
    print(f"Found {len(image_paths)} images. Composing {rows}x{columns} grid...")

    grid = compose_grid(
        image_paths=image_paths,
        grid_size=grid_size,
        cell_size=args.cell_size,
        grayscale=not args.no_grayscale,
        seed=args.seed,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {}
    if output_path.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs["quality"] = args.quality
        save_kwargs["optimize"] = True

    grid.save(output_path, **save_kwargs)

    canvas_width = args.cell_size * columns
    canvas_height = args.cell_size * rows
    print(f"Saved: {output_path} ({canvas_width}x{canvas_height} pixels, "
          f"{rows * columns} cells)")


if __name__ == "__main__":
    main()
