"""
figures.py
==========

Regenerate paper figures from saved generation records, CLIP embeddings, and
manual annotations.

Generated figures:
  - Figure 1: Manual annotation category distribution
  - Figure 2-4: CLIP feature dimension distributions (321, 178, 166)
  - Figure 5: Mean vector difference (SAFE - NSFW) across all 512 dimensions
  - Figure 6: Generation time distribution
  - Figure 7: NSFW classification rate convergence

(Figure 8, the typological grid, is generated separately by compose_grid.py.)

Usage
-----
    python -m src.figures \\
        --records output/generation_records.json \\
        --embeddings_nsfw output/embeddings_nsfw.npz \\
        --embeddings_safe output/embeddings_safe.npz \\
        --annotations data/manual_annotations.csv \\
        --output_dir figures

Reference: Kubota (2026), §4.1-4.5.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Figure styling
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

# Colors matching the paper figures
COLOR_NSFW = "#FF4D4D"
COLOR_SAFE = "#4DAF4D"
COLOR_BAR = "#7FBADC"  # Light blue for Figure 1
COLOR_LINE = "#1F4FBF"  # Blue for Figure 7
COLOR_DIFF = "#9933CC"  # Purple for Figure 5

# Categories in display order (paper Figure 1)
CATEGORY_ORDER = [
    "Sexual",
    "Violent",
    "Polydactyly",
    "Five-fingered",
    "Oligodactyly",
    "Atypical Arrangement",
    "Misclassified",
    "Other",
]


def figure1_category_distribution(annotations_path: Path, output_path: Path):
    """Figure 1: Manual annotation category distribution."""
    df = pd.read_csv(annotations_path)

    counts = df["category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(counts)), counts.values, color=COLOR_BAR,
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.set_ylabel("Number of Images")
    ax.set_xlabel("Category")
    ax.set_title(f"Category Distribution (Manual Annotation, n={len(df)})")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  -> {output_path}")


def figure_dimension_histogram(
    embeddings_nsfw: np.ndarray,
    embeddings_safe: np.ndarray,
    dim: int,
    output_path: Path,
):
    """Figures 2-4: Single CLIP dimension distribution comparison."""
    nsfw_values = embeddings_nsfw[:, dim]
    safe_values = embeddings_safe[:, dim]

    all_values = np.concatenate([nsfw_values, safe_values])
    bins = np.linspace(all_values.min(), all_values.max(), 30)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(nsfw_values, bins=bins, color=COLOR_NSFW, alpha=0.7,
            label="NSFW", edgecolor="white", linewidth=0.3)
    ax.hist(safe_values, bins=bins, color=COLOR_SAFE, alpha=0.7,
            label="SAFE", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution Comparison: CLIP Dimension {dim}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  -> {output_path}")


def figure5_mean_vector_difference(
    embeddings_nsfw: np.ndarray,
    embeddings_safe: np.ndarray,
    output_path: Path,
):
    """Figure 5: Mean vector difference (SAFE - NSFW) across all dimensions."""
    diff = embeddings_safe.mean(axis=0) - embeddings_nsfw.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(diff)), diff, color=COLOR_DIFF, width=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Difference")
    ax.set_title("Mean Vector Difference (SAFE − NSFW)")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  -> {output_path}")


def figure6_generation_time(records: list, output_path: Path):
    """Figure 6: Generation time distribution for SAFE vs NSFW."""
    df = pd.DataFrame(records)
    nsfw_times = df.loc[df["is_nsfw"], "generation_time"]
    safe_times = df.loc[~df["is_nsfw"], "generation_time"]

    all_times = pd.concat([nsfw_times, safe_times])
    bins = np.linspace(all_times.min(), all_times.max(), 25)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(safe_times, bins=bins, color=COLOR_SAFE, alpha=0.7,
            label="SAFE", edgecolor="white", linewidth=0.3)
    ax.hist(nsfw_times, bins=bins, color=COLOR_NSFW, alpha=0.7,
            label="NSFW", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Generation Time (sec)")
    ax.set_ylabel("Frequency")
    ax.set_title("Generation Time Distribution: SAFE vs. NSFW")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  -> {output_path}")


def figure7_nsfw_convergence(records: list, output_path: Path):
    """Figure 7: NSFW classification rate convergence over trials."""
    df = pd.DataFrame(records).sort_values("trial").reset_index(drop=True)
    cumulative_nsfw = df["is_nsfw"].cumsum()
    trials = np.arange(1, len(df) + 1)
    rate = cumulative_nsfw / trials

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(trials, rate, color=COLOR_LINE, linewidth=1.2, label="NSFW Rate")
    ax.set_xlabel("Number of Generations")
    ax.set_ylabel("NSFW Rate")
    ax.set_title("NSFW Classification Rate over Generation Trials")
    ax.set_ylim(0, max(0.30, rate.max() * 1.1))
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate paper figures.")
    parser.add_argument("--records", required=True,
                        help="generation_records.json from generate.py")
    parser.add_argument("--embeddings_nsfw",
                        help="NPZ file with NSFW image CLIP embeddings")
    parser.add_argument("--embeddings_safe",
                        help="NPZ file with SAFE sample CLIP embeddings")
    parser.add_argument("--annotations",
                        help="manual_annotations.csv")
    parser.add_argument("--output_dir", default="figures")
    parser.add_argument("--dimensions", default="321,178,166",
                        help="Comma-separated CLIP dimensions for Figs 2-4")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Regenerating paper figures...")

    # Figure 1
    if args.annotations:
        figure1_category_distribution(
            Path(args.annotations),
            output_dir / "figure1_category_distribution.png"
        )

    # Figures 2-4 and 5
    if args.embeddings_nsfw and args.embeddings_safe:
        nsfw_data = np.load(args.embeddings_nsfw)
        safe_data = np.load(args.embeddings_safe)
        emb_nsfw = nsfw_data["embeddings"]
        emb_safe = safe_data["embeddings"]

        for dim_str in args.dimensions.split(","):
            dim = int(dim_str.strip())
            figure_dimension_histogram(
                emb_nsfw, emb_safe, dim,
                output_dir / f"figure_clip_dim{dim}.png"
            )

        figure5_mean_vector_difference(
            emb_nsfw, emb_safe,
            output_dir / "figure5_mean_vector_diff.png"
        )

    # Figures 6 and 7 from records
    with open(args.records) as f:
        records = json.load(f)

    figure6_generation_time(records, output_dir / "figure6_generation_time.png")
    figure7_nsfw_convergence(records, output_dir / "figure7_nsfw_convergence.png")

    print(f"\nAll figures written to {output_dir}/")


if __name__ == "__main__":
    main()
