"""
figures.py
==========

Generate updated descriptive figures from saved generation records, auxiliary
CLIP embeddings, and manual annotations. The comparison histograms normalize
each group internally; they do not reproduce the paper's archived raw-count
plots pixel for pixel.

Generated figures:
  - Figure 1: Manual annotation category distribution
  - Figure 2-4: CLIP feature dimension distributions (321, 178, 166)
  - Figure 5: Mean vector difference (unflagged - flagged) across all 512 dimensions
  - Figure 6: Generation time distribution
  - Figure 7: Cumulative observed flag rate in a single run

(Figure 8, the typological grid, is generated separately by compose_grid.py.)

Usage
-----
    python -m src.figures \\
        --records output/generation_records.json \\
        --embeddings_nsfw output/embeddings_nsfw.npz \\
        --embeddings_safe output/embeddings_safe.npz \\
        --annotations data/category_counts.csv \\
        --output_dir figures

Reference: Kubota (2026), §4.1-4.5.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter


# Figure styling
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

# Colorblind-accessible colors (Okabe--Ito palette). Line style and outlines
# provide a second visual channel so that group identity never depends on color
# alone.
COLOR_FLAGGED = "#D55E00"    # Vermillion
COLOR_UNFLAGGED = "#0072B2"  # Blue
COLOR_BAR = "#0072B2"        # Blue for Figure 1
COLOR_LINE = "#0072B2"       # Blue for Figure 7
COLOR_DIFF = "#CC79A7"       # Reddish purple for Figure 5

# Categories in display order (paper Figure 1)
CATEGORY_ORDER = [
    "Polydactyly-like",
    "Oligodactyly-like",
    "Five-fingered with anatomical irregularities",
    "Atypical finger arrangement",
    "Other generation artifacts",
    "Sexual content",
    "Violent content",
]

CATEGORY_DISPLAY_LABELS = {
    "Polydactyly-like": "Polydactyly-\nlike",
    "Oligodactyly-like": "Oligodactyly-\nlike",
    "Five-fingered with anatomical irregularities":
        "Five-fingered\nwith anatomical\nirregularities",
    "Atypical finger arrangement": "Atypical\nfinger\narrangement",
    "Other generation artifacts": "Other\ngeneration\nartifacts",
    "Sexual content": "Sexual\ncontent",
    "Violent content": "Violent\ncontent",
}

# Keep older repository annotation files usable while presenting the more
# cautious terminology adopted by the paper. Duplicate categories created by
# normalization are summed below.
LEGACY_CATEGORY_ALIASES = {
    "Polydactyly": "Polydactyly-like",
    "Oligodactyly": "Oligodactyly-like",
    "Five-fingered": "Five-fingered with anatomical irregularities",
    "Atypical Arrangement": "Atypical finger arrangement",
    "Other": "Other generation artifacts",
    "Misclassified": "Other generation artifacts",
    "Sexual": "Sexual content",
    "Violent": "Violent content",
}


def _normalized_category_names(series: pd.Series) -> pd.Series:
    """Map legacy category labels to the paper's current terminology."""
    return series.replace(LEGACY_CATEGORY_ALIASES)


def _common_bins(*arrays: np.ndarray, n_bins: int) -> np.ndarray:
    """Return finite common bin edges, including for constant-valued inputs."""
    flattened = [np.asarray(values).ravel() for values in arrays]
    if any(values.size == 0 for values in flattened):
        raise ValueError("Both comparison groups must contain at least one value.")

    all_values = np.concatenate(flattened)
    if not np.isfinite(all_values).all():
        raise ValueError("Plot values must all be finite.")

    lower = float(all_values.min())
    upper = float(all_values.max())
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 0.5)
        lower -= padding
        upper += padding
    return np.linspace(lower, upper, n_bins + 1)


def figure1_category_distribution(input_path: Path, output_path: Path):
    """Figure 1: Manual annotation category distribution.

    Accepts either format:
    - Aggregate counts (category_counts.csv): columns = category, count, percentage
    - Full per-image labels: columns = image_id, category, notes

    The aggregate format is what is published in this repository (see paper
    §3.3 and data/README.md). The bundled manual_annotations_example.csv is
    schema documentation, not empirical input. Full per-image data are
    supported for any future extension where such labels are released.
    """
    input_path = Path(input_path)
    if input_path.name == "manual_annotations_example.csv":
        raise ValueError(
            "manual_annotations_example.csv contains illustrative schema rows, "
            "not empirical Figure 1 data; use data/category_counts.csv"
        )

    df = pd.read_csv(input_path)
    if "category" not in df.columns:
        raise ValueError("Annotation data must include a 'category' column.")
    df["category"] = _normalized_category_names(df["category"])
    if df["category"].isna().any():
        raise ValueError("Annotation categories must not be empty.")
    unknown_categories = sorted(set(df["category"]) - set(CATEGORY_ORDER))
    if unknown_categories:
        raise ValueError(
            "Unrecognized annotation categories: "
            + ", ".join(map(str, unknown_categories))
        )

    if "count" in df.columns:
        # Aggregate format (category_counts.csv) — direct counts per category
        subset_title = (
            "Random Flagged Subset"
            if input_path.name == "category_counts.csv"
            else "Supplied Flagged Subset"
        )
        df["count"] = pd.to_numeric(df["count"], errors="raise")
        counts = (
            df.groupby("category", sort=False)["count"].sum()
            .reindex(CATEGORY_ORDER, fill_value=0)
        )
    else:
        # Full per-image format supplied by the user — count occurrences.
        subset_title = "Supplied Flagged Subset"
        counts = df["category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)

    counts = pd.to_numeric(counts, errors="raise")
    if (counts < 0).any() or not np.allclose(counts, np.round(counts)):
        raise ValueError("Category counts must be non-negative integers.")
    counts = counts.astype(int)
    n = int(counts.sum())
    if n == 0:
        raise ValueError("Annotation data contains no observations.")
    shares = counts / n * 100.0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(
        range(len(counts)),
        shares.values,
        color=COLOR_BAR,
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(
        [CATEGORY_DISPLAY_LABELS[name] for name in counts.index], fontsize=8.5,
    )
    ax.set_ylabel("Share of manually observed subset (%)")
    ax.set_xlabel("Category")
    ax.set_title(f"Manual Observation of {subset_title} (n={n})")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(40.0, float(shares.max()) * 1.20))
    for bar, count, share in zip(bars, counts, shares):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.7,
            f"{count}\n({share:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

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

    bins = _common_bins(nsfw_values, safe_values, n_bins=29)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(
        nsfw_values,
        bins=bins,
        weights=np.full(nsfw_values.shape, 1.0 / len(nsfw_values)),
        histtype="step",
        color=COLOR_FLAGGED,
        linestyle="-",
        linewidth=2.0,
        label=f"Flagged (n={len(nsfw_values):,})",
    )
    ax.hist(
        safe_values,
        bins=bins,
        weights=np.full(safe_values.shape, 1.0 / len(safe_values)),
        histtype="step",
        color=COLOR_UNFLAGGED,
        linestyle="--",
        linewidth=2.0,
        label=f"Unflagged (n={len(safe_values):,})",
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Proportion within group")
    ax.set_title(f"Descriptive CLIP Distribution: Dimension {dim}")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, prune=None))
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
    """Figure 5: Mean difference (unflagged - flagged) across dimensions."""
    diff = embeddings_safe.mean(axis=0) - embeddings_nsfw.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(
        range(len(diff)), diff, color=COLOR_DIFF, edgecolor=COLOR_DIFF,
        linewidth=0.25, width=1.0,
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Difference in group means")
    ax.set_title("Descriptive CLIP Mean Difference (Unflagged − Flagged)")
    ticks = np.arange(0, len(diff), 100, dtype=int)
    ax.set_xticks(ticks)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  -> {output_path}")


def figure6_generation_time(records: list, output_path: Path):
    """Figure 6: Within-group generation-time distributions."""
    df = pd.DataFrame(records)
    nsfw_times = df.loc[df["is_nsfw"], "generation_time"]
    safe_times = df.loc[~df["is_nsfw"], "generation_time"]

    bins = _common_bins(
        nsfw_times.to_numpy(), safe_times.to_numpy(), n_bins=24,
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(
        safe_times,
        bins=bins,
        weights=np.full(len(safe_times), 1.0 / len(safe_times)),
        histtype="step",
        color=COLOR_UNFLAGGED,
        linestyle="--",
        linewidth=2.0,
        label=f"Unflagged (n={len(safe_times):,})",
    )
    ax.hist(
        nsfw_times,
        bins=bins,
        weights=np.full(len(nsfw_times), 1.0 / len(nsfw_times)),
        histtype="step",
        color=COLOR_FLAGGED,
        linestyle="-",
        linewidth=2.0,
        label=f"Flagged (n={len(nsfw_times):,})",
    )
    ax.set_xlabel("Generation Time (sec)")
    ax.set_ylabel("Proportion within group")
    ax.set_title("Generation Time Distributions: Flagged and Unflagged Outputs")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, prune=None))
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  -> {output_path}")


def figure7_nsfw_convergence(records: list, output_path: Path):
    """Figure 7: Cumulative observed flag rate in one recorded run.

    The legacy function name is retained for CLI and downstream compatibility;
    the figure deliberately makes no statistical convergence claim.
    """
    df = pd.DataFrame(records).sort_values("trial").reset_index(drop=True)
    if df.empty:
        raise ValueError("At least one generation record is required.")
    cumulative_nsfw = df["is_nsfw"].cumsum()
    trials = np.arange(1, len(df) + 1)
    rate = cumulative_nsfw / trials

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        trials,
        rate,
        color=COLOR_LINE,
        linestyle="-",
        linewidth=1.4,
        label="Cumulative observed rate",
    )
    ax.set_xlabel("Number of Generations")
    ax.set_ylabel("Cumulative flagged-output proportion")
    ax.set_title("Cumulative Observed Flag Rate in a Single Run")
    ax.set_ylim(0, min(1.0, max(0.30, float(rate.max()) * 1.05)))
    if len(df) > 1:
        # The series begins at generation 1; do not introduce an artificial
        # observation at the origin or let an x=0 margin imply one.
        ax.set_xlim(1, len(df))
    else:
        ax.set_xlim(0.5, 1.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune=None))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate updated descriptive analysis figures."
    )
    parser.add_argument("--records", required=True,
                        help="generation_records.json from generate.py")
    parser.add_argument("--embeddings_nsfw",
                        help="NPZ file with flagged-output CLIP embeddings")
    parser.add_argument("--embeddings_safe",
                        help="NPZ file with unflagged-output CLIP embeddings")
    parser.add_argument(
        "--annotations",
        default="data/category_counts.csv",
        help=(
            "aggregate counts (default: data/category_counts.csv) or a full "
            "per-image annotation CSV; the bundled example CSV is not data"
        ),
    )
    parser.add_argument("--output_dir", default="figures")
    parser.add_argument("--dimensions", default="321,178,166",
                        help="Comma-separated CLIP dimensions for Figs 2-4")
    args = parser.parse_args()

    if bool(args.embeddings_nsfw) != bool(args.embeddings_safe):
        parser.error(
            "--embeddings_nsfw and --embeddings_safe must be supplied together"
        )

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Figure output path is not a directory: {output_dir}"
            )
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Figure output directory is not empty: {output_dir}. "
                "Choose a new directory or archive the existing figures first."
            )
    else:
        output_dir.mkdir(parents=True)

    print("Generating updated descriptive figures...")

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

    print(f"\nAll requested figures written to {output_dir}/")


if __name__ == "__main__":
    main()
