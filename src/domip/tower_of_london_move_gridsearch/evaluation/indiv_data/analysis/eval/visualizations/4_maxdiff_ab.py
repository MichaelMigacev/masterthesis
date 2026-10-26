"""
Max-difference alpha/beta visualization for Tower of London gridsearch data.

Instead of dimensionality reduction, this script selects the 2 (alpha, beta)
parameter pairs that maximally separate the input-folder groups. It then uses
the ``kendall_tau_mean`` values at those 2 pairs as the x and y axes of a
scatter plot where each subfolder gets its own color and legend entry.

Algorithm
---------
1. Load all .csv files from each subfolder.
2. Average the ``kendall_tau_mean`` values across all files within each
   subfolder (group), producing one group-level DataFrame per subfolder.
3. For every unique (alpha, beta) combination compute the sum of all
   pairwise Euclidean distances between the N groups (C(N,2) distances).
4. Pick the top-2 (alpha, beta) pairs with the highest summed distance.
5. Go through every individual .csv file, extract ``kendall_tau_mean`` at
   those 2 selected (alpha, beta) pairs, and plot them as (x, y).
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Max-difference alpha/beta visualization of TOL gridsearch data.",
    )
    parser.add_argument(
        "--input_folder",
        type=Path,
        help="Path to the folder containing subfolders with .csv files.",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        help="Path to the folder where the plot will be saved.",
    )
    parser.add_argument(
        "--plot_name",
        type=str,
        help="Name for the output plot file (without extension).",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="png",
        help="File extension for the saved plot (default: png).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Number of maximally-different (alpha, beta) pairs to select (default: 2).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_subfolder_data(subfolder_path: Path) -> list[pd.DataFrame]:
    """
    Load all .csv files from a single subfolder.

    Each CSV is expected to have at least the columns
    ``alpha``, ``beta``, ``kendall_tau_mean``.

    Returns a list of DataFrames, one per CSV file.
    """
    csv_files = sorted(subfolder_path.glob("*.csv"), key=lambda p: p.stem)
    frames: list[pd.DataFrame] = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        frames.append(df)

    return frames


def load_all_data(input_folder: Path) -> tuple[dict[str, list[pd.DataFrame]], list[str]]:
    """
    Walk through all subfolders of *input_folder*, load CSV data, and return
    a dict mapping subfolder name -> list of DataFrames, plus a sorted list
    of group names.
    """
    subfolders = sorted(
        [d for d in input_folder.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    group_data: dict[str, list[pd.DataFrame]] = {}
    for subfolder in subfolders:
        group_data[subfolder.name] = load_subfolder_data(subfolder)

    if not group_data:
        raise FileNotFoundError(
            f"No subfolders with .csv files found in '{input_folder}'."
        )

    return group_data, sorted(group_data.keys())


# ---------------------------------------------------------------------------
# Averaging within groups
# ---------------------------------------------------------------------------

def average_group_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Average a list of DataFrames from the same group.

    Groups by (alpha, beta) and computes the mean of ``kendall_tau_mean``.
    """
    combined = pd.concat(frames, ignore_index=True)
    averaged = (
        combined.groupby(["alpha", "beta"], as_index=False)
        .agg({"kendall_tau_mean": "mean", "kendall_tau_std": "mean"})
    )
    return averaged


# ---------------------------------------------------------------------------
# Selecting maximally-different (alpha, beta) pairs
# ---------------------------------------------------------------------------

def compute_pairwise_distance_sum(
    alpha: float,
    beta: float,
    group_averaged: dict[str, pd.DataFrame],
) -> float:
    """
    For a given (alpha, beta) pair, compute the sum of all pairwise
    Euclidean distances between groups.

    With N groups there are C(N, 2) pairwise distances.
    Euclidean distance between two scalar kendall_tau_mean values is
    simply their absolute difference.
    """
    values: list[tuple[str, float]] = []
    for group_name, df in group_averaged.items():
        mask = (df["alpha"] == alpha) & (df["beta"] == beta)
        if mask.any():
            val = df.loc[mask, "kendall_tau_mean"].iloc[0]
            values.append((group_name, val))

    if len(values) < 2:
        return 0.0

    total = 0.0
    for (_, v1), (_, v2) in combinations(values, 2):
        total += np.sqrt((v1 - v2) ** 2)  # Euclidean distance in 1-D
    return total


def find_maxdiff_pairs(
    group_averaged: dict[str, pd.DataFrame],
    top_k: int = 2,
) -> list[tuple[float, float]]:
    """
    Iterate over all unique (alpha, beta) combinations present in the
    averaged data and return the *top_k* pairs with the highest summed
    pairwise Euclidean distance across groups.
    """
    # Collect all unique (alpha, beta) pairs from any group
    all_pairs: set[tuple[float, float]] = set()
    for df in group_averaged.values():
        for _, row in df.iterrows():
            all_pairs.add((row["alpha"], row["beta"]))

    scores: list[tuple[float, float, float]] = []  # (alpha, beta, score)
    for alpha, beta in all_pairs:
        score = compute_pairwise_distance_sum(alpha, beta, group_averaged)
        scores.append((alpha, beta, score))

    # Sort descending by score
    scores.sort(key=lambda x: x[2], reverse=True)

    print("Top-10 (alpha, beta) pairs by summed pairwise distance:")
    for i, (a, b, s) in enumerate(scores[:10]):
        print(f"  {i+1}. alpha={a}, beta={b}  ->  score={s:.6f}")

    return [(a, b) for a, b, _ in scores[:top_k]]


# ---------------------------------------------------------------------------
# Building the 2-D coordinates for individual files
# ---------------------------------------------------------------------------

def extract_coordinates(
    frames: list[pd.DataFrame],
    pair1: tuple[float, float],
    pair2: tuple[float, float],
) -> list[tuple[float, float]]:
    """
    For each individual CSV file, extract kendall_tau_mean at pair1 and pair2.

    Returns a list of (x, y) tuples, one per file.
    """
    coords: list[tuple[float, float]] = []
    for df in frames:
        x_mask = (df["alpha"] == pair1[0]) & (df["beta"] == pair1[1])
        y_mask = (df["alpha"] == pair2[0]) & (df["beta"] == pair2[1])

        x_val = df.loc[x_mask, "kendall_tau_mean"].values
        y_val = df.loc[y_mask, "kendall_tau_mean"].values

        if x_val.size > 0 and y_val.size > 0:
            coords.append((float(x_val[0]), float(y_val[0])))

    return coords


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    group_coords: dict[str, list[tuple[float, float]]],
    pair1: tuple[float, float],
    pair2: tuple[float, float],
    output_folder: Path,
    plot_name: str,
    extension: str,
):
    """Create and save a scatter plot of the 2-D data."""
    groups = sorted(group_coords.keys())
    n_groups = len(groups)

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = (
        matplotlib.colormaps["Dark2"]
        if n_groups <= 20
        else matplotlib.colormaps["tab10"]
    )
    colors = [cmap(i % cmap.N) for i in range(n_groups)]
    group_to_color = dict(zip(groups, colors))

    for group in groups:
        coords = group_coords[group]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        ax.scatter(
            xs,
            ys,
            label=group,
            color=group_to_color[group],
            alpha=0.7,
            edgecolors="k",
            s=40,
        )

    ax.set_xlabel(
        f"kendall_tau_mean  (alpha={pair1[0]}, beta={pair1[1]})"
    )
    ax.set_ylabel(
        f"kendall_tau_mean  (alpha={pair2[0]}, beta={pair2[1]})"
    )
    ax.set_title("Max-Difference Alpha/Beta – TOL Gridsearch Data")
    ax.legend(title="Group", fontsize="small", title_fontsize="medium")
    ax.grid(True, alpha=0.3)

    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = output_folder / f"{plot_name}.{extension}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Loading data from: {args.input_folder}")
    group_data, group_names = load_all_data(args.input_folder)
    total_files = sum(len(frames) for frames in group_data.values())
    print(
        f"Loaded {total_files} files across {len(group_names)} groups."
    )

    # Step 1: Average within each group
    print("Averaging within each group…")
    group_averaged: dict[str, pd.DataFrame] = {}
    for group_name in group_names:
        group_averaged[group_name] = average_group_frames(group_data[group_name])

    # Step 2: Find the top-k maximally-different (alpha, beta) pairs
    print("Finding maximally-different (alpha, beta) pairs…")
    top_pairs = find_maxdiff_pairs(group_averaged, top_k=args.top_k)
    pair1, pair2 = top_pairs[0], top_pairs[1]
    print(f"Selected pair 1 (x-axis): alpha={pair1[0]}, beta={pair1[1]}")
    print(f"Selected pair 2 (y-axis): alpha={pair2[0]}, beta={pair2[1]}")

    # Step 3: Extract coordinates from individual files
    print("Extracting coordinates from individual files…")
    group_coords: dict[str, list[tuple[float, float]]] = {}
    for group_name in group_names:
        group_coords[group_name] = extract_coordinates(
            group_data[group_name], pair1, pair2
        )
        print(
            f"  {group_name}: {len(group_coords[group_name])} points"
        )

    # Step 4: Plot
    plot_results(
        group_coords,
        pair1,
        pair2,
        args.output_folder,
        args.plot_name,
        args.extension,
    )


if __name__ == "__main__":
    main()
