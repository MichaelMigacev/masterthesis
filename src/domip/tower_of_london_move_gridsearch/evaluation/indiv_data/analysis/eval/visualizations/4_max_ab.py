"""
Top-k alpha/beta visualization for Tower of London gridsearch data.

For each CSV file, selects the rows with the top-k ``kendall_tau_mean``
values, averages their ``alpha`` and ``beta`` parameters, and plots the
resulting (alpha, beta) point on a log-log scatter plot. Each subfolder
(group) gets its own color and legend entry.

Algorithm
---------
1. Load all .csv files from each subfolder.
2. For each individual .csv file, find the top-k rows by ``kendall_tau_mean``.
3. Average the ``alpha`` and ``beta`` values of those top-k rows to produce
   a single (alpha, beta) coordinate per file.
4. Plot all points on a scatter with logarithmic x (alpha) and y (beta) axes.

This file contains code written by AI.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Top-k alpha/beta visualization of TOL gridsearch data.",
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
        default=5,
        help="Number of top kendall_tau_mean rows to average (default: 5).",
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
# Extracting top-k averaged (alpha, beta) per file
# ---------------------------------------------------------------------------

def extract_top_k_coordinates(
    frames: list[pd.DataFrame],
    top_k: int = 5,
) -> list[tuple[float, float]]:
    """
    For each individual CSV file, find the top-k rows by ``kendall_tau_mean``,
    then average their ``alpha`` and ``beta`` values.

    Returns a list of (alpha, beta) tuples, one per file.
    """
    coords: list[tuple[float, float]] = []
    for df in frames:
        top_rows = df.nlargest(top_k, "kendall_tau_mean")
        avg_alpha = float(top_rows["alpha"].mean())
        avg_beta = float(top_rows["beta"].mean())
        coords.append((avg_alpha, avg_beta))

    return coords


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    group_coords: dict[str, list[tuple[float, float]]],
    output_folder: Path,
    plot_name: str,
    extension: str,
):
    """Create and save a log-log scatter plot of (alpha, beta) data."""
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

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("beta (log scale)")
    ax.set_title("Top-k Alpha/Beta – TOL Gridsearch Data")
    ax.legend(title="Group", fontsize="small", title_fontsize="medium")
    ax.grid(True, alpha=0.3, which="both")

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

    # Extract top-k averaged (alpha, beta) coordinates per file
    print(f"Extracting top-{args.top_k} averaged (alpha, beta) per file…")
    group_coords: dict[str, list[tuple[float, float]]] = {}
    for group_name in group_names:
        group_coords[group_name] = extract_top_k_coordinates(
            group_data[group_name], top_k=args.top_k
        )
        print(f"  {group_name}: {len(group_coords[group_name])} points")

    # Plot
    plot_results(
        group_coords,
        args.output_folder,
        args.plot_name,
        args.extension,
    )


if __name__ == "__main__":
    main()
