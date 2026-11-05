"""
Dimensionality reduction visualization for Tower of London ranking data.

Reads CSV files from subfolders, sorts each by TOL-ID label (1–24),
applies PCA or t-SNE, and produces a colored scatter plot where each
subfolder gets its own color and legend entry.

This file contains code written by AI.
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA/t-SNE visualization of TOL ranking data."
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
        "--method",
        type=str,
        choices=["pca", "tsne"],
        default="pca",
        help="Dimensionality reduction method: 'pca' or 'tsne' (default: pca).",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="png",
        help="File extension for the saved plot (default: png).",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of components for the reduction (default: 2).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42).",
    )
    return parser.parse_args()


def extract_label_number(label: str) -> int:
    """Extract the numeric part from a label like 'TOL-ID-12' -> 12."""
    return int(label.split("-")[-1])


def load_subfolder_data(subfolder_path: Path) -> list[np.ndarray]:
    """
    Load all .csv files from a single subfolder.

    Each CSV is expected to have a 'TOL-ID' column (labels 1–24) and a
    numeric data column.  Rows are sorted by label number so the 24
    values are in the correct order.

    Returns a list of 1-D arrays, one per CSV file.
    """
    samples: list[np.ndarray] = []
    csv_files = sorted(subfolder_path.glob("*.csv"), key=lambda p: p.stem)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Sort by the numeric part of TOL-ID so values are in order 1–24
        df["label_num"] = df.iloc[:, 0].apply(extract_label_number)
        df = df.sort_values("label_num")

        # Take the data column (second column) as the feature vector
        data = df.iloc[:, 1].values.astype(float)
        samples.append(data)

    return samples


def load_all_data(input_folder: Path) -> tuple[np.ndarray, list[str]]:
    """
    Walk through all subfolders of *input_folder*, load CSV data, and
    return a stacked feature matrix and a parallel list of group labels
    (the subfolder name for each sample).
    """
    all_samples: list[np.ndarray] = []
    group_labels: list[str] = []

    subfolders = sorted(
        [d for d in input_folder.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    for subfolder in subfolders:
        samples = load_subfolder_data(subfolder)
        group_name = subfolder.name
        all_samples.extend(samples)
        group_labels.extend([group_name] * len(samples))

    if not all_samples:
        raise FileNotFoundError(
            f"No .csv files found in any subfolder of '{input_folder}'."
        )

    X = np.vstack(all_samples)
    return X, group_labels


def reduce_dimensions(
    X: np.ndarray,
    method: str,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    """Apply PCA or t-SNE dimensionality reduction."""
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            init="pca",
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(X)


def plot_results(
    X_red: np.ndarray,
    group_labels: list[str],
    output_folder: Path,
    plot_name: str,
    method: str,
    extension: str,
):
    """Create and save a scatter plot of the reduced data."""
    groups = sorted(set(group_labels))
    n_groups = len(groups)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a qualitative colormap with enough distinct colors
    cmap = matplotlib.colormaps["Dark2"] if n_groups <= 20 else matplotlib.colormaps["tab10"]
    colors = [cmap(i % cmap.N) for i in range(n_groups)]
    group_to_color = dict(zip(groups, colors))

    for i, group in enumerate(groups):
        mask = np.array(group_labels) == group
        ax.scatter(
            X_red[mask, 0],
            X_red[mask, 1],
            label=group,
            color=group_to_color[group],
            alpha=0.7,
            edgecolors="k",
            s=40,
        )

    ax.set_xlabel(f"{method.upper()} component 1")
    ax.set_ylabel(f"{method.upper()} component 2")
    ax.set_title(f"{method.upper()} – TOL Ranking Data")
    ax.legend(title="Group", fontsize="small", title_fontsize="medium")
    ax.grid(True, alpha=0.3)

    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = output_folder / f"{plot_name}.{extension}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


def main():
    args = parse_args()

    print(f"Loading data from: {args.input_folder}")
    X, group_labels = load_all_data(args.input_folder)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features "
          f"across {len(set(group_labels))} groups.")

    print(f"Applying {args.method.upper()} reduction…")
    X_red = reduce_dimensions(
        X,
        method=args.method,
        n_components=args.n_components,
        random_state=args.random_state,
    )

    plot_results(
        X_red,
        group_labels,
        args.output_folder,
        args.plot_name,
        args.method,
        args.extension,
    )


if __name__ == "__main__":
    main()
