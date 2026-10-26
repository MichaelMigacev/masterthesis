"""
Dimensionality reduction visualization for Tower of London gridsearch data.

Reads CSV files from subfolders, extracts the 'kendall_tau_mean' column
(rows kept in their natural file order), applies PCA or t-SNE, and
produces a colored scatter plot where each subfolder gets its own color
and legend entry.
"""

import argparse
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
        description="PCA/t-SNE visualization of TOL gridsearch data."
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
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Within each CSV, keep only the top k largest kendall_tau_mean values (default: None, keep all).",
    )
    return parser.parse_args()


def load_subfolder_data(
    subfolder_path: Path, top_k: int | None = None
) -> list[np.ndarray]:
    """
    Load all .csv files from a single subfolder.

    Each CSV is expected to have at least a 'kendall_tau_mean' column.
    Rows are kept in their natural file order (alpha/beta grid order is
    assumed to be identical across all files).

    If *top_k* is set, only the top_k largest kendall_tau_mean values
    from each file are retained.

    Returns a list of 1-D arrays, one per CSV file.
    """
    samples: list[np.ndarray] = []
    csv_files = sorted(subfolder_path.glob("*.csv"), key=lambda p: p.stem)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Extract kendall_tau_mean values in file order
        data = df["kendall_tau_mean"].values.astype(float)

        if top_k is not None:
            # Keep only the top_k largest values
            data = np.sort(data)[::-1][:top_k]

        samples.append(data)

    return samples


def load_all_data(
    input_folder: Path, top_k: int | None = None
) -> tuple[np.ndarray, list[str]]:
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
        samples = load_subfolder_data(subfolder, top_k=top_k)
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


def _draw_marginal_bars(
    ax: matplotlib.axes.Axes,
    data: np.ndarray,
    groups: list[str],
    group_labels: list[str],
    group_to_color: dict[str, tuple],
    n_bins: int = 20,
    vertical: bool = True,
):
    """
    Draw a stacked bar histogram on *ax* showing per-class distributions.

    Groups are sorted by total count descending so that larger bars are
    drawn behind smaller ones for full visibility.
    """
    bins = np.linspace(data.min() - 1e-9, data.max() + 1e-9, n_bins + 1)
    # Sort groups by total count descending (biggest drawn first = in back)
    group_counts = [(g, sum(gl == g for gl in group_labels)) for g in groups]
    group_counts.sort(key=lambda tc: tc[1], reverse=True)

    for group, _ in group_counts:
        mask = np.array(group_labels) == group
        counts, bin_edges = np.histogram(data[mask], bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = bin_edges[1] - bin_edges[0]

        if vertical:
            ax.bar(
                bin_centers,
                counts,
                width=bin_widths,
                color=group_to_color[group],
                alpha=0.7,
                edgecolor="k",
                linewidth=0.5,
            )
        else:
            ax.barh(
                bin_centers,
                counts,
                height=bin_widths,
                color=group_to_color[group],
                alpha=0.7,
                edgecolor="k",
                linewidth=0.5,
            )

    ax.tick_params(labelsize="small")


def plot_results(
    X_red: np.ndarray,
    group_labels: list[str],
    output_folder: Path,
    plot_name: str,
    method: str,
    extension: str,
):
    """Create and save a scatter plot with marginal class distributions."""
    groups = sorted(set(group_labels))
    n_groups = len(groups)

    # Use a qualitative colormap with enough distinct colors
    cmap = (
        matplotlib.colormaps["Dark2"]
        if n_groups <= 20
        else matplotlib.colormaps["tab10"]
    )
    colors = [cmap(i % cmap.N) for i in range(n_groups)]
    group_to_color = dict(zip(groups, colors))

    # Layout: main scatter + top marginal (x) + right marginal (y)
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[1, 4, 0.3],
        width_ratios=[4, 1, 0.3],
        hspace=0.08,
        wspace=0.08,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # --- Main scatter plot ---
    for group in groups:
        mask = np.array(group_labels) == group
        ax_main.scatter(
            X_red[mask, 0],
            X_red[mask, 1],
            label=group,
            color=group_to_color[group],
            alpha=0.7,
            edgecolors="k",
            s=40,
        )

    ax_main.set_xlabel(f"{method.upper()} component 1")
    ax_main.set_ylabel(f"{method.upper()} component 2")
    ax_main.set_title(f"{method.upper()} – TOL Gridsearch Data")
    ax_main.legend(title="Group", fontsize="small", title_fontsize="medium")
    ax_main.grid(True, alpha=0.3)

    # --- Top marginal: component 1 distribution ---
    _draw_marginal_bars(
        ax_top,
        X_red[:, 0],
        groups,
        group_labels,
        group_to_color,
        vertical=True,
    )
    ax_top.set_ylabel("Count")
    ax_top.set_xlabel("")

    # --- Right marginal: component 2 distribution ---
    _draw_marginal_bars(
        ax_right,
        X_red[:, 1],
        groups,
        group_labels,
        group_to_color,
        vertical=False,
    )
    ax_right.set_xlabel("Count")
    ax_right.set_ylabel("")

    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = output_folder / f"{plot_name}.{extension}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


def main():
    args = parse_args()

    print(f"Loading data from: {args.input_folder}")
    X, group_labels = load_all_data(args.input_folder, top_k=args.top_k)
    print(
        f"Loaded {X.shape[0]} samples with {X.shape[1]} features "
        f"across {len(set(group_labels))} groups."
    )

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
