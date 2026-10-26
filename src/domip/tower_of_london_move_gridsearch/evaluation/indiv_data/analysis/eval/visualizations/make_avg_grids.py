"""Generate average heatmap PNGs from multiple gridsearch result CSVs.

Each input CSV has columns: alpha, beta, kendall_tau_mean, kendall_tau_std, source_file.
The script aggregates all CSVs by (alpha, beta) pairs, computes the mean and standard
deviation of kendall_tau_mean across files, and produces:

  - average.csv  in the input folder
      columns: alpha, beta, avg_kendall_tau_mean, kendall_tau_mean_std
  - two heatmap PNGs in the output folder:
      one for avg_kendall_tau_mean and one for kendall_tau_mean_std
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def _plot_grid(grid, unique_alphas, unique_betas, title, colorbar_label,
               out_path, prefix):
    """Helper to create a single heatmap PNG from a 2D grid."""
    n_alpha = len(unique_alphas)
    n_beta = len(unique_betas)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        grid,
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(colorbar_label, rotation=270, labelpad=15)

    # Tick labels
    ax.set_xticks(list(range(n_beta)))
    ax.set_xticklabels(
        [f"{b:.2f}" if b < 10 else f"{b:g}" for b in unique_betas],
        rotation=45, ha="right",
    )
    ax.set_yticks(list(range(n_alpha)))
    ax.set_yticklabels(
        [f"{a:.2f}" if a < 10 else f"{a:g}" for a in unique_alphas],
    )

    # Annotate each cell with the value (rounded to 2 decimals)
    for i in range(n_alpha):
        for j in range(n_beta):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=6,
                    color="white",
                )

    ax.set_xlabel("beta")
    ax.set_ylabel("alpha")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create average heatmap PNGs from multiple gridsearch CSVs."
    )
    parser.add_argument(
        "--input_folder",
        help="Folder containing gridsearch result CSVs.",
    )
    parser.add_argument(
        "--output_folder",
        help="Folder where the heatmap PNGs will be written.",
    )
    parser.add_argument(
        "--prefix",
        help="Prefix for the output PNG filenames.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    csv_files = sorted(
        f for f in os.listdir(args.input_folder) if f.endswith(".csv")
    )
    if not csv_files:
        print("Error: no CSV files found in input folder.", flush=True)
        return

    # ------------------------------------------------------------------
    # Read all CSVs and collect kendall_tau_mean per (alpha, beta)
    # ------------------------------------------------------------------
    # Dict: (alpha, beta) -> list of kendall_tau_mean values
    tau_accum: dict[tuple[float, float], list[float]] = {}

    for csv_file in csv_files:
        csv_path = os.path.join(args.input_folder, csv_file)

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                alpha = float(row["alpha"])
                beta = float(row["beta"])
                tau_mean = float(row["kendall_tau_mean"])
                key = (alpha, beta)
                tau_accum.setdefault(key, []).append(tau_mean)

    # ------------------------------------------------------------------
    # Compute avg_kendall_tau_mean and kendall_tau_mean_std
    # ------------------------------------------------------------------
    unique_alphas = sorted({k[0] for k in tau_accum})
    unique_betas = sorted({k[1] for k in tau_accum})
    n_alpha = len(unique_alphas)
    n_beta = len(unique_betas)

    alpha_idx = {a: i for i, a in enumerate(unique_alphas)}
    beta_idx = {b: j for j, b in enumerate(unique_betas)}

    grid_mean = np.full((n_alpha, n_beta), np.nan)
    grid_std = np.full((n_alpha, n_beta), np.nan)

    for (alpha, beta), taus in tau_accum.items():
        i, j = alpha_idx[alpha], beta_idx[beta]
        grid_mean[i, j] = np.mean(taus)
        grid_std[i, j] = np.std(taus, ddof=0)  # population std

    # ------------------------------------------------------------------
    # Write average.csv into the input folder
    # ------------------------------------------------------------------
    avg_csv_path = os.path.join(args.input_folder, "average.csv")
    with open(avg_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["alpha", "beta", "avg_kendall_tau_mean",
                         "kendall_tau_mean_std"])
        for i, alpha in enumerate(unique_alphas):
            for j, beta in enumerate(unique_betas):
                writer.writerow([alpha, beta, grid_mean[i, j], grid_std[i, j]])

    print(f"Wrote {avg_csv_path}")

    # ------------------------------------------------------------------
    # Plot avg_kendall_tau_mean grid
    # ------------------------------------------------------------------
    out_mean_path = os.path.join(args.output_folder, f"{args.prefix}_avg_kendall_tau_mean.png")
    _plot_grid(
        grid=grid_mean,
        unique_alphas=unique_alphas,
        unique_betas=unique_betas,
        title="avg_kendall_tau_mean",
        colorbar_label="avg_kendall_tau_mean",
        out_path=out_mean_path,
        prefix=args.prefix,
    )

    # ------------------------------------------------------------------
    # Plot kendall_tau_mean_std grid
    # ------------------------------------------------------------------
    out_std_path = os.path.join(args.output_folder, f"{args.prefix}_kendall_tau_mean_std.png")
    _plot_grid(
        grid=grid_std,
        unique_alphas=unique_alphas,
        unique_betas=unique_betas,
        title="kendall_tau_mean_std",
        colorbar_label="kendall_tau_mean_std",
        out_path=out_std_path,
        prefix=args.prefix,
    )

    print(f"\nWrote 2 heatmap(s) to {args.output_folder}")
    print(f"  - {out_mean_path}")
    print(f"  - {out_std_path}")


if __name__ == "__main__":
    main()
