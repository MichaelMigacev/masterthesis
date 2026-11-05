"""Generate heatmap PNGs from gridsearch result CSVs.

Each input CSV has columns: alpha, beta, kendall_tau_mean, kendall_tau_std, source_file.
The script pivots alpha/beta into a 2D grid of kendall_tau_mean values and plots it
as a heatmap using the viridis colormap (dark = low, bright = high).

This file contains code written by AI.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Create heatmap PNGs from gridsearch CSV files."
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

    for csv_file in csv_files:
        csv_path = os.path.join(args.input_folder, csv_file)
        base_name = os.path.splitext(csv_file)[0]
        out_name = f"{args.prefix}_{base_name}.png"
        out_path = os.path.join(args.output_folder, out_name)

        # ------------------------------------------------------------------
        # Read data
        # ------------------------------------------------------------------
        alphas = []
        betas = []
        taus = []

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                alphas.append(float(row["alpha"]))
                betas.append(float(row["beta"]))
                taus.append(float(row["kendall_tau_mean"]))

        unique_alphas = sorted(set(alphas))
        unique_betas = sorted(set(betas))
        n_alpha = len(unique_alphas)
        n_beta = len(unique_betas)

        # ------------------------------------------------------------------
        # Build 2D grid (rows = alpha, cols = beta)
        # ------------------------------------------------------------------
        grid = np.full((n_alpha, n_beta), np.nan)

        alpha_idx = {a: i for i, a in enumerate(unique_alphas)}
        beta_idx = {b: j for j, b in enumerate(unique_betas)}

        for a, b, t in zip(alphas, betas, taus):
            grid[alpha_idx[a], beta_idx[b]] = t

        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(
            grid,
            cmap="viridis",
            aspect="auto",
            interpolation="nearest",
        )

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("kendall_tau_mean", rotation=270, labelpad=15)

        # Tick labels
        ax.set_xticks(list(range(n_beta)))
        ax.set_xticklabels([f"{b:.2f}" if b < 10 else f"{b:g}" for b in unique_betas], rotation=45, ha="right")
        ax.set_yticks(list(range(n_alpha)))
        ax.set_yticklabels([f"{a:.2f}" if a < 10 else f"{a:g}" for a in unique_alphas])

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
        ax.set_title(base_name)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Wrote {out_path}")

    print(f"\nWrote {len(csv_files)} heatmap(s) to {args.output_folder}")


if __name__ == "__main__":
    main()
