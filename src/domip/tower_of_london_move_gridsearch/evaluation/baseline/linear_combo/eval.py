"""
Aggregation script for testing results.

Reads all ``*_tau_results.csv`` files produced by ``testing.py`` across every
split, aggregates the per-split Kendall-tau means per human subject, and
writes a single summary CSV with 8 rows (one per human) and 2 columns
(mean and standard deviation of the per-split means).

Output layout
-------------
{output_dir}/
    summary.csv          <- human_file | mean | std
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def collect_tau_results(testing_output_dir: str) -> Dict[str, List[float]]:
    """Walk all ``split_*`` folders and collect per-human mean-tau values.

    Args:
        testing_output_dir: root directory produced by ``testing.py``
            (contains ``split_000/``, ``split_001/``, …).

    Returns:
        Dict mapping human ranking name → list of mean-tau values
        (one value per split).
    """
    human_means: Dict[str, List[float]] = {}

    if not os.path.isdir(testing_output_dir):
        raise FileNotFoundError(
            f"Testing output directory not found: {testing_output_dir}"
        )

    # Find all split_* directories
    split_dirs = sorted(
        d for d in os.listdir(testing_output_dir)
        if d.startswith("split_") and os.path.isdir(
            os.path.join(testing_output_dir, d)
        )
    )

    if not split_dirs:
        raise FileNotFoundError(
            f"No split_* directories found in: {testing_output_dir}"
        )

    for split_dir_name in split_dirs:
        split_path = os.path.join(testing_output_dir, split_dir_name)

        # Find all *_tau_results.csv files in this split
        tau_files = sorted(
            f for f in os.listdir(split_path)
            if f.endswith("_tau_results.csv")
        )

        for tau_file in tau_files:
            # Extract human name: "human_1_tau_results.csv" -> "human_1"
            human_name = Path(tau_file).stem.replace("_tau_results", "")
            csv_path = os.path.join(split_path, tau_file)

            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"  WARNING: {csv_path} is empty, skipping.")
                continue

            mean_tau = float(df["mean"].iloc[0])
            if np.isnan(mean_tau):
                print(f"  WARNING: {csv_path} has NaN mean_tau, skipping.")
                continue
            human_means.setdefault(human_name, []).append(mean_tau)

    return human_means


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    human_means: Dict[str, List[float]],
) -> pd.DataFrame:
    """Compute overall mean and std of per-split means for each human.

    Args:
        human_means: dict mapping human_name → list of mean-tau values.

    Returns:
        DataFrame with columns ``human_file``, ``mean``, ``std``,
        sorted by human_file.
    """
    rows: List[dict] = []

    for human_name in sorted(human_means.keys()):
        values = human_means[human_name]
        rows.append({
            "human_file": f"{human_name}.csv",
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-split Kendall-tau results from "
                    "testing.py into a single summary CSV."
    )

    parser.add_argument(
        "--testing-output-dir",
        type=str,
        required=True,
        help="Root directory produced by testing.py "
             "(contains split_000/, split_001/, … with *_tau_results.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the aggregated summary.csv into.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Reading testing results from: {args.testing_output_dir}")
    human_means = collect_tau_results(args.testing_output_dir)

    print(f"Found {len(human_means)} human ranking(s): "
          f"{sorted(human_means.keys())}")
    for human_name, means in sorted(human_means.items()):
        print(f"  {human_name}: {len(means)} split(s)")

    summary_df = aggregate_results(human_means)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(output_path, index=False)

    print(f"\nAggregated summary written to: {output_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
