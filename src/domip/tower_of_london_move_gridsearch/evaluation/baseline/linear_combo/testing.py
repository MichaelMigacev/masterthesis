"""
Testing script for linear combination of human and machine problem orderings.

Loads the same splits, human rankings, and machine rankings as training.py,
plus the training results (learned parameters) produced by training.py.
For each split and each human ranking one parameter set is loaded, so that
machine rankings can be scaled with the learned weights during validation.

This file contains code written by AI.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from training import (
    load_rankings_from_dir,
    normalize_id,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_splits(splits_path: str) -> pd.DataFrame:
    """Load problem splits CSV.

    Expected columns: train_problems, validation_problems
    Problem IDs are space-separated strings like '01 02 03 ...'.
    """
    df = pd.read_csv(splits_path)
    df["train_problems"] = df["train_problems"].str.split()
    df["validation_problems"] = df["validation_problems"].str.split()
    return df


def load_training_results(
    training_results_dir: str,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Load all training result CSVs from the output directory of training.py.

    Expected layout:
        {training_results_dir}/
            split_000/
                {human_ranking_1}.csv
                {human_ranking_2}.csv
                ...
            split_001/
                ...

    Each CSV has columns: machine_ranking, parameter, kendall_tau, n_pairs

    Returns:
        Nested dict: {split_idx: {human_ranking_name: {machine_ranking_name: parameter_value}}}
    """
    results: Dict[int, Dict[str, Dict[str, float]]] = {}

    if not os.path.isdir(training_results_dir):
        raise FileNotFoundError(
            f"Training results directory not found: {training_results_dir}"
        )

    # Find all split_* directories
    split_dirs = sorted(
        d for d in os.listdir(training_results_dir)
        if d.startswith("split_") and os.path.isdir(
            os.path.join(training_results_dir, d)
        )
    )

    for split_dir_name in split_dirs:
        # Extract split index from directory name (e.g. "split_000" -> 0)
        split_idx = int(split_dir_name.replace("split_", ""))
        split_path = os.path.join(training_results_dir, split_dir_name)

        human_results: Dict[str, Dict[str, float]] = {}

        # Load each human ranking CSV in this split folder
        csv_files = sorted(
            f for f in os.listdir(split_path) if f.endswith(".csv")
        )

        for csv_file in csv_files:
            human_name = Path(csv_file).stem
            csv_path = os.path.join(split_path, csv_file)

            df = pd.read_csv(csv_path)
            param_dict = dict(zip(df["machine_ranking"], df["parameter"]))
            human_results[human_name] = param_dict

        results[split_idx] = human_results

    return results


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class TestingData:
    """Holds all data needed for testing/validation."""
    splits: pd.DataFrame
    human_rankings: Dict[str, Dict[str, float]]
    mach_rankings: Dict[str, Dict[str, float]]
    training_results: Dict[int, Dict[str, Dict[str, float]]]


def load_all_data(
    splits_path: str,
    human_rankings_dir: str,
    mach_rankings_dir: str,
    training_results_dir: str,
    standardize: bool = False,
) -> TestingData:
    """Load all data sources and return a TestingData object.

    Args:
        splits_path: path to the splits CSV file.
        human_rankings_dir: path to folder with human ranking CSVs.
        mach_rankings_dir: path to folder with machine ranking CSVs.
        training_results_dir: path to the root output directory from
            training.py (contains split_000/, split_001/, …).
        standardize: if True, apply z-score standardization to machine
            rankings before returning (must match what was used during
            training).

    Returns:
        TestingData object with all loaded data.
    """
    splits = load_splits(splits_path)
    human_rankings = load_rankings_from_dir(human_rankings_dir, standardize_flag=False)
    mach_rankings = load_rankings_from_dir(mach_rankings_dir, standardize_flag=standardize)
    training_results = load_training_results(training_results_dir)

    return TestingData(
        splits=splits,
        human_rankings=human_rankings,
        mach_rankings=mach_rankings,
        training_results=training_results,
    )


# ---------------------------------------------------------------------------
# Monte Carlo Kendall tau computation
# ---------------------------------------------------------------------------

def random_tie_resolved_order(
    df: pd.DataFrame, value_col: str, rng: np.random.Generator
) -> np.ndarray:
    """Resolve ties in *value_col* by randomly shuffling within each tie group.

    Returns an array of rank positions (1-based) aligned with rows of *df*.
    """
    work = df[["TOL-ID", value_col]].copy()
    work["_sort_value"] = work[value_col].fillna(np.inf)

    ordered_ids: list[str] = []
    for _, tie_group in work.groupby("_sort_value", sort=True, dropna=False):
        ids = tie_group["TOL-ID"].to_numpy(copy=True)
        rng.shuffle(ids)
        ordered_ids.extend(ids.tolist())

    rank_map = {tol_id: rank for rank, tol_id in enumerate(ordered_ids, start=1)}
    return df["TOL-ID"].map(rank_map).to_numpy()


def compute_monte_carlo_tau(
    ordered_df: pd.DataFrame,
    metric_col: str,
    permutation_runs: int,
    seed: int,
) -> Tuple[float, float]:
    """Compute mean and std of Kendall tau over random tie-breaking permutations.

    Args:
        ordered_df: DataFrame with columns ``TOL-ID``, ``metric``, ``human_rank``.
        metric_col: name of the metric column to resolve ties on.
        permutation_runs: number of Monte Carlo tie-breaking runs.
        seed: random seed.

    Returns:
        (mean_tau, std_tau)
    """
    # Check if metric has zero variance (all values identical).
    # In that case Kendall tau is undefined (nan) for every permutation.
    if ordered_df[metric_col].nunique(dropna=False) <= 1:
        print(
            f"  WARNING: metric column '{metric_col}' has only "
            f"{ordered_df[metric_col].nunique(dropna=False)} unique value(s). "
            f"Kendall tau is undefined; returning (0.0, 0.0)."
        )
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    tau_values: list[float] = []

    for _ in range(permutation_runs):
        randomized_positions = random_tie_resolved_order(ordered_df, metric_col, rng)
        tau, _ = kendalltau(
            randomized_positions, ordered_df["human_rank"].to_numpy()
        )
        tau_val = float(tau)
        if not np.isnan(tau_val):
            tau_values.append(tau_val)

    if len(tau_values) == 0:
        print(
            f"  WARNING: all {permutation_runs} tau values were NaN "
            f"(rank correlation undefined). Returning (0.0, 0.0)."
        )
        return 0.0, 0.0
    if len(tau_values) < permutation_runs:
        print(
            f"  WARNING: {permutation_runs - len(tau_values)}/{permutation_runs} "
            f"tau values were NaN and were dropped."
        )

    return float(np.mean(tau_values)), float(np.std(tau_values))


# ---------------------------------------------------------------------------
# Testing loop
# ---------------------------------------------------------------------------

def run_testing(
    data: TestingData,
    output_dir: str,
    permutation_runs: int = 1000,
    seed: int = 42,
) -> None:
    """Run the full testing loop over all splits.

    For each split the learned weights (from the matching training split) are
    applied to the machine rankings on the **validation** problems.  The
    resulting combined metric is compared to the human ranking via Kendall tau
    with Monte Carlo tie-breaking.

    Output layout (mirrors ``po_pipeline_pp_mult_val.py``):

        {output_dir}/
            split_000/
                {human_ranking_1}_tau_results.csv
                {human_ranking_2}_tau_results.csv
                summary.csv
            split_001/
                ...
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_idx, split_row in data.splits.iterrows():
        validation_problems = split_row["validation_problems"]
        norm_val = [normalize_id(pid) for pid in validation_problems]
        split_label = f"split_{split_idx:03d}"
        split_output_dir = os.path.join(output_dir, split_label)
        os.makedirs(split_output_dir, exist_ok=True)

        print(f"\n--- Testing {split_label} ({len(validation_problems)} problems) ---")

        results: List[Dict] = []
        processed = 0

        for human_name, human_ranking in data.human_rankings.items():
            # --- load learned weights for this split + human ranking ---------
            human_weights = data.training_results[split_idx][human_name]

            # --- build combined metric for validation problems ---------------
            metric_values: List[float] = []
            for pid in norm_val:
                combined = 0.0
                valid = True
                for mach_name, weight in human_weights.items():
                    val = data.mach_rankings[mach_name].get(pid)
                    if val is None:
                        valid = False
                        break
                    combined += weight * val
                metric_values.append(combined if valid else float("nan"))

            # --- attach human ranking values ---------------------------------
            human_values: List[float] = []
            for pid in norm_val:
                val = human_ranking.get(pid)
                if val is None:
                    val = human_ranking.get(f"TOL-ID-{pid}")
                human_values.append(val if val is not None else float("nan"))

            df = pd.DataFrame({
                "TOL-ID": norm_val,
                "metric": metric_values,
                "human_rank": human_values,
            })

            # --- compute Kendall tau with Monte Carlo tie-breaking -----------
            mean_tau, std_tau = compute_monte_carlo_tau(
                df,
                metric_col="metric",
                permutation_runs=permutation_runs,
                seed=seed,
            )

            # --- save per-human result CSV -----------------------------------
            output_csv = os.path.join(
                split_output_dir, f"{human_name}_tau_results.csv"
            )
            pd.DataFrame({
                "metric": ["kendall_tau"],
                "mean": [mean_tau],
                "std": [std_tau],
            }).to_csv(output_csv, index=False)

            results.append({
                "human_file": f"{human_name}.csv",
                "mean_tau": mean_tau,
                "std_tau": std_tau,
                "output_file": f"{human_name}_tau_results.csv",
            })
            processed += 1
            print(
                f"  {human_name}: mean_tau={mean_tau:.4f}, std_tau={std_tau:.4f}"
            )

        if processed == 0:
            raise RuntimeError(
                f"No valid human rankings were processed for {split_label}."
            )

        # --- save summary CSV for this split ---------------------------------
        summary_df = pd.DataFrame(results)
        summary_csv = os.path.join(split_output_dir, "summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"  Saved summary: summary.csv")

    print(f"\nDone.  Results written to: {output_dir}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test/validate linear combination of human and machine "
                    "problem orderings using trained parameters."
    )

    parser.add_argument(
        "--splits",
        type=str,
        required=True,
        help="Path to the splits CSV file.",
    )
    parser.add_argument(
        "--human-rankings-dir",
        type=str,
        required=True,
        help="Path to the folder containing human ranking CSV files.",
    )
    parser.add_argument(
        "--mach-rankings-dir",
        type=str,
        required=True,
        help="Path to the folder containing machine ranking CSV files.",
    )
    parser.add_argument(
        "--training-results-dir",
        type=str,
        required=True,
        help="Path to the root output directory from training.py "
             "(contains split_000/, split_001/, … with parameter CSVs).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory for output.  One sub-folder per split "
             "(split_000/, split_001/, …), one CSV per human ranking "
             "inside each.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        default=False,
        help="Apply z-score standardization (Y'=(Y-μY)/σY) to machine "
             "rankings. Must match the setting used during training.",
    )
    parser.add_argument(
        "--permutation-runs",
        type=int,
        default=1000,
        help="Number of random tie-resolution permutations for Kendall tau.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the Monte Carlo tie-breaking.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    data = load_all_data(
        splits_path=args.splits,
        human_rankings_dir=args.human_rankings_dir,
        mach_rankings_dir=args.mach_rankings_dir,
        training_results_dir=args.training_results_dir,
        standardize=args.standardize,
    )

    print(f"Loaded {len(data.splits)} splits")
    print(f"Loaded {len(data.human_rankings)} human ranking(s): "
          f"{list(data.human_rankings.keys())}")
    print(f"Loaded {len(data.mach_rankings)} machine ranking(s): "
          f"{list(data.mach_rankings.keys())}")
    print(f"Loaded training results for {len(data.training_results)} splits")

    # Verify alignment: every split should have results for every human ranking
    expected_human_names = set(data.human_rankings.keys())
    for split_idx, human_results in data.training_results.items():
        actual_human_names = set(human_results.keys())
        if actual_human_names != expected_human_names:
            print(f"WARNING: split_{split_idx:03d} human rankings mismatch!")
            print(f"  Expected: {expected_human_names}")
            print(f"  Got:      {actual_human_names}")

    # Verify alignment: every result should have parameters for every machine ranking
    expected_mach_names = set(data.mach_rankings.keys())
    for split_idx, human_results in data.training_results.items():
        for human_name, param_dict in human_results.items():
            actual_mach_names = set(param_dict.keys())
            if actual_mach_names != expected_mach_names:
                print(
                    f"WARNING: split_{split_idx:03d}/{human_name} "
                    f"machine ranking mismatch!"
                )
                print(f"  Expected: {expected_mach_names}")
                print(f"  Got:      {actual_mach_names}")

    print("\nAll data loaded successfully. Running testing loop.")

    # -----------------------------------------------------------------------
    # Testing loop
    # -----------------------------------------------------------------------
    run_testing(
        data=data,
        output_dir=args.output_dir,
        permutation_runs=args.permutation_runs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
