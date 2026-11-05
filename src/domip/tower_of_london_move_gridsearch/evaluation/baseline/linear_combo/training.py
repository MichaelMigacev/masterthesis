"""
Training script for linear combination of human and machine problem orderings.

Loads problem splits, human difficulty rankings, and machine rankings
from a directory to train scaling parameters via RankSVM.  For every
split and every human ranking file a separate RankSVM model is trained
so that the learned weight vector (one parameter per machine ranking)
produces a linear combination whose rank order matches the human
ranking as closely as possible (measured by Kendall-tau).

Output layout
-------------
{output_dir}/
    split_001/
        {human_ranking_1}.csv   <- parameters + score
        {human_ranking_2}.csv
        ...
    split_002/
        ...
    ...

This file contains code written by AI.
"""

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------------
# Standardization utilities
# ---------------------------------------------------------------------------

def standardize(ranking: Dict[str, float]) -> Dict[str, float]:
    """Apply z-score standardization: Y' = (Y - μY) / σY.

    Args:
        ranking: dict mapping TOL-ID to ranking value.

    Returns:
        New dict with standardized values.  If σY == 0 the original
        values are returned unchanged.
    """
    values = np.array(list(ranking.values()), dtype=float)
    mu = values.mean()
    sigma = values.std()

    if sigma == 0:
        return dict(ranking)

    standardized_values = (values - mu) / sigma
    return dict(zip(ranking.keys(), standardized_values))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_ranking_file(ranking_path: str) -> Dict[str, float]:
    """Load a single ranking CSV and return a dict mapping normalized ID to value.

    Expected format: first column is the TOL-ID (e.g. 'TOL-ID-01' or 'TOL-ID-1'),
    second column is the numeric ranking value.

    All IDs are normalized to zero-padded 2-digit strings (e.g. '01', '08')
    so that different file formats are compatible.
    """
    df = pd.read_csv(ranking_path)
    if df.shape[1] < 2:
        raise ValueError(
            f"Ranking file {ranking_path} must have at least 2 columns "
            f"(TOL-ID and value), got {df.shape[1]}"
        )
    id_col = df.columns[0]
    val_col = df.columns[1]
    raw_ids = df[id_col].astype(str)
    norm_ids = [normalize_id(raw) for raw in raw_ids]
    return dict(zip(norm_ids, df[val_col].astype(float)))


def load_rankings_from_dir(
    rankings_dir: str,
    standardize_flag: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Load all ranking CSVs from a directory.

    Args:
        rankings_dir: path to folder containing ranking CSV files.
        standardize_flag: if True, apply z-score standardization to each
            ranking before returning.

    Returns:
        Dict mapping CSV filename (without extension) to ranking dict
        {TOL-ID: value}.
    """
    rankings: Dict[str, Dict[str, float]] = {}
    csv_files = sorted(glob.glob(os.path.join(rankings_dir, "*.csv")))

    if not csv_files:
        raise FileNotFoundError(
            f"No .csv files found in rankings directory: {rankings_dir}"
        )

    for csv_path in csv_files:
        key = Path(csv_path).stem
        ranking = load_ranking_file(csv_path)
        if standardize_flag:
            ranking = standardize(ranking)
        rankings[key] = ranking

    return rankings


def normalize_id(tol_id: str) -> str:
    """Normalize a TOL-ID to a zero-padded numeric string (e.g. '01', '13').

    Handles formats like 'TOL-ID-1', 'TOL-ID-01', '01', '1'.
    """
    if tol_id.startswith("TOL-ID-"):
        tol_id = tol_id.replace("TOL-ID-", "")
    return f"{int(tol_id):02d}"


def ranking_to_array(
    ranking: Dict[str, float], problem_ids: List[str]
) -> Optional[np.ndarray]:
    """Convert a ranking dict to a numpy array aligned with given problem IDs.

    Returns None if any problem ID is missing from the ranking.
    """
    values = []
    for pid in problem_ids:
        norm = normalize_id(pid)
        val = ranking.get(norm)
        if val is None:
            val = ranking.get(f"TOL-ID-{norm}")
        if val is None:
            return None
        values.append(val)
    return np.array(values)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class TrainingData:
    """Holds all data needed for training."""
    splits: pd.DataFrame
    human_rankings: Dict[str, Dict[str, float]]
    mach_rankings: Dict[str, Dict[str, float]]


def load_all_data(
    splits_path: str,
    human_rankings_dir: str,
    mach_rankings_dir: str,
    standardize: bool = False,
) -> TrainingData:
    """Load all data sources and return a TrainingData object.

    Args:
        splits_path: path to the splits CSV file.
        human_rankings_dir: path to folder with human ranking CSVs.
        mach_rankings_dir: path to folder with machine ranking CSVs.
        standardize: if True, apply z-score standardization to machine
            rankings before returning.
    """
    splits = pd.read_csv(splits_path)
    splits["train_problems"] = splits["train_problems"].str.split()
    splits["validation_problems"] = splits["validation_problems"].str.split()

    human_rankings = load_rankings_from_dir(human_rankings_dir, standardize_flag=False)
    mach_rankings = load_rankings_from_dir(mach_rankings_dir, standardize_flag=standardize)

    return TrainingData(
        splits=splits,
        human_rankings=human_rankings,
        mach_rankings=mach_rankings,
    )


# ---------------------------------------------------------------------------
# RankSVM training
# ---------------------------------------------------------------------------

def _build_pairwise_instances(
    X: np.ndarray,
    human_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build pairwise training instances for RankSVM.

    For every unordered pair {i, j} we create TWO instances:
      - the correctly-ordered difference (label +1)
      - the incorrectly-ordered difference (label -1)

    The SVM learns a weight vector w such that w · (x_i - x_j) > 0
    whenever human[i] > human[j].

    Args:
        X: feature matrix, shape (n_problems, n_machine_rankings).
        human_values: human ranking values aligned with rows of X.

    Returns:
        (X_pair, y_pair) ready for LinearSVC.fit().
    """
    n = X.shape[0]
    pair_X, pair_y = [], []

    for i in range(n):
        for j in range(i + 1, n):
            if human_values[i] == human_values[j]:
                continue
            if human_values[i] > human_values[j]:
                # i should rank above j
                pair_X.append(X[i] - X[j])
                pair_y.append(1)
                pair_X.append(X[j] - X[i])
                pair_y.append(-1)
            else:
                # j should rank above i
                pair_X.append(X[j] - X[i])
                pair_y.append(1)
                pair_X.append(X[i] - X[j])
                pair_y.append(-1)

    return np.array(pair_X), np.array(pair_y)


def train_ranksvm_for_split(
    train_problems: List[str],
    mach_rankings: Dict[str, Dict[str, float]],
    human_ranking: Dict[str, float],
) -> Optional[Dict]:
    """Train a RankSVM model for one split and one human ranking.

    Args:
        train_problems: list of problem IDs in the training set.
        mach_rankings: dict mapping ranking-name → {TOL-ID: value}.
        human_ranking: {TOL-ID: value} for this human subject/file.

    Returns:
        Dict with keys:
            - weights: {ranking_name: parameter_value}
            - kendall_tau: float
            - n_pairs: int  (number of pairwise instances used)
        Returns None if data is insufficient (e.g. missing problems).
    """
    # --- filter rankings to training problems ---------------------------------
    norm_train = [normalize_id(pid) for pid in train_problems]

    # Build feature matrix: rows = problems, cols = machine rankings
    # All ranking dicts use normalized keys (e.g. '01', '08')
    mach_names = sorted(mach_rankings.keys())
    feature_rows: List[List[float]] = []

    for pid in norm_train:
        row = []
        for name in mach_names:
            val = mach_rankings[name].get(pid)
            if val is None:
                return None  # missing problem in a machine ranking
            row.append(val)
        feature_rows.append(row)

    X = np.array(feature_rows)

    # Human target values aligned with rows of X
    human_vals = []
    for pid in norm_train:
        val = human_ranking.get(pid)
        if val is None:
            return None  # missing problem in human ranking
        human_vals.append(val)
    human_vals = np.array(human_vals)

    # --- build pairwise instances --------------------------------------------
    X_pair, y_pair = _build_pairwise_instances(X, human_vals)

    if X_pair.shape[0] == 0:
        return None  # no useful pairs (all human values identical?)

    # --- train RankSVM (LinearSVC on pairwise differences) -------------------
    svm = LinearSVC(C=1.0, max_iter=10_000, random_state=42, dual=False)
    svm.fit(X_pair, y_pair)

    weights = svm.coef_.ravel()  # shape: (n_machine_rankings,)

    # --- evaluate: predictions vs. human ranking -----------------------------
    predictions = X.dot(weights)
    kendall, _ = kendalltau(predictions, human_vals)

    return {
        "weights": {name: float(w) for name, w in zip(mach_names, weights)},
        "kendall_tau": float(kendall),
        "n_pairs": int(X_pair.shape[0]),
    }


def save_split_results(
    output_dir: str,
    split_idx: int,
    results: Dict[str, Dict],
) -> None:
    """Save training results for one split.

    Creates {output_dir}/split_{idx:03d}/ and writes one CSV per
    human ranking containing the learned parameters and Kendall-tau.

    Args:
        output_dir: root output directory.
        split_idx: zero-based split index (used for folder naming).
        results: dict mapping human_ranking_name → result dict
                 (as returned by train_ranksvm_for_split).
    """
    split_dir = os.path.join(output_dir, f"split_{split_idx:03d}")
    os.makedirs(split_dir, exist_ok=True)

    for human_name, result in results.items():
        rows = []
        for ranking_name, weight in result["weights"].items():
            rows.append({
                "machine_ranking": ranking_name,
                "parameter": weight,
            })
        df = pd.DataFrame(rows)
        df["kendall_tau"] = result["kendall_tau"]
        df["n_pairs"] = result["n_pairs"]
        df.to_csv(
            os.path.join(split_dir, f"{human_name}.csv"),
            index=False,
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RankSVM for linear combination of "
                    "human and machine problem orderings."
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
        "--output-dir",
        type=str,
        required=True,
        help="Root directory for output.  One sub-folder per split "
             "(split_001/, split_002/, …), one CSV per human ranking "
             "inside each.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        default=False,
        help="Apply z-score standardization (Y'=(Y-μY)/σY) to machine "
             "rankings before training.",
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
        standardize=args.standardize,
    )

    print(f"Loaded {len(data.splits)} splits")
    print(f"Loaded {len(data.human_rankings)} human ranking(s): "
          f"{list(data.human_rankings.keys())}")
    print(f"Loaded {len(data.mach_rankings)} machine ranking(s): "
          f"{list(data.mach_rankings.keys())}")
    print(f"Output directory: {args.output_dir}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    for split_idx, split_row in data.splits.iterrows():
        train_problems = split_row["train_problems"]
        split_label = f"split_{split_idx:03d}"
        print(f"[{split_label}]  training problems: {len(train_problems)}")

        results: Dict[str, Dict] = {}

        for human_name, human_ranking in data.human_rankings.items():
            result = train_ranksvm_for_split(
                train_problems=train_problems,
                mach_rankings=data.mach_rankings,
                human_ranking=human_ranking,
            )

            if result is not None:
                results[human_name] = result
                print(f"  {human_name:30s}  "
                      f"kendall_tau={result['kendall_tau']:.4f}  "
                      f"n_pairs={result['n_pairs']}")
            else:
                print(f"  {human_name:30s}  SKIPPED (missing data)")

        if results:
            save_split_results(args.output_dir, split_idx, results)

    print(f"\nDone.  Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
