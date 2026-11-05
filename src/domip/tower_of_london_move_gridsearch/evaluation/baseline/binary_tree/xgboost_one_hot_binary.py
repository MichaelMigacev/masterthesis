"""
Binary XGBoost classifier for Tower of London problem difficulty comparison.

Trains a binary classifier that receives a pair of problems and predicts
whether problem 1 is harder than problem 2 (label=1) or not (label=0).
Labels are derived from human difficulty scores: if score(p1) > score(p2) then 1,
otherwise 0.

Training loop:
    - Randomly sample pairs from the training set
    - Build pair features (concatenation of both problem vectors)
    - Label based on human ranking scores
    - Train XGBClassifier with max_depth=4

Evaluation:
    - Predict pairwise comparisons on validation set
    - Derive a ranking from pairwise predictions
    - Compare to human ranking via Kendall tau with Monte Carlo tie-breaking

Output layout (mirrors linear_combo/testing.py):
    {output_dir}/
        split_001/
            {human_ranking_1}_tau_results.csv
            {human_ranking_2}_tau_results.csv
            summary.csv
        split_002/
            ...

This file contains code written by AI.
"""

import argparse
import ast
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import xgboost as xgb


# ---------------------------------------------------------------------------
# Vector parsing / feature construction
# ---------------------------------------------------------------------------

def parse_one_hot(matrix_str: str) -> List[int]:
    """Parse a one-hot matrix string like '[[0,0,0],[0,0,1],...]' into a flat list.

    Expects a 6x3 one-hot matrix (JSON-formatted list of lists).
    Flattens to an 18-element list.
    """
    parsed = ast.literal_eval(matrix_str)
    # If nested (6x3 matrix format), flatten it
    if isinstance(parsed[0], list):
        return [val for row in parsed for val in row]
    return list(parsed)


def flatten_one_hot(row: pd.Series) -> np.ndarray:
    """Flatten start_one_hot + goal_one_hot from a problems_one_hot row into one feature vector.

    Each one-hot column is a 6x3 matrix, flattened to 18 elements.
    Returns a numpy array of length 36 (18 start + 18 goal features).
    """
    start = parse_one_hot(row["start_one_hot"])
    goal = parse_one_hot(row["goal_one_hot"])
    return np.array(start + goal, dtype=float)


def _build_id_lookup(
    problems_one_hot: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Build a lookup dict mapping normalized TOL-ID -> flattened feature vector.

    Normalizes IDs the same way as normalize_id() so that 'TOL-ID-1', 'TOL-ID-01',
    '01', and '1' all map to the same key '01'.
    """
    lookup: Dict[str, np.ndarray] = {}
    for _, row in problems_one_hot.iterrows():
        nid = normalize_id(row["ID"])
        lookup[nid] = flatten_one_hot(row)
    return lookup


def build_features(
    problems_one_hot: pd.DataFrame,
    problem_ids: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Build a feature matrix from the problems_one_hot for a subset of problem IDs.

    Args:
        problems_one_hot: full DataFrame with columns ID, start_one_hot, goal_one_hot.
        problem_ids: list of TOL-ID strings (e.g. ['01', '02', ...]).

    Returns:
        (X, valid_ids) where X is shape (n_valid, 36) and valid_ids are the
        normalized IDs that were actually found.  Returns empty arrays if
        no problems match.
    """
    lookup = _build_id_lookup(problems_one_hot)
    norm_ids = [normalize_id(pid) for pid in problem_ids]
    rows = []
    valid = []
    skipped = []
    for nid in norm_ids:
        if nid not in lookup:
            skipped.append(nid)
            continue
        rows.append(lookup[nid])
        valid.append(nid)
    if skipped:
        print(f"  WARNING: {len(skipped)} problem ID(s) not found in problems_one_hot: {skipped}")
    return np.array(rows) if rows else np.array([]).reshape(0, 36), valid


# ---------------------------------------------------------------------------
# Pairwise feature construction for binary classification
# ---------------------------------------------------------------------------

def build_pair_features(
    X_all: np.ndarray,
    id_to_idx: Dict[str, int],
    id1: str,
    id2: str,
) -> Optional[np.ndarray]:
    """Build feature vector for a pair of problems.

    Concatenates the feature vectors of problem 1 and problem 2.
    Shape: (72,) = 36 features for problem 1 + 36 features for problem 2.

    Args:
        X_all: full feature matrix of all problems.
        id_to_idx: mapping from normalized ID to row index in X_all.
        id1: normalized ID of problem 1.
        id2: normalized ID of problem 2.

    Returns:
        Flattened pair feature vector of shape (72,), or None if either ID
        is not found.
    """
    if id1 not in id_to_idx or id2 not in id_to_idx:
        return None
    return np.concatenate([X_all[id_to_idx[id1]], X_all[id_to_idx[id2]]])


def generate_training_pairs(
    train_ids: List[str],
    human_ranking: Dict[str, float],
    X_all: np.ndarray,
    id_to_idx: Dict[str, int],
    n_pairs: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training pairs with binary labels from human rankings.

    Randomly samples pairs of problems from the training set. For each pair
    (p1, p2), the label is 1 if human_score(p1) > human_score(p2), else 0.

    Args:
        train_ids: list of normalized training problem IDs.
        human_ranking: dict mapping normalized TOL-ID to human difficulty score.
        X_all: full feature matrix (all problems, not just training).
        id_to_idx: mapping from normalized ID to row index in X_all.
        n_pairs: number of training pairs to generate.
        rng: numpy random generator for reproducibility.

    Returns:
        (X_pairs, y_pairs) where X_pairs is shape (n_pairs, 72) and
        y_pairs is shape (n_pairs,) with binary labels (0 or 1).
    """
    # Filter to IDs that have human scores
    scored_ids = [pid for pid in train_ids if pid in human_ranking]
    if len(scored_ids) < 2:
        raise ValueError(
            f"Need at least 2 scored problems for pairwise training, "
            f"got {len(scored_ids)}"
        )

    X_pairs: List[np.ndarray] = []
    y_pairs: List[int] = []

    for _ in range(n_pairs):
        # Sample two distinct problem indices
        idx1, idx2 = rng.choice(len(scored_ids), size=2, replace=False)
        p1_id = scored_ids[idx1]
        p2_id = scored_ids[idx2]

        # Build pair features
        pair_feat = build_pair_features(X_all, id_to_idx, p1_id, p2_id)
        if pair_feat is None:
            continue

        # Label: 1 if problem 1 is harder (higher score) than problem 2
        score1 = human_ranking[p1_id]
        score2 = human_ranking[p2_id]
        label = 1 if score1 > score2 else 0

        X_pairs.append(pair_feat)
        y_pairs.append(label)

    if len(X_pairs) == 0:
        raise ValueError("No valid training pairs could be generated.")

    return np.array(X_pairs), np.array(y_pairs, dtype=int)


def predict_pairwise_ranking(
    model: xgb.XGBClassifier,
    test_ids: List[str],
    X_all: np.ndarray,
    id_to_idx: Dict[str, int],
) -> np.ndarray:
    """Derive a difficulty score for each test problem via pairwise comparisons.

    For each test problem, compute the fraction of wins against all other
    test problems using the trained binary classifier. This fraction serves
    as the predicted difficulty score.

    Args:
        model: trained AdaBoostClassifier.
        test_ids: list of normalized test problem IDs.
        X_all: full feature matrix (all problems).
        id_to_idx: mapping from normalized ID to row index in X_all.

    Returns:
        Array of predicted difficulty scores, one per test_ids entry.
    """
    n = len(test_ids)
    wins = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair_feat = build_pair_features(X_all, id_to_idx, test_ids[i], test_ids[j])
            if pair_feat is None:
                continue
            # Predict: 1 means test_ids[i] is harder than test_ids[j]
            pred = model.predict(pair_feat.reshape(1, -1))[0]
            if pred == 1:
                wins[i] += 1

    # Normalize by number of comparisons (n-1)
    if n > 1:
        scores = wins / (n - 1)
    else:
        scores = wins
    return scores


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def normalize_id(tol_id: str) -> str:
    """Normalize a TOL-ID to a zero-padded numeric string (e.g. '01', '13').

    Handles formats like 'TOL-ID-1', 'TOL-ID-01', '01', '1'.
    """
    if tol_id.startswith("TOL-ID-"):
        tol_id = tol_id.replace("TOL-ID-", "")
    return f"{int(tol_id):02d}"


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
) -> Dict[str, Dict[str, float]]:
    """Load all ranking CSVs from a directory.

    Args:
        rankings_dir: path to folder containing ranking CSV files.

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
        rankings[key] = ranking

    return rankings


def load_splits(splits_path: str) -> pd.DataFrame:
    """Load and parse the splits CSV file.

    Args:
        splits_path: path to the splits CSV file.

    Returns:
        DataFrame with 'train_problems' and 'validation_problems' columns
        containing lists of problem ID strings.
    """
    splits = pd.read_csv(splits_path)
    splits["train_problems"] = splits["train_problems"].str.split()
    splits["validation_problems"] = splits["validation_problems"].str.split()
    return splits


def load_problems_one_hot(problems_one_hot_path: str) -> pd.DataFrame:
    """Load the problem one-hot matrices CSV file.

    Args:
        problems_one_hot_path: path to the problems_one_hot.csv file.

    Returns:
        DataFrame with columns: 'ID', 'start_one_hot', 'goal_one_hot'.
    """
    return pd.read_csv(problems_one_hot_path)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class TrainingData:
    """Holds all data needed for training."""
    splits: pd.DataFrame
    human_rankings: Dict[str, Dict[str, float]]
    problems_one_hot: pd.DataFrame


def load_all_data(
    splits_path: str,
    human_rankings_dir: str,
    problems_one_hot_path: str,
) -> TrainingData:
    """Load all data sources and return a TrainingData object.

    Args:
        splits_path: path to the splits CSV file.
        human_rankings_dir: path to folder with human ranking CSVs.
        problems_one_hot_path: path to the problems_one_hot.csv file.
    """
    splits = load_splits(splits_path)
    human_rankings = load_rankings_from_dir(human_rankings_dir)
    problems_one_hot = load_problems_one_hot(problems_one_hot_path)

    return TrainingData(
        splits=splits,
        human_rankings=human_rankings,
        problems_one_hot=problems_one_hot,
    )


# ---------------------------------------------------------------------------
# Training data construction helpers
# ---------------------------------------------------------------------------

def extract_target_values(
    human_ranking: Dict[str, float],
    valid_ids: List[str],
) -> Optional[np.ndarray]:
    """Extract human ranking values aligned with valid problem IDs.

    Args:
        human_ranking: dict mapping normalized TOL-ID to ranking value.
        valid_ids: list of normalized problem IDs (e.g. ['01', '05']).

    Returns:
        numpy array of values aligned with valid_ids, or None if any ID is
        missing from the human ranking.
    """
    values = []
    for pid in valid_ids:
        val = human_ranking.get(pid)
        if val is None:
            val = human_ranking.get(f"TOL-ID-{pid}")
        if val is None:
            return None
        values.append(val)
    return np.array(values)


def build_full_feature_matrix(
    problems_one_hot: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build a feature matrix for ALL problems and an ID-to-index lookup.

    Args:
        problems_one_hot: full problems_one_hot DataFrame.

    Returns:
        (X_all, id_to_idx) where X_all is shape (n_problems, 36) and
        id_to_idx maps normalized ID -> row index.
    """
    lookup = _build_id_lookup(problems_one_hot)
    ids = sorted(lookup.keys())
    rows = [lookup[nid] for nid in ids]
    X_all = np.array(rows)
    id_to_idx = {nid: idx for idx, nid in enumerate(ids)}
    return X_all, id_to_idx


# ---------------------------------------------------------------------------
# Monte Carlo Kendall tau evaluation  (from linear_combo/testing.py)
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
        ordered_df: DataFrame with columns TOL-ID, metric_col, human_rank.
        metric_col: name of the metric column to resolve ties on.
        permutation_runs: number of Monte Carlo tie-breaking runs.
        seed: random seed.

    Returns:
        (mean_tau, std_tau)
    """
    if ordered_df[metric_col].nunique(dropna=False) <= 1:
        print(
            f"  WARNING: metric column '{metric_col}' has only "
            f"{ordered_df[metric_col].nunique(dropna=False)} unique value(s). "
            f"Kendall tau is undefined; returning (0.0, 0.0)."
        )
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    tau_values: list[float] = []

    print(f"  metric unique values: {ordered_df[metric_col].nunique()}, range: [{ordered_df[metric_col].min()}, {ordered_df[metric_col].max()}]")
    print(f"  human_rank unique values: {ordered_df['human_rank'].nunique()}, range: [{ordered_df['human_rank'].min()}, {ordered_df['human_rank'].max()}]")
    print(f"  human_rank has NaN: {ordered_df['human_rank'].isna().any()}")
    print(f"  df shape: {ordered_df.shape}")

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
            f"  WARNING: all {permutation_runs} tau values were NaN. "
            f"Returning (0.0, 0.0)."
        )
        return 0.0, 0.0
    if len(tau_values) < permutation_runs:
        print(
            f"  WARNING: {permutation_runs - len(tau_values)}/{permutation_runs} "
            f"tau values were NaN and were dropped."
        )

    return float(np.mean(tau_values)), float(np.std(tau_values))


# ---------------------------------------------------------------------------
# Evaluation helper  (mirrors linear_combo/testing.py output)
# ---------------------------------------------------------------------------

def evaluate_and_save(
    output_dir: str,
    split_idx: int,
    human_name: str,
    test_ids: List[str],
    predictions: np.ndarray,
    y_test: np.ndarray,
    permutation_runs: int,
    seed: int,
) -> Dict:
    """Evaluate predictions vs human ranking and save results.

    Writes {human_name}_tau_results.csv into {output_dir}/split_{idx:03d}/
    in the same format as linear_combo/testing.py.

    Args:
        output_dir: root output directory.
        split_idx: zero-based split index.
        human_name: name key for this human ranking.
        test_ids: normalized problem IDs for the validation set.
        predictions: model predictions (pairwise-derived scores) aligned with test_ids.
        y_test: ground-truth human values aligned with test_ids.
        permutation_runs: number of Monte Carlo tie-breaking runs.
        seed: random seed.

    Returns:
        Dict with keys: human_file, mean_tau, std_tau, output_file.
    """
    split_label = f"split_{split_idx:03d}"
    split_output_dir = os.path.join(output_dir, split_label)
    os.makedirs(split_output_dir, exist_ok=True)

    df = pd.DataFrame({
        "TOL-ID": test_ids,
        "metric": predictions,
        "human_rank": y_test,
    })

    mean_tau, std_tau = compute_monte_carlo_tau(
        df,
        metric_col="metric",
        permutation_runs=permutation_runs,
        seed=seed,
    )

    # Save per-human tau results CSV (same schema as testing.py)
    output_csv = os.path.join(
        split_output_dir, f"{human_name}_tau_results.csv"
    )
    pd.DataFrame({
        "metric": ["kendall_tau"],
        "mean": [mean_tau],
        "std": [std_tau],
    }).to_csv(output_csv, index=False)

    return {
        "human_file": f"{human_name}.csv",
        "mean_tau": mean_tau,
        "std_tau": std_tau,
        "output_file": f"{human_name}_tau_results.csv",
    }


# ---------------------------------------------------------------------------
# Main training + evaluation loop
# ---------------------------------------------------------------------------

def run(
    data: TrainingData,
    output_dir: str,
    permutation_runs: int = 1000,
    seed: int = 42,
    n_train_pairs: int = 5000,
) -> None:
    """Run the full binary XGBoost training and evaluation loop.

    For each split and each human ranking:
      1. Build full feature matrix for all problems.
      2. Generate pairwise training samples from the training set.
      3. Train a binary XGBClassifier (max_depth=4) on the pairs.
      4. Derive validation scores via pairwise comparisons.
      5. Evaluate via Kendall tau against human rankings.

    Output layout mirrors linear_combo/testing.py:
        {output_dir}/
            split_001/
                {human_ranking_1}_tau_results.csv
                summary.csv
            split_002/
                ...

    Args:
        data: TrainingData with splits, human rankings, problem vectors.
        output_dir: root output directory.
        permutation_runs: number of Monte Carlo tie-breaking runs.
        seed: random seed.
        n_train_pairs: number of random pairs to sample for training.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build full feature matrix once (all problems)
    X_all, id_to_idx = build_full_feature_matrix(data.problems_one_hot)
    print(f"Built full feature matrix: {X_all.shape}")

    for split_idx, split_row in data.splits.iterrows():
        split_label = f"split_{split_idx:03d}"
        print(f"\n--- {split_label} "
              f"({len(split_row['train_problems'])} train, "
              f"{len(split_row['validation_problems'])} val) ---")

        # Get valid training and test IDs
        _, train_ids = build_features(
            data.problems_one_hot, split_row["train_problems"]
        )
        _, test_ids = build_features(
            data.problems_one_hot, split_row["validation_problems"]
        )

        results: List[Dict] = []
        processed = 0

        for human_name, human_ranking in data.human_rankings.items():
            # --- extract human scores for test set --------------------------
            y_test = extract_target_values(human_ranking, test_ids)
            if y_test is None or len(test_ids) == 0:
                print(f"  {human_name:30s}  SKIPPED (missing test data)")
                continue

            # --- generate pairwise training data ----------------------------
            rng = np.random.default_rng(seed)
            try:
                X_pairs, y_pairs = generate_training_pairs(
                    train_ids=train_ids,
                    human_ranking=human_ranking,
                    X_all=X_all,
                    id_to_idx=id_to_idx,
                    n_pairs=n_train_pairs,
                    rng=rng,
                )
            except ValueError as e:
                print(f"  {human_name:30s}  SKIPPED ({e})")
                continue

            print(f"  {human_name}: X_pairs={X_pairs.shape}, "
                  f"class balance: {np.mean(y_pairs):.2f}")

            # --- train binary XGBoost classifier (max_depth=4) --------------
            model = xgb.XGBClassifier(
                max_depth=4,
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
            )
            model.fit(X_pairs, y_pairs)

            # --- derive validation scores via pairwise comparisons ----------
            predictions = predict_pairwise_ranking(
                model=model,
                test_ids=test_ids,
                X_all=X_all,
                id_to_idx=id_to_idx,
            )

            # --- evaluate predictions vs human ranking ----------------------
            result = evaluate_and_save(
                output_dir=output_dir,
                split_idx=split_idx,
                human_name=human_name,
                test_ids=test_ids,
                predictions=predictions,
                y_test=y_test,
                permutation_runs=permutation_runs,
                seed=seed,
            )

            results.append(result)
            processed += 1
            print(
                f"    mean_tau={result['mean_tau']:.4f}, "
                f"std_tau={result['std_tau']:.4f}"
            )

        if processed == 0:
            raise RuntimeError(
                f"No valid human rankings were processed for {split_label}."
            )

        # --- save summary CSV for this split ---------------------------------
        summary_df = pd.DataFrame(results)
        summary_csv = os.path.join(
            os.path.join(output_dir, split_label), "summary.csv"
        )
        summary_df.to_csv(summary_csv, index=False)
        print(f"  Saved summary: {split_label}/summary.csv")

    print(f"\nDone.  Results written to: {output_dir}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train binary XGBoost classifier for Tower of London "
                    "problem difficulty comparison."
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
        "--problems-one-hot",
        type=str,
        required=True,
        help="Path to the problems_one_hot.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory for output.  One sub-folder per split "
             "(split_001/, split_002/, …), matching testing.py layout.",
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
    parser.add_argument(
        "--n-train-pairs",
        type=int,
        default=5000,
        help="Number of random problem pairs to generate for training.",
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
        problems_one_hot_path=args.problems_one_hot,
    )

    print(f"Loaded {len(data.splits)} splits")
    print(f"Loaded {len(data.human_rankings)} human ranking(s): "
          f"{list(data.human_rankings.keys())}")
    print(f"Loaded {len(data.problems_one_hot)} problem one-hot matrices")
    print(f"Output directory: {args.output_dir}")
    print()

    run(
        data=data,
        output_dir=args.output_dir,
        permutation_runs=args.permutation_runs,
        seed=args.seed,
        n_train_pairs=args.n_train_pairs,
    )


if __name__ == "__main__":
    main()
