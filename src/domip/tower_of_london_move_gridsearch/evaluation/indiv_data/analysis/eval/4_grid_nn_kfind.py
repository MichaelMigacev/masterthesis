"""
Grid-based k-NN classification with automatic k selection via inner cross-validation.

For each class (subfolder), loads individual kendall_tau_mean grids. During testing,
each individual grid is classified by k-nearest neighbors (euclidean distance) among
the individual training grids, with majority voting.

The optimal k (1..30) is found automatically via an inner CV loop:
  - Data is split into 5 outer folds.
  - For each outer fold, the 4 training folds are used for an inner 4-fold CV
    to evaluate every candidate k in [1, 30].
  - The k with the highest mean inner-CV accuracy is selected.
  - Performance is reported on the held-out outer validation fold.

Usage:
    python 4_grid_nn_kfind.py --input <folder> --output <folder> [--folds 5] [--repeats 20] [--top_k K] [--group_clinical]

This file contains code written by AI.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, confusion_matrix


def load_grid(csv_path: str, top_k: int | None = None) -> pd.DataFrame:
    """Load a single grid CSV and return a pivot table indexed by (alpha, beta) with kendall_tau_mean values.

    If *top_k* is set, only the rows with the top_k largest kendall_tau_mean values are retained.
    """
    df = pd.read_csv(csv_path)
    grid = df.pivot_table(index=["alpha", "beta"], columns="source_file", values="kendall_tau_mean")
    # Flatten to single column of kendall_tau_mean values per (alpha, beta)
    # If there are multiple source_files, take the mean (shouldn't happen for individual grids)
    grid = df.pivot_table(index=["alpha", "beta"], values="kendall_tau_mean", aggfunc="mean")
    grid.columns = ["kendall_tau_mean"]

    if top_k is not None:
        # Keep only the rows with the top_k largest kendall_tau_mean values
        grid = grid.nlargest(top_k, "kendall_tau_mean")

    return grid


def compute_average_grid(grids: list[pd.DataFrame]) -> pd.DataFrame:
    """Compute the average kendall_tau_mean grid across a list of individual grids."""
    # Concatenate all grids and group by (alpha, beta) to compute mean
    combined = pd.concat(grids)
    avg_grid = combined.groupby(level=["alpha", "beta"]).mean()
    return avg_grid


def align_grids(grids: list[pd.DataFrame]) -> tuple[list[np.ndarray], pd.MultiIndex]:
    """Align multiple grids to a common (alpha, beta) index and return flattened arrays."""
    # Find the union of all indices
    all_indices = set()
    for g in grids:
        all_indices.update(g.index.tolist())
    common_index = pd.MultiIndex.from_tuples(sorted(all_indices), names=["alpha", "beta"])

    arrays = []
    for g in grids:
        aligned = g.reindex(index=common_index)["kendall_tau_mean"]
        # Replace NaN with 0 (grid points not present in some individuals)
        aligned = aligned.fillna(0.0)
        arrays.append(aligned.values)

    return arrays, common_index


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute euclidean distance between two vectors, ignoring positions where both are NaN."""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def classify_individual_knn(
    test_vec: np.ndarray,
    train_data: dict[str, list[tuple[str, np.ndarray]]],
    k: int = 1,
) -> str:
    """Classify a single test grid by k-nearest neighbors among individual training grids.

    Args:
        test_vec: Flattened test grid vector.
        train_data: Dict mapping class_name -> list of (filename, grid_vector) tuples.
        k: Number of nearest neighbors for majority voting.

    Returns:
        The class label predicted by majority vote among the k nearest neighbors.
        On ties, the class with the smallest average distance among its neighbors wins.
    """
    # Collect all training vectors with their class labels
    all_train: list[tuple[str, np.ndarray]] = []
    for class_name, individuals in train_data.items():
        for fname, vec in individuals:
            all_train.append((class_name, vec))

    # Compute distances to all training samples
    distances = [(euclidean_distance(test_vec, vec), cls) for cls, vec in all_train]
    distances.sort(key=lambda x: x[0])

    # Take k nearest neighbors and vote
    neighbors = distances[:k]
    votes = Counter(cls for _, cls in neighbors)
    max_count = votes.most_common(1)[0][1]

    # Get all classes tied for the most votes
    tied_classes = [cls for cls, count in votes.items() if count == max_count]

    if len(tied_classes) == 1:
        return tied_classes[0]

    # Tie-break: choose the class with the smallest average distance among its neighbors
    best_class = None
    best_avg_dist = np.inf
    for cls in tied_classes:
        cls_dists = [dist for dist, ncls in neighbors if ncls == cls]
        avg_dist = np.mean(cls_dists)
        if avg_dist < best_avg_dist:
            best_avg_dist = avg_dist
            best_class = cls
    return best_class


def load_all_data(input_folder: str, top_k: int | None = None) -> dict[str, list[tuple[str, pd.DataFrame]]]:
    """
    Load all CSV files from subfolders.
    Returns: {class_name: [(filename, grid_df), ...]}
    """
    data = {}
    input_path = Path(input_folder)

    for class_dir in sorted(input_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        grids = []
        for csv_file in sorted(class_dir.glob("*.csv")):
            grid = load_grid(str(csv_file), top_k=top_k)
            grids.append((csv_file.name, grid))
        if grids:
            data[class_name] = grids
    return data


def merge_clinical_groups(
    data: dict[str, list[tuple[str, pd.DataFrame]]],
    healthy_label: str = "Healthy",
) -> dict[str, list[tuple[str, pd.DataFrame]]]:
    """
    Merge all non-Healthy classes into a single 'Clinical' group.

    Detects the healthy group by case-insensitive match against *healthy_label*.
    All other groups are combined under the key 'Clinical'.
    """
    merged: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    healthy_key = None

    for class_name in data:
        if class_name.lower() == healthy_label.lower():
            healthy_key = class_name
            merged[class_name] = data[class_name]
        else:
            merged.setdefault("Clinical", []).extend(data[class_name])

    if healthy_key is None:
        raise ValueError(
            f"No class matching '{healthy_label}' (case-insensitive) found. "
            f"Classes: {sorted(data.keys())}"
        )
    if "Clinical" not in merged:
        raise ValueError(
            f"Only the healthy class '{healthy_key}' was found; need at least one "
            "clinical group to merge."
        )
    return merged


def _build_fold_assignments(data: dict, n_folds: int, rng: np.random.RandomState) -> dict[str, dict[int, list[int]]]:
    """Shuffle and assign each individual to a fold within each class."""
    fold_assignments: dict[str, dict[int, list[int]]] = {}
    for class_name, individuals in data.items():
        n_indiv = len(individuals)
        indices = list(range(n_indiv))
        rng.shuffle(indices)
        folds: dict[int, list[int]] = {f: [] for f in range(n_folds)}
        for i, idx in enumerate(indices):
            folds[i % n_folds].append(idx)
        fold_assignments[class_name] = folds
    return fold_assignments


def _classify_fold(
    test_indices_map: dict[str, list[int]],
    train_indices_map: dict[str, list[int]],
    data: dict[str, list[tuple[str, pd.DataFrame]]],
    k: int,
) -> tuple[list[str], list[str], list[str]]:
    """Classify all test individuals using k-NN and return (true_labels, pred_labels, filenames)."""
    test_true_labels = []
    test_pred_labels = []
    test_filenames = []

    # Collect all grids that need alignment
    all_grids_to_align: list[pd.DataFrame] = []
    test_grid_refs: list[tuple[str, str, int]] = []  # (class_name, filename, position in all_grids_to_align)

    # First add test grids
    for class_name, individuals in data.items():
        for idx in test_indices_map.get(class_name, []):
            fname, grid = individuals[idx]
            pos = len(all_grids_to_align)
            all_grids_to_align.append(grid)
            test_grid_refs.append((class_name, fname, pos))

    # Then add train grids
    train_grid_refs: list[tuple[str, str, int]] = []
    for class_name, individuals in data.items():
        for idx in train_indices_map.get(class_name, []):
            fname, grid = individuals[idx]
            pos = len(all_grids_to_align)
            all_grids_to_align.append(grid)
            train_grid_refs.append((class_name, fname, pos))

    if not all_grids_to_align:
        return test_true_labels, test_pred_labels, test_filenames

    aligned_arrays, _ = align_grids(all_grids_to_align)

    # Build aligned training data: class -> list of (filename, vector)
    train_data_aligned: dict[str, list[tuple[str, np.ndarray]]] = {}
    for cls_name, fname, pos in train_grid_refs:
        train_data_aligned.setdefault(cls_name, []).append((fname, aligned_arrays[pos]))

    # Classify each test individual
    for class_name, fname, pos in test_grid_refs:
        pred_label = classify_individual_knn(aligned_arrays[pos], train_data_aligned, k=k)
        test_true_labels.append(class_name)
        test_pred_labels.append(pred_label)
        test_filenames.append(fname)

    return test_true_labels, test_pred_labels, test_filenames


def _evaluate_predictions(
    true_labels: list[str],
    pred_labels: list[str],
    n_classes: int,
) -> tuple[dict, list[str]]:
    """Compute accuracy, macro F1/precision/recall, and per-class F1/precision/recall/accuracy."""
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    macro_precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    macro_recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)

    per_class_f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)
    per_class_precision = precision_score(true_labels, pred_labels, average=None, zero_division=0)
    per_class_recall = recall_score(true_labels, pred_labels, average=None, zero_division=0)
    classes_sorted = sorted(set(true_labels))
    per_class_f1_dict = {cls: float(per_class_f1[i]) for i, cls in enumerate(classes_sorted)}
    per_class_precision_dict = {cls: float(per_class_precision[i]) for i, cls in enumerate(classes_sorted)}
    per_class_recall_dict = {cls: float(per_class_recall[i]) for i, cls in enumerate(classes_sorted)}

    # Per-class accuracy: how often is class i predicted correctly when it is the true class?
    cm = confusion_matrix(true_labels, pred_labels, labels=classes_sorted)
    per_class_acc_dict: dict[str, float] = {}
    for i, cls in enumerate(classes_sorted):
        per_class_acc_dict[cls] = float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0.0

    result: dict = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "n_test_samples": len(true_labels),
        "n_classes": n_classes,
        "confusion_matrix": cm,  # will be stripped before DataFrame conversion
    }
    for cls in sorted(classes_sorted):
        result[f"f1_{cls}"] = per_class_f1_dict.get(cls, 0.0)
        result[f"precision_{cls}"] = per_class_precision_dict.get(cls, 0.0)
        result[f"recall_{cls}"] = per_class_recall_dict.get(cls, 0.0)
        result[f"accuracy_{cls}"] = per_class_acc_dict.get(cls, 0.0)
    return result, classes_sorted


def find_best_k(
    data: dict[str, list[tuple[str, pd.DataFrame]]],
    fold_assignments: dict[str, dict[int, list[int]]],
    inner_folds: list[int],
    k_range: range,
) -> int:
    """
    Inner cross-validation to find the best k in [1, 30].

    Uses only the folds in *inner_folds* (the 4 training folds, excluding the outer
    validation fold): for each inner fold, train on the other 3 and validate on
    the held-out inner fold. Average macro F1 across 4 inner folds per k value;
    return the k with highest mean macro F1.
    """
    inner_f1s: dict[int, list[float]] = {k: [] for k in k_range}

    # For each of the inner folds, use it as validation and the rest as training
    for inner_val_fold in inner_folds:
        inner_train_folds = [f for f in inner_folds if f != inner_val_fold]

        # Build index maps: which indices go to train vs val for this inner split
        val_indices: dict[str, list[int]] = {}
        train_indices: dict[str, list[int]] = {}
        for class_name, folds in fold_assignments.items():
            val_indices[class_name] = folds[inner_val_fold]
            train_indices[class_name] = [idx for f in inner_train_folds for idx in folds[f]]

        # Classify with each k
        for k in k_range:
            true_labels, pred_labels, _ = _classify_fold(
                val_indices, train_indices, data, k
            )
            if true_labels:
                macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
                inner_f1s[k].append(macro_f1)

    # Pick k with highest mean inner-CV macro F1
    best_k = max(k_range, key=lambda k: np.mean(inner_f1s[k]) if inner_f1s[k] else -1)
    return best_k


def run_cross_validation(
    data: dict[str, list[tuple[str, pd.DataFrame]]],
    n_folds: int,
    n_repeats: int,
    random_seed: int = 42,
) -> tuple[list[tuple[dict, np.ndarray, list[str]]], list[tuple[dict, np.ndarray, list[str]]]]:
    """
    Run repeated k-fold cross-validation with automatic k selection and a random baseline.

    For each repeat:
      - Randomly split individuals within each class into n_folds folds
      - For each outer fold:
          * Use the 4 training folds for inner CV to find best k in [1, 20]
          * Classify the held-out fold using the best k
          * Also classify the held-out fold by random sampling from the training
            set class distribution (using all 4 inner CV folds as the distribution)
          * Report performance on the held-out validation fold

    Returns: (knn_results, random_results) each a list of (result_dict, cm, classes) tuples.
    """
    n_classes = len(data)
    knn_results: list[tuple[dict, np.ndarray, list[str]]] = []
    random_results: list[tuple[dict, np.ndarray, list[str]]] = []
    rng = np.random.RandomState(random_seed)
    k_range = range(1, 21)  # k in [1, 20]

    for repeat in range(n_repeats):
        fold_assignments = _build_fold_assignments(data, n_folds, rng)

        for outer_fold_idx in range(n_folds):
            # --- Inner CV: find best k using the 4 training folds (excluding outer fold) ---
            inner_folds = [f for f in range(n_folds) if f != outer_fold_idx]
            best_k = find_best_k(data, fold_assignments, inner_folds, k_range)

            # --- Outer validation: classify held-out fold with best k ---
            test_indices: dict[str, list[int]] = {}
            train_indices: dict[str, list[int]] = {}
            for class_name, folds in fold_assignments.items():
                test_indices[class_name] = folds[outer_fold_idx]
                train_indices[class_name] = [
                    idx for f in range(n_folds) if f != outer_fold_idx for idx in folds[f]
                ]

            true_labels, pred_labels, filenames = _classify_fold(
                test_indices, train_indices, data, best_k
            )

            if not true_labels:
                continue

            metrics, classes_in_fold = _evaluate_predictions(true_labels, pred_labels, n_classes)

            result_entry = {
                "repeat": repeat + 1,
                "fold": outer_fold_idx + 1,
                "best_k": best_k,
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            }
            knn_results.append((result_entry, metrics["confusion_matrix"], classes_in_fold))

            # --- Random baseline classification ---
            true_labels_r, pred_labels_r, filenames_r = _classify_fold_random(
                test_indices, train_indices, data, rng
            )

            metrics_r, classes_in_fold_r = _evaluate_predictions(true_labels_r, pred_labels_r, n_classes)

            result_entry_r = {
                "repeat": repeat + 1,
                "fold": outer_fold_idx + 1,
                **{k: v for k, v in metrics_r.items() if k != "confusion_matrix"},
            }
            random_results.append((result_entry_r, metrics_r["confusion_matrix"], classes_in_fold_r))

    return knn_results, random_results


def _classify_fold_random(
    test_indices_map: dict[str, list[int]],
    train_indices_map: dict[str, list[int]],
    data: dict[str, list[tuple[str, pd.DataFrame]]],
    rng: np.random.RandomState,
) -> tuple[list[str], list[str], list[str]]:
    """Classify all test individuals by random sampling from the training set class distribution.

    For each test sample, pick a class randomly with probability proportional to the
    fraction of that class in the training set.

    Returns (true_labels, pred_labels, filenames).
    """
    # Compute class probabilities from training set distribution
    train_counts: dict[str, int] = {}
    for class_name, indices in train_indices_map.items():
        train_counts[class_name] = len(indices)
    total_train = sum(train_counts.values())

    classes = sorted(train_counts.keys())
    probs = np.array([train_counts[c] / total_train for c in classes])

    test_true_labels = []
    test_pred_labels = []
    test_filenames = []

    for class_name, individuals in data.items():
        for idx in test_indices_map.get(class_name, []):
            fname, _ = individuals[idx]
            pred_label = rng.choice(classes, p=probs)
            test_true_labels.append(class_name)
            test_pred_labels.append(pred_label)
            test_filenames.append(fname)

    return test_true_labels, test_pred_labels, test_filenames


def aggregate_confusion_matrices(
    results: list[tuple[dict, np.ndarray, list[str]]],
) -> tuple[np.ndarray, list[str]]:
    """Sum per-fold confusion matrices into a single aggregated matrix.

    Returns (aggregated_matrix, canonical_class_order).
    """
    # Use the class order from the first fold as canonical
    canonical_classes = results[0][2]
    canonical = np.zeros((len(canonical_classes), len(canonical_classes)), dtype=int)

    class_to_idx = {cls: i for i, cls in enumerate(canonical_classes)}
    for _, cm, fold_classes in results:
        for fi, fc in enumerate(fold_classes):
            for fj, pc in enumerate(fold_classes):
                ci = class_to_idx.get(fc)
                cj = class_to_idx.get(pc)
                if ci is not None and cj is not None:
                    canonical[ci, cj] += cm[fi, fj]
    return canonical, canonical_classes


def main():
    parser = argparse.ArgumentParser(
        description="Grid-based k-NN classification with automatic k selection via inner cross-validation."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input folder containing class subfolders, each with .csv grid files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output folder for results.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Number of times to repeat the cross-validation (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Within each grid CSV, keep only the top k largest kendall_tau_mean values (default: None, keep all).",
    )
    parser.add_argument(
        "--group_clinical",
        action="store_true",
        help="Merge all non-Healthy classes into a single 'Clinical' group for binary classification (Clinical vs Healthy).",
    )
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output
    n_folds = args.folds
    n_repeats = args.repeats
    seed = args.seed
    top_k = args.top_k
    group_clinical = args.group_clinical

    # Validate input
    input_path = Path(input_folder)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input folder '{input_folder}' does not exist or is not a directory.")
        sys.exit(1)

    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    if len(class_dirs) < 2:
        print(f"Error: Need at least 2 class subfolders, found {len(class_dirs)}.")
        sys.exit(1)

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Classes found: {[d.name for d in class_dirs]}")
    print(f"Folds: {n_folds}, Repeats: {n_repeats}, Seed: {seed}")
    print(f"k search range: [1, 20] (selected via inner cross-validation)")
    if top_k is not None:
        print(f"Top k: {top_k} (keeping only top {top_k} kendall_tau_mean values per grid)")

    # Load data
    print("\nLoading data...")
    data = load_all_data(input_folder, top_k=top_k)
    for class_name, individuals in sorted(data.items()):
        print(f"  {class_name}: {len(individuals)} individuals")

    # Optionally merge clinical groups into a single group
    if group_clinical:
        print("\nMerging all non-Healthy classes into 'Clinical' group...")
        original_classes = sorted(data.keys())
        data = merge_clinical_groups(data)
        for class_name, individuals in sorted(data.items()):
            print(f"  {class_name}: {len(individuals)} individuals")
        print(f"  (merged {len(original_classes) - 1} clinical groups + 1 healthy group -> 2 groups)")

    # Run cross-validation with automatic k selection
    print(f"\nRunning {n_repeats}x{n_folds}-fold cross-validation with inner k selection (k in [1,20])...")
    knn_results, random_results = run_cross_validation(data, n_folds, n_repeats, random_seed=seed)

    if not knn_results:
        print("Error: No results produced. Check data and parameters.")
        sys.exit(1)

    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── k-NN results (best k) ─────────────────────────────────────────────────
    result_dicts = [r[0] for r in knn_results]
    results_df = pd.DataFrame(result_dicts)

    # Aggregate and save confusion matrix
    aggregated_cm, cm_classes = aggregate_confusion_matrices(knn_results)
    # Normalize row-wise so each row sums to 1.0
    row_sums = aggregated_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    normalized_cm = aggregated_cm.astype(float) / row_sums
    cm_df = pd.DataFrame(normalized_cm, index=cm_classes, columns=cm_classes)
    cm_df.index.name = "true_label"
    cm_df.columns.name = "predicted_label"
    cm_path = output_path / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"Aggregated normalized confusion matrix saved to: {cm_path}")

    # Compute summary statistics
    summary = {
        "accuracy_mean": results_df["accuracy"].mean(),
        "accuracy_std": results_df["accuracy"].std(),
        "accuracy_min": results_df["accuracy"].min(),
        "accuracy_max": results_df["accuracy"].max(),
        "accuracy_median": results_df["accuracy"].median(),
        "macro_f1_mean": results_df["macro_f1"].mean(),
        "macro_f1_std": results_df["macro_f1"].std(),
        "macro_f1_min": results_df["macro_f1"].min(),
        "macro_f1_max": results_df["macro_f1"].max(),
        "macro_f1_median": results_df["macro_f1"].median(),
        "macro_precision_mean": results_df["macro_precision"].mean(),
        "macro_precision_std": results_df["macro_precision"].std(),
        "macro_precision_min": results_df["macro_precision"].min(),
        "macro_precision_max": results_df["macro_precision"].max(),
        "macro_precision_median": results_df["macro_precision"].median(),
        "macro_recall_mean": results_df["macro_recall"].mean(),
        "macro_recall_std": results_df["macro_recall"].std(),
        "macro_recall_min": results_df["macro_recall"].min(),
        "macro_recall_max": results_df["macro_recall"].max(),
        "macro_recall_median": results_df["macro_recall"].median(),
        "best_k_mean": results_df["best_k"].mean(),
        "best_k_std": results_df["best_k"].std(),
        "best_k_median": results_df["best_k"].median(),
        "best_k_min": results_df["best_k"].min(),
        "best_k_max": results_df["best_k"].max(),
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "n_total_test_samples": results_df["n_test_samples"].sum(),
        "n_classes": results_df["n_classes"].iloc[0],
    }

    # Per-class F1, precision, recall, accuracy summary statistics
    all_classes = sorted(data.keys())
    for cls in all_classes:
        for metric in ("f1", "precision", "recall", "accuracy"):
            col = f"{metric}_{cls}"
            if col in results_df.columns:
                summary[f"{metric}_{cls}_mean"] = results_df[col].mean()
                summary[f"{metric}_{cls}_std"] = results_df[col].std()
                summary[f"{metric}_{cls}_min"] = results_df[col].min()
                summary[f"{metric}_{cls}_max"] = results_df[col].max()
                summary[f"{metric}_{cls}_median"] = results_df[col].median()

    # Save per-fold results
    per_fold_path = output_path / "per_fold_results.csv"
    results_df.to_csv(per_fold_path, index=False)
    print(f"\nPer-fold results saved to: {per_fold_path}")

    # Save per-class test counts for transparency
    class_counts = {cls: len(indivs) for cls, indivs in data.items()}
    class_counts_df = pd.DataFrame([
        {"class": cls, "n_individuals": count}
        for cls, count in sorted(class_counts.items())
    ])
    class_counts_path = output_path / "class_counts.csv"
    class_counts_df.to_csv(class_counts_path, index=False)

    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_path = output_path / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (best k via inner CV)")
    print("=" * 60)
    print(f"Best k:    {summary['best_k_mean']:.1f} ± {summary['best_k_std']:.1f}")
    print(f"  min={summary['best_k_min']}, max={summary['best_k_max']}, median={summary['best_k_median']}")
    print(f"Accuracy:  {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"  min={summary['accuracy_min']:.4f}, max={summary['accuracy_max']:.4f}, median={summary['accuracy_median']:.4f}")
    print(f"Macro F1:      {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    print(f"  min={summary['macro_f1_min']:.4f}, max={summary['macro_f1_max']:.4f}, median={summary['macro_f1_median']:.4f}")
    print(f"Macro Precision: {summary['macro_precision_mean']:.4f} ± {summary['macro_precision_std']:.4f}")
    print(f"  min={summary['macro_precision_min']:.4f}, max={summary['macro_precision_max']:.4f}, median={summary['macro_precision_median']:.4f}")
    print(f"Macro Recall:    {summary['macro_recall_mean']:.4f} ± {summary['macro_recall_std']:.4f}")
    print(f"  min={summary['macro_recall_min']:.4f}, max={summary['macro_recall_max']:.4f}, median={summary['macro_recall_median']:.4f}")
    print(f"\nPer-class metrics:")
    for cls in all_classes:
        print(f"  {cls}:")
        for metric in ("f1", "precision", "recall", "accuracy"):
            mean_key = f"{metric}_{cls}_mean"
            if mean_key in summary:
                print(f"    {metric:>10s}: {summary[mean_key]:.4f} ± {summary[f'{metric}_{cls}_std']:.4f}")
                print(f"      min={summary[f'{metric}_{cls}_min']:.4f}, max={summary[f'{metric}_{cls}_max']:.4f}, median={summary[f'{metric}_{cls}_median']:.4f}")
    print(f"Total test samples: {summary['n_total_test_samples']}")
    print(f"Classes: {summary['n_classes']}")
    print("=" * 60)

    # ── Random baseline results ───────────────────────────────────────────────
    print(f"\nRandom baseline (distribution-aware, train-set prior)...")
    random_dicts = [r[0] for r in random_results]
    random_df = pd.DataFrame(random_dicts)

    # Save random per-fold results
    random_per_fold_path = output_path / "random_per_fold_results.csv"
    random_df.to_csv(random_per_fold_path, index=False)
    print(f"Random per-fold results saved to: {random_per_fold_path}")

    # Aggregate and save random confusion matrix
    random_agg_cm, random_cm_classes = aggregate_confusion_matrices(random_results)
    random_row_sums = random_agg_cm.sum(axis=1, keepdims=True)
    random_row_sums[random_row_sums == 0] = 1
    random_norm_cm = random_agg_cm.astype(float) / random_row_sums
    random_cm_df = pd.DataFrame(random_norm_cm, index=random_cm_classes, columns=random_cm_classes)
    random_cm_df.index.name = "true_label"
    random_cm_df.columns.name = "predicted_label"
    random_cm_path = output_path / "random_confusion_matrix.csv"
    random_cm_df.to_csv(random_cm_path)
    print(f"Random aggregated normalized confusion matrix saved to: {random_cm_path}")

    # Compute random summary statistics
    random_summary = {
        "accuracy_mean": random_df["accuracy"].mean(),
        "accuracy_std": random_df["accuracy"].std(),
        "accuracy_min": random_df["accuracy"].min(),
        "accuracy_max": random_df["accuracy"].max(),
        "accuracy_median": random_df["accuracy"].median(),
        "macro_f1_mean": random_df["macro_f1"].mean(),
        "macro_f1_std": random_df["macro_f1"].std(),
        "macro_f1_min": random_df["macro_f1"].min(),
        "macro_f1_max": random_df["macro_f1"].max(),
        "macro_f1_median": random_df["macro_f1"].median(),
        "macro_precision_mean": random_df["macro_precision"].mean(),
        "macro_precision_std": random_df["macro_precision"].std(),
        "macro_precision_min": random_df["macro_precision"].min(),
        "macro_precision_max": random_df["macro_precision"].max(),
        "macro_precision_median": random_df["macro_precision"].median(),
        "macro_recall_mean": random_df["macro_recall"].mean(),
        "macro_recall_std": random_df["macro_recall"].std(),
        "macro_recall_min": random_df["macro_recall"].min(),
        "macro_recall_max": random_df["macro_recall"].max(),
        "macro_recall_median": random_df["macro_recall"].median(),
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "n_total_test_samples": random_df["n_test_samples"].sum(),
        "n_classes": random_df["n_classes"].iloc[0],
    }

    for cls in all_classes:
        for metric in ("f1", "precision", "recall", "accuracy"):
            col = f"{metric}_{cls}"
            if col in random_df.columns:
                random_summary[f"{metric}_{cls}_mean"] = random_df[col].mean()
                random_summary[f"{metric}_{cls}_std"] = random_df[col].std()
                random_summary[f"{metric}_{cls}_min"] = random_df[col].min()
                random_summary[f"{metric}_{cls}_max"] = random_df[col].max()
                random_summary[f"{metric}_{cls}_median"] = random_df[col].median()

    # Save random summary
    random_summary_df = pd.DataFrame([random_summary])
    random_summary_path = output_path / "random_summary.csv"
    random_summary_df.to_csv(random_summary_path, index=False)
    print(f"Random summary saved to: {random_summary_path}")

    # Print random summary
    print("\n" + "=" * 60)
    print("RANDOM BASELINE SUMMARY")
    print("=" * 60)
    print(f"Accuracy:  {random_summary['accuracy_mean']:.4f} ± {random_summary['accuracy_std']:.4f}")
    print(f"  min={random_summary['accuracy_min']:.4f}, max={random_summary['accuracy_max']:.4f}, median={random_summary['accuracy_median']:.4f}")
    print(f"Macro F1:      {random_summary['macro_f1_mean']:.4f} ± {random_summary['macro_f1_std']:.4f}")
    print(f"  min={random_summary['macro_f1_min']:.4f}, max={random_summary['macro_f1_max']:.4f}, median={random_summary['macro_f1_median']:.4f}")
    print(f"Macro Precision: {random_summary['macro_precision_mean']:.4f} ± {random_summary['macro_precision_std']:.4f}")
    print(f"  min={random_summary['macro_precision_min']:.4f}, max={random_summary['macro_precision_max']:.4f}, median={random_summary['macro_precision_median']:.4f}")
    print(f"Macro Recall:    {random_summary['macro_recall_mean']:.4f} ± {random_summary['macro_recall_std']:.4f}")
    print(f"  min={random_summary['macro_recall_min']:.4f}, max={random_summary['macro_recall_max']:.4f}, median={random_summary['macro_recall_median']:.4f}")
    print(f"\nPer-class metrics:")
    for cls in all_classes:
        print(f"  {cls}:")
        for metric in ("f1", "precision", "recall", "accuracy"):
            mean_key = f"{metric}_{cls}_mean"
            if mean_key in random_summary:
                print(f"    {metric:>10s}: {random_summary[mean_key]:.4f} ± {random_summary[f'{metric}_{cls}_std']:.4f}")
                print(f"      min={random_summary[f'{metric}_{cls}_min']:.4f}, max={random_summary[f'{metric}_{cls}_max']:.4f}, median={random_summary[f'{metric}_{cls}_median']:.4f}")
    print(f"Total test samples: {random_summary['n_total_test_samples']}")
    print(f"Classes: {random_summary['n_classes']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
