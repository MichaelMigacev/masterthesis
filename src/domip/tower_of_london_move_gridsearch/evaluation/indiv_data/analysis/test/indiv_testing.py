"""


This file contains code written by AI.
"""


from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tower_of_london_plots" / "indiv_plots" / "PD" / "test" / "Controls"


def parse_moves_taken(value) -> int | float:
    """Parse the 'moves taken' column value into a move count.

    Copied from po_pipeline.py.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (list, tuple)):
        return len(value)
    return len(ast.literal_eval(value))


def optimal_moves_for_tol_num(tol_num: int) -> int:
    """Return the optimal move count for a given TOL problem number (1–24).

    Copied from po_pipeline.py (embedded in build_problem_map).
    """
    if 1 <= tol_num <= 8:
        return 4
    if 9 <= tol_num <= 16:
        return 5
    if 17 <= tol_num <= 24:
        return 6
    raise ValueError(f"Unexpected TOL problem number: {tol_num}")


def build_problem_map(gridsearch_df: pd.DataFrame) -> pd.DataFrame:
    """Build a problem map from gridsearch data, assigning TOL-IDs and optimal moves.

    Copied/adapted from po_pipeline.py build_problem_map.
    """
    problem_map = (
        gridsearch_df[["init", "goal"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )
    problem_map["tol_num"] = np.arange(1, len(problem_map) + 1)
    problem_map["TOL-ID"] = problem_map["tol_num"].map(lambda v: f"TOL-ID-{v:02d}")
    problem_map["optimal_moves"] = problem_map["tol_num"].map(optimal_moves_for_tol_num)
    return problem_map


def rank_with_random_tie_break(
    df: pd.DataFrame,
    value_col: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rank problems by value_col with random tie-breaking.

    Copied from po_pipeline.py random_tie_resolved_order.
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


def compute_human_vs_model_tau(
    human_df: pd.DataFrame,
    gridsearch_df: pd.DataFrame,
    problem_map: pd.DataFrame,
    permutation_runs: int,
    seed: int,
) -> tuple[float, float]:
    """Compute Kendall tau between human and model problem rankings.

    For each permutation run, ties are broken randomly in both the human
    extra-moves ranking and the model extra-moves ranking, and the
    Kendall tau between the two rankings is recorded.

    Returns (mean_tau, std_tau).
    """
    rng = np.random.default_rng(seed)

    merged = human_df.merge(problem_map, on="TOL-ID", how="inner")

    merged = merged.merge(
        gridsearch_df[["init", "goal", "finished", "moves taken"]],
        on=["init", "goal"],
        how="left",
    )

    merged["moves_taken_count"] = merged["moves taken"].map(parse_moves_taken)
    merged["extra_moves_model"] = np.where(
        merged["finished"],
        merged["moves_taken_count"] - merged["optimal_moves"],
        np.nan,
    )

    tau_values: list[float] = []
    for _ in range(permutation_runs):
        human_ranking = rank_with_random_tie_break(merged, "exmov", rng)
        model_ranking = rank_with_random_tie_break(merged, "extra_moves_model", rng)
        tau, _ = kendalltau(human_ranking, model_ranking)
        tau_values.append(float(tau))

    return float(np.mean(tau_values)), float(np.std(tau_values))


def parse_gridsearch_params(filename: str) -> tuple[float, float]:
    """Extract alpha and beta from a gridsearch filename like
    'gridsearch_bmsa_alpha-0_beta-1.csv' or 'gridsearch_bmsa_alpha-0p5_beta-0p1.csv'.
    """
    match = re.search(r"alpha-([\d.]+(?:p\d+)?)_beta-([\d.]+(?:p\d+)?)", filename)
    if match is None:
        raise ValueError(f"Cannot parse alpha/beta from filename: {filename}")
    alpha_raw, beta_raw = match.group(1), match.group(2)
    return float(alpha_raw.replace("p", ".")), float(beta_raw.replace("p", "."))


def process_indiv_human_file(
    human_file: Path,
    gridsearch_folder: Path,
    permutation_runs: int,
    output_dir: Path,
    result_prefix: str,
    seed: int = 42,
) -> Path:
    """Process one human data CSV against all gridsearch alpha/beta combinations.

    Produces one long-form CSV with columns:
        alpha, beta, kendall_tau_mean, kendall_tau_std
    """
    human_df = pd.read_csv(human_file)

    required_cols = {"TOL-ID", "exmov"}
    missing = required_cols - set(human_df.columns)
    if missing:
        raise ValueError(
            f"Human file {human_file.name} missing required columns: {missing}"
        )

    gridsearch_files = sorted(gridsearch_folder.glob("*.csv"))
    if not gridsearch_files:
        raise FileNotFoundError(f"No gridsearch CSV files found in: {gridsearch_folder}")

    results = []
    for gs_file in gridsearch_files:
        alpha, beta = parse_gridsearch_params(gs_file.name)
        gs_df = pd.read_csv(gs_file)

        required_gs_cols = {"init", "goal", "finished", "moves taken"}
        missing_gs = required_gs_cols - set(gs_df.columns)
        if missing_gs:
            raise ValueError(
                f"Gridsearch file {gs_file.name} missing required columns: {missing_gs}"
            )

        problem_map = build_problem_map(gs_df)
        mean_tau, std_tau = compute_human_vs_model_tau(
            human_df, gs_df, problem_map, permutation_runs, seed
        )

        results.append(
            {
                "alpha": alpha,
                "beta": beta,
                "kendall_tau_mean": mean_tau,
                "kendall_tau_std": std_tau,
                "source_file": gs_file.name,
            }
        )

    result_df = pd.DataFrame(results).sort_values(["alpha", "beta"]).reset_index(drop=True)

    output_file = output_dir / f"{result_prefix}_{human_file.stem}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"Saved {output_file.name} ({len(result_df)} alpha/beta combinations)")
    return output_file


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare individual human Tower-of-London problem rankings against "
            "all gridsearch alpha/beta model combinations using Kendall tau."
        )
    )
    parser.add_argument(
        "--human_data_folder",
        type=Path,
        help="Folder containing human data CSV files (e.g. 0.csv … 50.csv)",
    )
    parser.add_argument(
        "--gridsearch_data_folder",
        type=Path,
        help="Folder containing gridsearch CSV files (one per alpha/beta combination)",
    )
    parser.add_argument(
        "--permutation-runs",
        type=int,
        default=1000,
        help="Number of random tie-resolution permutations per comparison (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory for result CSVs (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--result-prefix",
        type=str,
        default="indiv_results",
        help="Prefix for output filenames (default: indiv_results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Monte Carlo tie-breaking (default: 42)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    human_folder = args.human_data_folder
    if not human_folder.is_dir():
        raise FileNotFoundError(f"Human data folder does not exist: {human_folder}")

    gridsearch_folder = args.gridsearch_data_folder
    if not gridsearch_folder.is_dir():
        raise FileNotFoundError(f"Gridsearch data folder does not exist: {gridsearch_folder}")

    human_files = sorted(human_folder.glob("*.csv"))
    if not human_files:
        raise FileNotFoundError(f"No CSV files found in human data folder: {human_folder}")

    print(f"Found {len(human_files)} human data file(s) in {human_folder}")
    print(f"Gridsearch folder: {gridsearch_folder}")
    print(f"Output directory: {args.output_dir}")
    print(f"Permutation runs: {args.permutation_runs}")
    print(f"Result prefix: {args.result_prefix}")

    for human_file in human_files:
        print(f"\n--- Processing {human_file.name} ---")
        process_indiv_human_file(
            human_file,
            gridsearch_folder,
            permutation_runs=args.permutation_runs,
            output_dir=args.output_dir,
            result_prefix=args.result_prefix,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
