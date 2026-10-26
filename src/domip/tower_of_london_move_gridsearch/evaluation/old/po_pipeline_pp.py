from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tower_of_london_plots" / "problem_ordering_outputs"


def normalize_tol_id(value: str) -> str:
	number = int(str(value).split("-")[-1])
	return f"TOL-ID-{number:02d}"





def tol_id_from_number(number: int) -> str:
	if not 1 <= number <= 24:
		raise ValueError(f"Problem number out of range [1, 24]: {number}")
	return f"TOL-ID-{number:02d}"


def load_metric_csv(metric_csv_path: Path) -> pd.DataFrame:
	"""Load a CSV with TOL-ID and metric columns.
	
	Expects first column to be TOL-ID, second to be a metric value.
	Returns DataFrame with columns ['TOL-ID', 'metric'].
	"""
	if not metric_csv_path.exists():
		raise FileNotFoundError(f"Metric CSV does not exist: {metric_csv_path}")
	
	df = pd.read_csv(metric_csv_path)
	if len(df.columns) < 2:
		raise ValueError(f"CSV must have at least 2 columns, got {len(df.columns)}")
	
	# Use first two columns
	tol_id_col = df.columns[0]
	metric_col = df.columns[1]
	
	result = df[[tol_id_col, metric_col]].copy()
	result.columns = ["TOL-ID", "metric"]
	
	# Normalize TOL-ID format
	result["TOL-ID"] = result["TOL-ID"].map(normalize_tol_id)
	
	return result


def collect_human_csv_files(input_path: Path) -> list[Path]:
	"""Collect CSV files from a folder or return single CSV as a list (human orderings)."""
	if input_path.is_dir():
		csv_files = sorted(input_path.glob("*.csv"))
		if not csv_files:
			raise FileNotFoundError(f"No CSV files found in folder: {input_path}")
		return csv_files
	elif input_path.is_file() and input_path.suffix.lower() == ".csv":
		return [input_path.resolve()]
	else:
		raise FileNotFoundError(f"Path is neither a folder nor a CSV file: {input_path}")


def load_human_rank_map(human_order_path: Path) -> dict[str, int]:
	human_df = pd.read_csv(human_order_path)
	if "TOL-ID" not in human_df.columns:
		raise ValueError(f"Expected column 'TOL-ID' in {human_order_path.name}")

	human_df = human_df.copy()
	human_df["TOL-ID"] = human_df["TOL-ID"].map(normalize_tol_id)
	human_df["human_rank"] = np.arange(1, len(human_df) + 1)
	return dict(zip(human_df["TOL-ID"], human_df["human_rank"]))


def load_problem_selection(problem_selection_path: Path) -> set[str]:
	if not problem_selection_path.exists():
		raise FileNotFoundError(f"Problem selection file does not exist: {problem_selection_path}")

	text = problem_selection_path.read_text(encoding="utf-8")
	number_strings = re.findall(r"\d+", text)
	if not number_strings:
		raise ValueError(
			f"Problem selection file must contain at least one integer in [1, 24]: {problem_selection_path}"
		)

	problem_numbers = sorted({int(value) for value in number_strings})
	invalid_numbers = [number for number in problem_numbers if number < 1 or number > 24]
	if invalid_numbers:
		raise ValueError(
			"Problem selection contains values outside [1, 24]: "
			+ ", ".join(str(value) for value in invalid_numbers)
		)

	return {tol_id_from_number(number) for number in problem_numbers}


def process_metric_and_compute_tau(
	metric_df: pd.DataFrame,
	*,
	human_rank_map: dict[str, int],
	selected_problem_ids: set[str] | None,
	permutation_runs: int,
	seed: int,
) -> tuple[float, float]:
	"""Process a metric CSV and compute mean and std of Kendall tau vs. human rank.
	
	Filters by selected problems, sorts by metric with tie-breaking permutations,
	and computes Kendall tau correlation with the human_rank_map.
	"""
	# Join metric values with human ranks
	df = metric_df.copy()
	# Filter by selected problems if provided
	if selected_problem_ids is not None:
		df = df[df["TOL-ID"].isin(selected_problem_ids)].copy()
		if df.empty:
			raise ValueError(
				f"No selected problems found in metric CSV. Selection size: {len(selected_problem_ids)}"
			)

	# Attach human rank for each TOL-ID (reference ranking)
	df["human_rank"] = df["TOL-ID"].map(human_rank_map)
	if df["human_rank"].isna().any():
		missing_ids = df.loc[df["human_rank"].isna(), "TOL-ID"].tolist()
		raise ValueError(f"Missing human rank entries for TOL IDs: {missing_ids}")

	# Compute Kendall tau with random tie resolution applied to metric values
	mean_tau, std_tau = compute_monte_carlo_tau(
		df,
		"metric",
		permutation_runs,
		seed,
	)
	return mean_tau, std_tau

def random_tie_resolved_order(df: pd.DataFrame, value_col: str, rng: np.random.Generator) -> np.ndarray:
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
) -> tuple[float, float]:
	rng = np.random.default_rng(seed)
	tau_values: list[float] = []

	for _ in range(permutation_runs):
		randomized_positions = random_tie_resolved_order(ordered_df, metric_col, rng)
		tau, _ = kendalltau(randomized_positions, ordered_df["human_rank"].to_numpy())
		tau_values.append(float(tau))

	return float(np.mean(tau_values)), float(np.std(tau_values))





def process_metric_csv(
	metric_csv_path: Path,
	*,
	human_rank_map: dict[str, int],
	output_dir: Path,
	permutation_runs: int,
	seed: int,
	selected_problem_ids: set[str] | None,
) -> tuple[float, float, Path]:
	"""Compatibility wrapper: load metric CSV and compute tau against a human ordering map.

	Returns (mean_tau, std_tau, output_csv_path).
	"""
	metric_df = load_metric_csv(metric_csv_path)
	mean_tau, std_tau = process_metric_and_compute_tau(
		metric_df,
		human_rank_map=human_rank_map,
		selected_problem_ids=selected_problem_ids,
		permutation_runs=permutation_runs,
		seed=seed,
	)

	# Save results to CSV
	output_dir.mkdir(parents=True, exist_ok=True)
	output_csv = output_dir / f"{metric_csv_path.stem}_tau_results.csv"
	result_df = pd.DataFrame({
		"metric": ["kendall_tau"],
		"mean": [mean_tau],
		"std": [std_tau],
	})
	result_df.to_csv(output_csv, index=False)
	return mean_tau, std_tau, output_csv


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Analyze metric CSV(s) (TOL-ID + metric columns) and compute Kendall tau "
			"against a reference ranking (e.g., healthy ordering) with random tie-breaking permutations. "
			"Accepts either a single CSV or a folder of CSVs."
		)
	)
	parser.add_argument(
		"input",
		type=Path,
		help="CSV file or folder containing CSVs with TOL-ID (first column) and metric (second column)",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=DEFAULT_OUTPUT_ROOT,
		help="Root directory where result folder will be created",
	)
	parser.add_argument(
		"--output-folder",
		required=True,
		help="Output folder name created under the output root",
	)
	parser.add_argument(
		"--permutation-runs",
		type=int,
		default=1000,
		help="Number of random tie-resolution permutations",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed used for the Monte Carlo tie-breaking",
	)
	parser.add_argument(
		"--metric-path",
		type=Path,
		required=True,
		help="Path to the single metric CSV (TOL-ID + metric column) to compare against human orderings",
	)
	parser.add_argument(
		"--problem-selection-path",
		type=Path,
		default=None,
		help=(
			"Optional CSV/text file containing problem numbers in [1, 24]. "
			"Only selected problems are used in Kendall tau computation."
		),
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	input_path = args.input.resolve()
	if not input_path.exists():
		raise FileNotFoundError(f"Input path not found: {input_path}")

	metric_path = args.metric_path.resolve()
	if not metric_path.exists():
		raise FileNotFoundError(f"Metric CSV not found: {metric_path}")

	# Load metric once (single metric per run)
	metric_df = load_metric_csv(metric_path)

	selected_problem_ids = None
	if args.problem_selection_path is not None:
		selected_problem_ids = load_problem_selection(args.problem_selection_path)
		print(
			f"Loaded {len(selected_problem_ids)} selected problems from {args.problem_selection_path.name}: "
			f"{', '.join(sorted(selected_problem_ids))}"
		)

	output_dir = args.output_root / f"{args.output_folder}"
	print(f"Output directory: {output_dir}")

	# Collect human CSV files
	human_csv_files = collect_human_csv_files(input_path)
	print(f"Processing {len(human_csv_files)} human CSV file(s)...")
	
	# Process each human ordering file and collect results
	results = []
	processed = 0
	for human_file in human_csv_files:
		# quick validation: ensure file has a TOL-ID column
		try:
			cols = pd.read_csv(human_file, nrows=0).columns
		except Exception as e:
			print(f"Skipping {human_file.name}: cannot read CSV ({e})")
			continue
		if "TOL-ID" not in cols:
			print(f"Skipping {human_file.name}: missing 'TOL-ID' column")
			continue

		# load human ordering and build rank map
		human_rank_map = load_human_rank_map(human_file)

		mean_tau, std_tau = process_metric_and_compute_tau(
			metric_df,
			human_rank_map=human_rank_map,
			selected_problem_ids=selected_problem_ids,
			permutation_runs=args.permutation_runs,
			seed=args.seed,
		)

		# Save per-human result
		output_dir.mkdir(parents=True, exist_ok=True)
		output_csv = output_dir / f"{human_file.stem}_tau_results.csv"
		pd.DataFrame({"metric": ["kendall_tau"], "mean": [mean_tau], "std": [std_tau]}).to_csv(output_csv, index=False)

		results.append({
			"human_file": human_file.name,
			"mean_tau": mean_tau,
			"std_tau": std_tau,
			"output_file": output_csv.name,
		})
		processed += 1
		print(f"  {human_file.name}: mean_tau={mean_tau:.4f}, std_tau={std_tau:.4f}")

	if processed == 0:
		raise RuntimeError("No valid human ordering CSVs were processed (no files with 'TOL-ID' column found).")

	# Save summary CSV
	summary_df = pd.DataFrame(results)
	summary_csv = output_dir / "summary.csv"
	summary_df.to_csv(summary_csv, index=False)
	print(f"\nSaved summary: {summary_csv.name}")


if __name__ == "__main__":
	main()
    