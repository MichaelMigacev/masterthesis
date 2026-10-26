from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from po_pipeline_cross_validation import (
	aggregate_pair_summary,
	build_problem_map,
	collect_csv_files,
	compute_monte_carlo_tau,
	load_reference_rank_map,
	parse_moves_taken,
	to_bool_series,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tower_of_london_plots" / "problem_ordering_outputs"


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Run the Tower of London problem-ordering analysis in one pass and export top-N "
			"training and validation results with per-problem solved/extra-moves cells."
		)
	)
	parser.add_argument(
		"--reference-orderings-path",
		type=Path,
		required=True,
		help="CSV file or folder containing the reference orderings used as Kendall tau targets",
	)
	parser.add_argument(
		"--model-orderings-path",
		type=Path,
		required=True,
		help="CSV file or folder containing the raw model ordering outputs",
	)
	parser.add_argument(
		"--cross-validation-path",
		type=Path,
		required=True,
		help="Split file with train/validation problem columns used for this analysis",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=DEFAULT_OUTPUT_ROOT,
		help="Root directory where result folders will be created",
	)
	parser.add_argument(
		"--folder-name",
		required=True,
		help="Output folder name created under the output root",
	)
	parser.add_argument(
		"--metric-mode",
		choices=["num_moves", "extra_number_of_moves"],
		default="extra_number_of_moves",
		help="Choose whether ranking uses raw moves or extra moves over optimal",
	)
	parser.add_argument(
		"--permutation-runs",
		type=int,
		default=1000,
		help="Number of random tie-resolution permutations used for Kendall tau recomputation",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed used for Monte Carlo tie-breaking",
	)
	parser.add_argument(
		"--top-n",
		type=int,
		default=25,
		help="Number of top training and validation alpha/beta pairs to export",
	)
	parser.add_argument(
		"--moves-till-failure",
		type=int,
		default=None,
		help="Treat finished runs with more than this many moves as failures before downstream aggregation",
	)
	return parser


def _normalize_training_long_columns(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
	column_map = {str(column).strip().lower(): column for column in df.columns}
	required = {"alpha", "beta"}
	if not required.issubset(column_map):
		raise ValueError(f"Missing alpha/beta in {csv_path}")

	if "kendall_tau_random_tie_mean" in column_map:
		mean_column = column_map["kendall_tau_random_tie_mean"]
	elif "mean" in column_map:
		mean_column = column_map["mean"]
	else:
		raise ValueError(f"Missing Kendall tau mean column in {csv_path}")

	if "kendall_tau_random_tie_std" in column_map:
		std_column = column_map["kendall_tau_random_tie_std"]
	elif "std" in column_map:
		std_column = column_map["std"]
	else:
		raise ValueError(f"Missing Kendall tau std column in {csv_path}")

	return pd.DataFrame(
		{
			"alpha": pd.to_numeric(df[column_map["alpha"]], errors="raise"),
			"beta": pd.to_numeric(df[column_map["beta"]], errors="raise"),
			"mean": pd.to_numeric(df[mean_column], errors="coerce"),
			"std": pd.to_numeric(df[std_column], errors="coerce"),
		}
	)


def _extract_problem_numbers(cell_value: object) -> set[int]:
	return {int(match) for match in re.findall(r"\d+", str(cell_value))}


def load_problem_splits(problem_split_path: Path) -> list[tuple[set[str], set[str]]]:
	if not problem_split_path.exists():
		raise FileNotFoundError(f"Problem split file does not exist: {problem_split_path}")

	lines = problem_split_path.read_text(encoding="utf-8").splitlines()
	first_line = lines[0] if lines else ""
	if re.search(r"[A-Za-z]", first_line):
		split_df = pd.read_csv(problem_split_path, dtype=str)
	else:
		split_df = pd.read_csv(problem_split_path, header=None, dtype=str)

	column_lookup = {str(column).strip().lower(): column for column in split_df.columns}
	if {"train", "validation"}.issubset(column_lookup):
		train_column = column_lookup["train"]
		validation_column = column_lookup["validation"]
	elif {"train_problems", "validation_problems"}.issubset(column_lookup):
		train_column = column_lookup["train_problems"]
		validation_column = column_lookup["validation_problems"]
	elif split_df.shape[1] >= 2:
		train_column = split_df.columns[0]
		validation_column = split_df.columns[1]
	else:
		raise ValueError(
			f"Problem split file must contain at least two columns (train and validation): {problem_split_path}"
		)

	split_instances: list[tuple[set[str], set[str]]] = []
	for row_index, row in split_df.iterrows():
		train_numbers = _extract_problem_numbers(row[train_column])
		validation_numbers = _extract_problem_numbers(row[validation_column])
		if not train_numbers and not validation_numbers:
			continue
		if not train_numbers:
			raise ValueError(f"Row {row_index + 1} has no training problems in {problem_split_path.name}")
		if not validation_numbers:
			raise ValueError(f"Row {row_index + 1} has no validation problems in {problem_split_path.name}")

		invalid_train = [number for number in sorted(train_numbers) if number < 1 or number > 24]
		invalid_validation = [number for number in sorted(validation_numbers) if number < 1 or number > 24]
		if invalid_train or invalid_validation:
			raise ValueError(
				"Problem split contains values outside [1, 24]: "
				+ ", ".join(str(value) for value in sorted(set(invalid_train + invalid_validation)))
			)

		train_problem_ids = {f"TOL-ID-{number:02d}" for number in train_numbers}
		validation_problem_ids = {f"TOL-ID-{number:02d}" for number in validation_numbers}
		overlap = train_problem_ids & validation_problem_ids
		if overlap:
			raise ValueError(
				f"Row {row_index + 1} must not overlap between train and validation columns: "
				+ ", ".join(sorted(overlap))
			)

		split_instances.append((train_problem_ids, validation_problem_ids))

	if not split_instances:
		raise ValueError(f"No split rows were found in: {problem_split_path}")

	return split_instances


def _apply_moves_till_failure(df: pd.DataFrame, moves_till_failure: int | None) -> pd.DataFrame:
	work = df.copy()
	work["finished"] = to_bool_series(work["finished"])
	work["moves_taken_count"] = work["moves taken"].map(parse_moves_taken)
	if moves_till_failure is not None:
		work["finished"] = work["finished"] & work["moves_taken_count"].le(moves_till_failure)
	return work


def build_per_problem_metrics(
	model_csv_files: list[Path],
	problem_map: pd.DataFrame,
	metric_mode: str,
	moves_till_failure: int | None,
) -> pd.DataFrame:
	rows: list[pd.DataFrame] = []
	for csv_file in model_csv_files:
		df = pd.read_csv(csv_file)
		required_cols = {"init", "goal", "finished", "moves taken", "alpha", "beta"}
		missing = required_cols - set(df.columns)
		if missing:
			raise ValueError(f"{csv_file.name} missing required columns: {missing}")

		df = df.merge(problem_map, on=["init", "goal"], how="left")
		if df["TOL-ID"].isna().any():
			missing_pairs = df.loc[df["TOL-ID"].isna(), ["init", "goal"]].drop_duplicates()
			raise ValueError(f"Unknown problems in {csv_file.name}:\n{missing_pairs}")

		df = _apply_moves_till_failure(df, moves_till_failure)
		if metric_mode == "extra_number_of_moves":
			df["metric_value"] = np.where(
				df["finished"],
				df["moves_taken_count"] - df["optimal_moves"],
				np.nan,
			)
		else:
			df["metric_value"] = np.where(df["finished"], df["moves_taken_count"], np.nan)

		rows.append(df[["alpha", "beta", "TOL-ID", "finished", "metric_value"]])

	if not rows:
		raise ValueError("No model rows available to build per-problem metrics")

	full_df = pd.concat(rows, ignore_index=True)
	per_problem = (
		full_df.groupby(["alpha", "beta", "TOL-ID"], as_index=False)
		.agg(
			success_count=("finished", "sum"),
			total_runs=("finished", "size"),
			mean_metric_on_success=("metric_value", "mean"),
		)
	)
	per_problem["solved"] = per_problem["success_count"] > 0
	return per_problem


def _pair_key(alpha: float, beta: float) -> tuple[float, float]:
	return (round(float(alpha), 12), round(float(beta), 12))


def _problem_cell(solved: bool, value: float) -> str:
	if not solved or pd.isna(value):
		return "False|NaN"
	return f"True|{float(value):.6g}"


def attach_problem_columns(
	base_df: pd.DataFrame,
	per_problem_df: pd.DataFrame,
	problem_ids: set[str],
) -> pd.DataFrame:
	lookup = {
		(_pair_key(row.alpha, row.beta), row["TOL-ID"]): (bool(row["solved"]), row["mean_metric_on_success"])
		for _, row in per_problem_df.iterrows()
	}

	result_df = base_df.copy()
	for problem_id in sorted(problem_ids):
		column_values: list[str] = []
		for _, row in result_df.iterrows():
			pair = _pair_key(row["alpha"], row["beta"])
			solved, metric = lookup.get((pair, problem_id), (False, np.nan))
			column_values.append(_problem_cell(solved, metric))
		result_df[problem_id] = column_values

	return result_df


def rank_problems_for_file(
	file_path: Path,
	problem_map: pd.DataFrame,
	metric_mode: str,
	selected_problem_ids: set[str] | None = None,
	moves_till_failure: int | None = None,
) -> pd.DataFrame:
	df = pd.read_csv(file_path)

	required_cols = {"init", "goal", "num_moves", "finished", "moves taken", "alpha", "beta"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"{file_path.name} missing required columns: {missing}")

	df = df.merge(problem_map, on=["init", "goal"], how="left")
	if df["TOL-ID"].isna().any():
		missing_pairs = df.loc[df["TOL-ID"].isna(), ["init", "goal"]].drop_duplicates()
		raise ValueError(f"Unknown problems in {file_path.name}:\n{missing_pairs}")

	if "optimal_moves" not in df.columns:
		raise ValueError("Expected 'optimal_moves' to be present from the TOL-ID mapping step")

	if selected_problem_ids is not None:
		df = df[df["TOL-ID"].isin(selected_problem_ids)].copy()
		if df.empty:
			raise ValueError(
				f"No selected problems found in {file_path.name}. Selection size: {len(selected_problem_ids)}"
			)

	df = _apply_moves_till_failure(df, moves_till_failure)

	if metric_mode == "num_moves":
		metric_column = "moves_taken_count"
		metric_label = "mean_num_moves_on_success"
		df[metric_label] = np.where(df["finished"], df[metric_column], np.nan)
	elif metric_mode == "extra_number_of_moves":
		metric_column = "extra_moves_on_success"
		metric_label = "mean_extra_moves_on_success"
		df[metric_column] = np.where(
			df["finished"],
			df["moves_taken_count"] - df["optimal_moves"],
			np.nan,
		)
	else:
		raise ValueError("metric_mode must be 'num_moves' or 'extra_number_of_moves'")

	aggregated = (
		df.groupby(["alpha", "beta", "TOL-ID"], as_index=False)
		.agg(
			**{
				metric_label: (metric_column if metric_mode == "extra_number_of_moves" else metric_label, "mean"),
				"success_count": ("finished", "sum"),
				"total_runs": ("finished", "size"),
			}
		)
	)

	aggregated["_sort_mean"] = aggregated[metric_label].fillna(np.inf)
	aggregated = aggregated.sort_values(["_sort_mean", "TOL-ID"], ascending=[True, True]).reset_index(drop=True)
	aggregated["ordered_position"] = np.arange(1, len(aggregated) + 1)

	return aggregated.drop(columns=["_sort_mean"])


def build_summary_dataframe(
	csv_files: list[Path],
	problem_map: pd.DataFrame,
	reference_rank_map: dict[str, int],
	metric_mode: str,
	permutation_runs: int,
	seed: int,
	selected_problem_ids: set[str] | None,
	moves_till_failure: int | None,
) -> pd.DataFrame:
	summary_rows: list[dict[str, object]] = []
	for csv_file in csv_files:
		ordered_df = rank_problems_for_file(
			csv_file,
			problem_map,
			metric_mode,
			selected_problem_ids,
			moves_till_failure,
		)
		metric_col = "mean_num_moves_on_success" if metric_mode == "num_moves" else "mean_extra_moves_on_success"
		reference_ranks = ordered_df["TOL-ID"].map(reference_rank_map)
		if reference_ranks.isna().any():
			missing_ids = ordered_df.loc[reference_ranks.isna(), "TOL-ID"].unique().tolist()
			raise ValueError(
				f"Reference ordering is missing TOL IDs required by {csv_file.name}: {missing_ids}"
			)
		mean_tau, std_tau = compute_monte_carlo_tau(
			ordered_df,
			metric_col,
			reference_ranks.to_numpy(dtype=int),
			permutation_runs,
			seed,
		)

		summary_rows.append(
			{
				"alpha": float(ordered_df["alpha"].iloc[0]),
				"beta": float(ordered_df["beta"].iloc[0]),
				"kendall_tau_random_tie_mean": mean_tau,
				"kendall_tau_random_tie_std": std_tau,
				"source_file": csv_file.name,
			}
		)

	summary_df = pd.DataFrame(summary_rows)
	if summary_df.empty:
		raise ValueError("No summary rows were built")

	return summary_df


def _top_n_from_long_frame(long_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
	return (
		long_df.sort_values(["mean", "std", "alpha", "beta"], ascending=[False, True, True, True])
		.head(top_n)
		.reset_index(drop=True)
	)


def _summary_to_long_df(summary_df: pd.DataFrame) -> pd.DataFrame:
	return (
		summary_df[["alpha", "beta", "kendall_tau_random_tie_mean", "kendall_tau_random_tie_std"]]
		.rename(
			columns={
				"kendall_tau_random_tie_mean": "mean",
				"kendall_tau_random_tie_std": "std",
			}
		)
		.reset_index(drop=True)
	)


def main() -> None:
	args = build_arg_parser().parse_args()

	reference_csv_files = collect_csv_files([str(args.reference_orderings_path)])
	if not reference_csv_files:
		raise FileNotFoundError(f"No reference CSV files found in {args.reference_orderings_path}")

	model_csv_files = collect_csv_files([str(args.model_orderings_path)])
	if not model_csv_files:
		raise FileNotFoundError(f"No model CSV files found in {args.model_orderings_path}")

	split_instances = load_problem_splits(args.cross_validation_path)
	reference_rank_maps = [load_reference_rank_map(path) for path in reference_csv_files]
	problem_map = build_problem_map(model_csv_files[0])
	per_problem_df = build_per_problem_metrics(
		model_csv_files,
		problem_map,
		args.metric_mode,
		args.moves_till_failure,
	)

	output_dir = args.output_root / f"{args.folder_name}_{args.metric_mode}"
	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"Loaded {len(reference_csv_files)} reference ordering CSV files from {args.reference_orderings_path}")
	print(f"Loaded {len(split_instances)} split rows from {args.cross_validation_path.name}")
	print(f"Output directory: {output_dir}")

	threshold_suffix = f"_mtf{args.moves_till_failure}" if args.moves_till_failure is not None else ""

	for split_index, (train_problem_ids, validation_problem_ids) in enumerate(split_instances, start=1):
		split_dir = output_dir / f"split_{split_index:03d}"
		split_dir.mkdir(parents=True, exist_ok=True)

		for reference_csv_file, reference_rank_map in zip(reference_csv_files, reference_rank_maps):
			human_dir = split_dir / reference_csv_file.stem
			human_dir.mkdir(parents=True, exist_ok=True)

			train_summary_df = build_summary_dataframe(
				model_csv_files,
				problem_map,
				reference_rank_map,
				args.metric_mode,
				args.permutation_runs,
				args.seed,
				train_problem_ids,
				args.moves_till_failure,
			)
			train_pair_summary_df = aggregate_pair_summary(train_summary_df)
			train_long_df = _summary_to_long_df(train_pair_summary_df)
			train_long_df.to_csv(human_dir / "training_results_long.csv", index=False)

			# Compute validation summaries only for the top training alpha/beta pairs.
			# Filter model CSVs to those matching the top training pairs to avoid
			# computing a full validation long form for all parameterizations.
			top_training_df = _top_n_from_long_frame(train_long_df, args.top_n)
			# Build a set of canonical pair keys for quick membership tests
			top_pair_keys = {_pair_key(float(row.alpha), float(row.beta)) for _, row in top_training_df.iterrows()}

			def _filter_files_for_pairs(csv_files: list[Path], pair_keys: set[tuple[float, float]]) -> list[Path]:
				selected: list[Path] = []
				for f in csv_files:
					try:
						hdr = pd.read_csv(f, usecols=["alpha", "beta"], nrows=1)
						akey = _pair_key(hdr["alpha"].iloc[0], hdr["beta"].iloc[0])
						if akey in pair_keys:
							selected.append(f)
					except Exception:
						continue
				return selected

			selected_model_files = _filter_files_for_pairs(model_csv_files, top_pair_keys)
			if not selected_model_files:
				# Fallback: if no individual model files matched, compute validation over all files
				# but do not write the long form to disk.
				selected_model_files = model_csv_files

			validation_summary_df = build_summary_dataframe(
				selected_model_files,
				problem_map,
				reference_rank_map,
				args.metric_mode,
				args.permutation_runs,
				args.seed,
				validation_problem_ids,
				args.moves_till_failure,
			)
			validation_pair_summary_df = aggregate_pair_summary(validation_summary_df)
			validation_long_df = _summary_to_long_df(validation_pair_summary_df)

			top_training_df = _top_n_from_long_frame(train_long_df, args.top_n)
			top_validation_df = _top_n_from_long_frame(validation_long_df, args.top_n)

			top_training_extended = attach_problem_columns(top_training_df, per_problem_df, train_problem_ids)
			top_validation_extended = attach_problem_columns(top_validation_df, per_problem_df, validation_problem_ids)

			training_out = human_dir / f"top{args.top_n}_training_extended{threshold_suffix}.csv"
			validation_out = human_dir / f"top{args.top_n}_validation_extended{threshold_suffix}.csv"
			top_training_extended.to_csv(training_out, index=False)
			top_validation_extended.to_csv(validation_out, index=False)

			print(
				f"Saved split {split_index}/{len(split_instances)}, {reference_csv_file.stem}: "
				f"{training_out.name}, {validation_out.name}"
			)


if __name__ == "__main__":
	main()