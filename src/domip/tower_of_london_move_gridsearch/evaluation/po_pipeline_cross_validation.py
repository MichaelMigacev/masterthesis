from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HEALTHY_ORDER_PATH = PROJECT_ROOT / "src" / "tower_of_london_human_data" / "problems_ordered_Healthy.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tower_of_london_plots" / "problem_ordering_outputs"


def normalize_tol_id(value: str) -> str:
	number = int(str(value).split("-")[-1])
	return f"TOL-ID-{number:02d}"


def to_bool_series(series: pd.Series) -> pd.Series:
	return series.map(lambda value: str(value).strip().lower() == "true")


def parse_moves_taken(value):
	if pd.isna(value):
		return np.nan
	if isinstance(value, (list, tuple)):
		return len(value)
	return len(ast.literal_eval(value))


def format_param(value: float) -> str:
	return str(value).replace("-", "m").replace(".", "p")


def tol_id_from_number(number: int) -> str:
	if not 1 <= number <= 24:
		raise ValueError(f"Problem number out of range [1, 24]: {number}")
	return f"TOL-ID-{number:02d}"


def collect_csv_files(inputs: Iterable[str]) -> list[Path]:
	csv_files: list[Path] = []
	seen: set[Path] = set()

	for raw_input in inputs:
		input_path = Path(raw_input)
		if not input_path.exists():
			raise FileNotFoundError(f"Input path does not exist: {input_path}")

		if input_path.is_dir():
			candidates = sorted(input_path.rglob("*.csv"))
		else:
			candidates = [input_path] if input_path.suffix.lower() == ".csv" else []

		for csv_path in candidates:
			resolved = csv_path.resolve()
			if resolved not in seen:
				seen.add(resolved)
				csv_files.append(resolved)

	csv_files.sort()
	return csv_files


def load_reference_rank_map(reference_ordering_path: Path) -> dict[str, int]:
	reference_df = pd.read_csv(reference_ordering_path)
	if "TOL-ID" not in reference_df.columns:
		raise ValueError(f"Expected column 'TOL-ID' in {reference_ordering_path.name}")

	reference_df = reference_df.copy()
	reference_df["TOL-ID"] = reference_df["TOL-ID"].map(normalize_tol_id)
	reference_df["reference_rank"] = np.arange(1, len(reference_df) + 1)
	return dict(zip(reference_df["TOL-ID"], reference_df["reference_rank"]))


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


def _extract_problem_numbers(cell_value: object) -> set[int]:
	return {int(match) for match in re.findall(r"\d+", str(cell_value))}


def load_problem_splits(problem_split_path: Path) -> list[tuple[set[str], set[str]]]:
	if not problem_split_path.exists():
		raise FileNotFoundError(f"Problem split file does not exist: {problem_split_path}")

	first_line = problem_split_path.read_text(encoding="utf-8").splitlines()[0] if problem_split_path.exists() else ""
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

		train_problem_ids = {tol_id_from_number(number) for number in train_numbers}
		validation_problem_ids = {tol_id_from_number(number) for number in validation_numbers}
		overlap = train_problem_ids & validation_problem_ids
		if overlap:
			raise ValueError(
				f"Row {row_index + 1} must not overlap between train and validation columns: "
				+ ", ".join(sorted(overlap))
			)
		if len(train_problem_ids) != len(validation_problem_ids):
			raise ValueError(
				f"Row {row_index + 1} must use the same number of training and validation problems"
			)

		split_instances.append((train_problem_ids, validation_problem_ids))

	if not split_instances:
		raise ValueError(f"No split rows were found in: {problem_split_path}")

	return split_instances

def build_problem_map(reference_csv: Path) -> pd.DataFrame:
	ref_df = pd.read_csv(reference_csv)
	required_cols = {"init", "goal", "num_moves", "finished", "moves taken", "alpha", "beta"}
	missing_cols = required_cols - set(ref_df.columns)
	if missing_cols:
		raise ValueError(f"Missing required columns in {reference_csv.name}: {missing_cols}")

	problem_map = (
		ref_df[["init", "goal"]]
		.drop_duplicates()
		.reset_index(drop=True)
		.copy()
	)
	problem_map["tol_num"] = np.arange(1, len(problem_map) + 1)
	problem_map["TOL-ID"] = problem_map["tol_num"].map(lambda value: f"TOL-ID-{value:02d}")

	if len(problem_map) != 24:
		print(f"Warning: expected 24 unique problems, got {len(problem_map)}")

	def optimal_moves_for_tol_num(tol_num: int) -> int:
		if 1 <= tol_num <= 8:
			return 4
		if 9 <= tol_num <= 16:
			return 5
		if 17 <= tol_num <= 24:
			return 6
		raise ValueError(f"Unexpected TOL problem number: {tol_num}")

	problem_map["optimal_moves"] = problem_map["tol_num"].map(optimal_moves_for_tol_num)
	return problem_map


def rank_problems_for_file(
	file_path: Path,
	problem_map: pd.DataFrame,
	metric_mode: str,
	selected_problem_ids: set[str] | None = None,
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

	df["finished"] = to_bool_series(df["finished"])
	df["moves_taken_count"] = df["moves taken"].map(parse_moves_taken)

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
) -> pd.DataFrame:
	summary_rows = []
	for csv_file in csv_files:
		ordered_df = rank_problems_for_file(csv_file, problem_map, metric_mode, selected_problem_ids)
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


def aggregate_pair_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
	return (
		summary_df.groupby(["alpha", "beta"], as_index=False)
		.agg(
			kendall_tau_random_tie_mean=("kendall_tau_random_tie_mean", "mean"),
			kendall_tau_random_tie_std=("kendall_tau_random_tie_std", "mean"),
			source_file_count=("source_file", "count"),
		)
		.sort_values(["kendall_tau_random_tie_mean", "kendall_tau_random_tie_std", "alpha", "beta"], ascending=[False, True, True, True])
		.reset_index(drop=True)
	)


def build_pair_grid(pair_summary_df: pd.DataFrame, value_column: str) -> pd.DataFrame:
	return (
		pair_summary_df.pivot(index="alpha", columns="beta", values=value_column)
		.sort_index()
		.sort_index(axis=1)
	)


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
	reference_ranks: np.ndarray,
	permutation_runs: int,
	seed: int,
) -> tuple[float, float]:
	rng = np.random.default_rng(seed)
	tau_values: list[float] = []

	for _ in range(permutation_runs):
		randomized_positions = random_tie_resolved_order(ordered_df, metric_col, rng)
		tau, _ = kendalltau(randomized_positions, reference_ranks)
		tau_values.append(float(tau))

	return float(np.mean(tau_values)), float(np.std(tau_values))


def make_heatmap(
	grid: pd.DataFrame,
	*,
	title: str,
	colorbar_label: str,
	output_path: Path,
	hide_zero_alpha_row: bool = True,
) -> None:
	plot_grid = grid.copy()
	if hide_zero_alpha_row and 0.0 in plot_grid.index:
		plot_grid = plot_grid.loc[plot_grid.index != 0.0]

	values = plot_grid.to_numpy(dtype=float)
	finite_values = values[np.isfinite(values)]
	if finite_values.size == 0:
		raise ValueError("Heatmap grid contains no finite values to plot")

	vmin = float(np.nanmin(finite_values))
	vmax = float(np.nanmax(finite_values))
	if np.isclose(vmin, vmax):
		pad = 0.5 if np.isfinite(vmin) else 1.0
		vmin -= pad
		vmax += pad

	fig, ax = plt.subplots(figsize=(11, 8))
	image = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)

	ax.set_xticks(np.arange(len(plot_grid.columns)))
	ax.set_yticks(np.arange(len(plot_grid.index)))
	ax.set_xticklabels([str(value) for value in plot_grid.columns], rotation=45, ha="right")
	ax.set_yticklabels([str(value) for value in plot_grid.index])
	ax.set_xlabel("beta")
	ax.set_ylabel("alpha")
	ax.set_title(title)

	for row_index in range(plot_grid.shape[0]):
		for col_index in range(plot_grid.shape[1]):
			value = plot_grid.iloc[row_index, col_index]
			label = "nan" if pd.isna(value) else f"{value:.2f}"
			ax.text(col_index, row_index, label, ha="center", va="center", color="white", fontsize=8)

	colorbar = fig.colorbar(image, ax=ax)
	colorbar.set_label(colorbar_label)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def process_collection(
	input_path: Path,
	*,
	csv_files: list[Path],
	reference_rank_maps: dict[str, int],
	output_dir: Path,
	image_tag: str,
	metric_mode: str,
	permutation_runs: int,
	seed: int,
	selected_problem_ids: set[str] | None,
) -> Path:
	if not csv_files:
		raise FileNotFoundError(f"No CSV files found for input: {input_path}")

	problem_map = build_problem_map(csv_files[0])
	summary_df = build_summary_dataframe(
		csv_files,
		problem_map,
		reference_rank_maps,
		metric_mode,
		permutation_runs,
		seed,
		selected_problem_ids,
	)
	pair_summary_df = aggregate_pair_summary(summary_df)
	grid = build_pair_grid(pair_summary_df, "kendall_tau_random_tie_mean")
	grid_std = build_pair_grid(pair_summary_df, "kendall_tau_random_tie_std")

	collection_name = input_path.stem if input_path.is_file() else input_path.name
	output_file = output_dir / f"{collection_name}__{image_tag}_{metric_mode}.png"
	output_csv = output_dir / f"{collection_name}__{image_tag}_{metric_mode}.csv"
	output_long_csv = output_dir / f"{collection_name}__{image_tag}_{metric_mode}_long.csv"
	title = (
		f"Average Kendall tau after random tie resolution ({permutation_runs} runs)\n"
		f"{collection_name} | {metric_mode}"
	)

	long_form_df = (
		pair_summary_df.sort_values(["alpha", "beta"])
		[["alpha", "beta", "kendall_tau_random_tie_mean", "kendall_tau_random_tie_std"]]
		.reset_index(drop=True)
	)
	long_form_df.to_csv(output_long_csv, index=False)

	make_heatmap(
		grid,
		title=title,
		colorbar_label="Mean Kendall tau",
		output_path=output_file,
		hide_zero_alpha_row=True,
	)

	# Create a heatmap visualising the standard deviations
	output_file_std = output_dir / f"{collection_name}__{image_tag}_{metric_mode}_std.png"
	make_heatmap(
		grid_std,
		title=(
			f"Stddev of Kendall tau after random tie resolution ({permutation_runs} runs)\n"
			f"{collection_name} | {metric_mode}"
		),
		colorbar_label="Stddev Kendall tau",
		output_path=output_file_std,
		hide_zero_alpha_row=True,
	)

	print(
		f"Saved {output_file.name}, {output_file_std.name}, and {output_long_csv.name} from {len(csv_files)} CSV files"
	)
	return output_file


def process_cross_validation_collection(
	input_path: Path,
	*,
	csv_files: list[Path],
	reference_rank_maps: list[dict[str, int]],
	reference_csv_files: list[Path],
	output_dir: Path,
	image_tag: str,
	metric_mode: str,
	permutation_runs: int,
	seed: int,
	split_instances: list[tuple[set[str], set[str]]],
	top_n: int,
) -> Path:
	if not csv_files:
		raise FileNotFoundError(f"No CSV files found for input: {input_path}")
	if not split_instances:
		raise ValueError(f"No split instances were provided for input: {input_path}")
	if len(reference_rank_maps) != len(reference_csv_files):
		raise ValueError("Mismatch between reference_rank_maps and reference_csv_files")

	problem_map = build_problem_map(csv_files[0])

	for split_index, (train_problem_ids, validation_problem_ids) in enumerate(split_instances, start=1):
		split_dir = output_dir / f"split_{split_index:03d}"
		split_dir.mkdir(parents=True, exist_ok=True)

		for reference_csv_file, reference_rank_map in zip(reference_csv_files, reference_rank_maps):
			human_dir = split_dir / reference_csv_file.stem
			human_dir.mkdir(parents=True, exist_ok=True)

			train_summary_df = build_summary_dataframe(
				csv_files,
				problem_map,
				reference_rank_map,
				metric_mode,
				permutation_runs,
				seed,
				train_problem_ids,
			)
			train_pair_summary_df = aggregate_pair_summary(train_summary_df)

			grid = build_pair_grid(train_pair_summary_df, "kendall_tau_random_tie_mean")
			grid_std = build_pair_grid(train_pair_summary_df, "kendall_tau_random_tie_std")

			make_heatmap(
				grid,
				title=f"Training Kendall tau (split {split_index}, {reference_csv_file.stem})",
				colorbar_label="Mean Kendall tau",
				output_path=human_dir / "grid_means.png",
				hide_zero_alpha_row=True,
			)

			make_heatmap(
				grid_std,
				title=f"Training Kendall tau stddev (split {split_index}, {reference_csv_file.stem})",
				colorbar_label="Stddev Kendall tau",
				output_path=human_dir / "grid_stds.png",
				hide_zero_alpha_row=True,
			)

			long_form_df = (
				train_pair_summary_df.sort_values(["alpha", "beta"])
				[["alpha", "beta", "kendall_tau_random_tie_mean", "kendall_tau_random_tie_std"]]
				.reset_index(drop=True)
			)
			long_form_df.to_csv(human_dir / "training_results_long.csv", index=False)

			top_train_pairs = train_pair_summary_df.head(top_n)[["alpha", "beta"]].copy()
			if top_train_pairs.empty:
				raise ValueError(f"No top training pairs for split {split_index}, {reference_csv_file.stem}")

			validation_summary_df = build_summary_dataframe(
				csv_files,
				problem_map,
				reference_rank_map,
				metric_mode,
				permutation_runs,
				seed,
				validation_problem_ids,
			)
			validation_pair_summary_df = aggregate_pair_summary(validation_summary_df)

			top_validation_results = top_train_pairs.merge(
				validation_pair_summary_df[["alpha", "beta", "kendall_tau_random_tie_mean", "kendall_tau_random_tie_std"]],
				on=["alpha", "beta"],
				how="left",
			)
			top_validation_results.columns = ["alpha", "beta", "mean", "std"]
			top_validation_results.to_csv(human_dir / "top5_validation.csv", index=False)

			print(
				f"Saved split {split_index}/{len(split_instances)}, {reference_csv_file.stem}: "
				f"grid_means.png, grid_stds.png, training_results_long.csv, top5_validation.csv"
			)

	return output_dir

def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Create Tower of London problem-ordering heatmaps from reference and model CSV folders."
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
		help="CSV file or folder containing the model ordering outputs to score",
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
		"--image-tag",
		required=True,
		help="Tag appended to each output image filename after the source name",
	)
	parser.add_argument(
		"--metric-mode",
		choices=["num_moves", "extra_number_of_moves"],
		default="extra_number_of_moves",
		help="Choose whether the ranking metric uses raw moves or extra moves over optimal",
	)
	parser.add_argument(
		"--permutation-runs",
		type=int,
		default=1000,
		help="Number of random tie-resolution permutations per alpha/beta pair",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed used for the Monte Carlo tie-breaking",
	)
	parser.add_argument(
		"--healthy-order-path",
		type=Path,
		default=None,
		help="Deprecated alias for --reference-orderings-path",
	)
	parser.add_argument(
		"--problem-selection-path",
		type=Path,
		default=None,
		help=(
			"Optional CSV/text file containing problem numbers in [1, 24]. "
			"Only selected problems are used in ranking and Kendall tau computation."
		),
	)
	parser.add_argument(
		"--cross-validation-path",
		type=Path,
		default=None,
		help=(
			"Optional split file with a left training column and a right validation column. "
			"When provided, the training column builds the grid and the validation column scores the top pairs."
		),
	)
	parser.add_argument(
		"--top-n",
		type=int,
		default=5,
		help="Number of top training alpha/beta pairs to validate on the held-out split",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	reference_input_path = args.reference_orderings_path
	if args.healthy_order_path is not None:
		reference_input_path = args.healthy_order_path

	reference_csv_files = collect_csv_files([str(reference_input_path)])
	reference_rank_maps = [load_reference_rank_map(reference_csv_file) for reference_csv_file in reference_csv_files]
	model_input_path = args.model_orderings_path
	selected_problem_ids = None
	split_instances = None
	if args.cross_validation_path is not None:
		split_instances = load_problem_splits(args.cross_validation_path)
		print(
			f"Loaded {len(split_instances)} split rows from {args.cross_validation_path.name}"
		)
	elif args.problem_selection_path is not None:
		selected_problem_ids = load_problem_selection(args.problem_selection_path)
		print(
			f"Loaded {len(selected_problem_ids)} selected problems from {args.problem_selection_path.name}: "
			f"{', '.join(sorted(selected_problem_ids))}"
		)

	output_dir = args.output_root / f"{args.folder_name}_{args.metric_mode}"
	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"Loaded {len(reference_rank_maps)} reference ordering CSV files from {reference_input_path}")
	print(f"Output directory: {output_dir}")

	if model_input_path.is_dir():
		model_csv_files = collect_csv_files([str(model_input_path)])
	elif model_input_path.is_file() and model_input_path.suffix.lower() == ".csv":
		model_csv_files = [model_input_path.resolve()]
	else:
		raise FileNotFoundError(f"Unsupported model ordering path: {model_input_path}")

	if split_instances is not None:
		process_cross_validation_collection(
			model_input_path,
			csv_files=model_csv_files,
			reference_rank_maps=reference_rank_maps,
			reference_csv_files=reference_csv_files,
			output_dir=output_dir,
			image_tag=args.image_tag,
			metric_mode=args.metric_mode,
			permutation_runs=args.permutation_runs,
			seed=args.seed,
			split_instances=split_instances,
			top_n=args.top_n,
		)
	else:
		process_collection(
			model_input_path,
			csv_files=model_csv_files,
			reference_rank_maps=reference_rank_maps[0] if reference_rank_maps else None,
			output_dir=output_dir,
			image_tag=args.image_tag,
			metric_mode=args.metric_mode,
			permutation_runs=args.permutation_runs,
			seed=args.seed,
			selected_problem_ids=selected_problem_ids,
		)


if __name__ == "__main__":
	main()
    # Get-ChildItem "src\tower_of_london_human_data\problems_ordered\*.csv" | ForEach-Object { python "src\domip\tower_of_london_move_gridsearch\evaluation\po_pipeline.py" "tower_of_london_plots\gridsearch_bf_md_scf_attresh0-1_0p05" --healthy-order-path $_.FullName --output-root "tower_of_london_plots" --folder-name "po_outputs" --image-tag $_.BaseName --metric-mode extra_number_of_moves --permutation-runs 1000 --seed 42 }