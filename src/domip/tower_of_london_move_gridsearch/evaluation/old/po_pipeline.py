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


def load_healthy_rank_map(healthy_order_path: Path) -> dict[str, int]:
	healthy_df = pd.read_csv(healthy_order_path)
	if "TOL-ID" not in healthy_df.columns:
		raise ValueError(f"Expected column 'TOL-ID' in {healthy_order_path.name}")

	healthy_df = healthy_df.copy()
	healthy_df["TOL-ID"] = healthy_df["TOL-ID"].map(normalize_tol_id)
	healthy_df["healthy_rank"] = np.arange(1, len(healthy_df) + 1)
	return dict(zip(healthy_df["TOL-ID"], healthy_df["healthy_rank"]))


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


def build_problem_map(reference_csv: Path, healthy_rank_map: dict[str, int]) -> pd.DataFrame:
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
	problem_map["healthy_rank"] = problem_map["TOL-ID"].map(healthy_rank_map)

	if problem_map["healthy_rank"].isna().any():
		missing_ids = problem_map.loc[problem_map["healthy_rank"].isna(), "TOL-ID"].tolist()
		raise ValueError(f"Missing healthy rank entries for TOL IDs: {missing_ids}")

	return problem_map


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

	df["finished"] = to_bool_series(df["finished"])
	df["moves_taken_count"] = df["moves taken"].map(parse_moves_taken)

	# Treat runs that exceeded the moves_till_failure as failures
	if moves_till_failure is not None:
		df["finished"] = df["finished"] & df["moves_taken_count"].le(moves_till_failure)

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

	aggregated["healthy_rank"] = aggregated["TOL-ID"].map(problem_map.set_index("TOL-ID")["healthy_rank"]).astype(int)
	aggregated["_sort_mean"] = aggregated[metric_label].fillna(np.inf)
	aggregated = aggregated.sort_values(["_sort_mean", "healthy_rank", "TOL-ID"], ascending=[True, True, True]).reset_index(drop=True)
	aggregated["ordered_position"] = np.arange(1, len(aggregated) + 1)

	return aggregated.drop(columns=["_sort_mean"])


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
		tau, _ = kendalltau(randomized_positions, ordered_df["healthy_rank"].to_numpy())
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
	healthy_rank_map: dict[str, int],
	output_dir: Path,
	image_tag: str,
	metric_mode: str,
	metric_tag: str,
	permutation_runs: int,
	seed: int,
	selected_problem_ids: set[str] | None,
	moves_till_failure: int | None = None,
) -> Path:
	if not csv_files:
		raise FileNotFoundError(f"No CSV files found for input: {input_path}")

	problem_map = build_problem_map(csv_files[0], healthy_rank_map)

	summary_rows = []
	for csv_file in csv_files:
		ordered_df = rank_problems_for_file(
			csv_file, problem_map, metric_mode, selected_problem_ids, moves_till_failure
		)
		metric_col = "mean_num_moves_on_success" if metric_mode == "num_moves" else "mean_extra_moves_on_success"
		mean_tau, std_tau = compute_monte_carlo_tau(ordered_df, metric_col, permutation_runs, seed)

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
		raise ValueError(f"No summary rows were built for input: {input_path}")

	grid = (
		summary_df.groupby(["alpha", "beta"], as_index=False)
		.agg(
			kendall_tau_random_tie_mean=("kendall_tau_random_tie_mean", "mean"),
			kendall_tau_random_tie_std=("kendall_tau_random_tie_std", "mean"),
		)
		.pivot(index="alpha", columns="beta", values="kendall_tau_random_tie_mean")
		.sort_index()
		.sort_index(axis=1)
	)

	# Also build a grid for the standard deviations so we can visualise them
	grid_std = (
		summary_df.groupby(["alpha", "beta"], as_index=False)
		.agg(kendall_tau_random_tie_std=("kendall_tau_random_tie_std", "mean"))
		.pivot(index="alpha", columns="beta", values="kendall_tau_random_tie_std")
		.sort_index()
		.sort_index(axis=1)
	)

	collection_name = input_path.stem if input_path.is_file() else input_path.name
	output_file = output_dir / f"{collection_name}__{image_tag}_{metric_tag}.png"
	output_csv = output_dir / f"{collection_name}__{image_tag}_{metric_tag}.csv"
	output_long_csv = output_dir / f"{collection_name}__{image_tag}_{metric_tag}_long.csv"
	title = (
		f"Average Kendall tau after random tie resolution ({permutation_runs} runs)\n"
		f"{collection_name} | {metric_mode}"
	)

	grid_to_save = grid.copy()
	grid_to_save.to_csv(output_csv, index=True)

	long_form_df = (
		summary_df.sort_values(["alpha", "beta"])
		[["alpha", "beta", "kendall_tau_random_tie_mean", "kendall_tau_random_tie_std"]]
		.reset_index(drop=True)
	)
	long_form_df.to_csv(output_long_csv, index=False)

	# Save std grid CSV and a long-form table for std as well
	output_csv_std = output_dir / f"{collection_name}__{image_tag}_{metric_tag}_std.csv"
	output_long_csv_std = output_dir / f"{collection_name}__{image_tag}_{metric_tag}_std_long.csv"
	grid_std.to_csv(output_csv_std, index=True)
	long_form_std_df = (
		summary_df.sort_values(["alpha", "beta"])
		[["alpha", "beta", "kendall_tau_random_tie_std"]]
		.reset_index(drop=True)
	)
	long_form_std_df.to_csv(output_long_csv_std, index=False)

	make_heatmap(
		grid,
		title=title,
		colorbar_label="Mean Kendall tau",
		output_path=output_file,
		hide_zero_alpha_row=True,
	)

	# Create a heatmap visualising the standard deviations
	output_file_std = output_dir / f"{collection_name}__{image_tag}_{metric_tag}_std.png"
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
		f"Saved {output_file.name}, {output_csv.name}, {output_long_csv.name}, {output_file_std.name}, {output_csv_std.name}, and {output_long_csv_std.name} from {len(csv_files)} CSV files"
	)
	return output_file


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Create Tower of London problem-ordering heatmaps from one or more CSV files or folders."
		)
	)
	parser.add_argument(
		"inputs",
		nargs="+",
		help="CSV files and/or folders containing CSV files to process",
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
		default=DEFAULT_HEALTHY_ORDER_PATH,
		help="Path to a healthy ordering CSV file or folder containing multiple reference ordering CSVs. Each file will be processed separately.",
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
		"--moves-till-failure",
		type=int,
		default=None,
		help="Treat finished runs with more than this many moves as failures before aggregation",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	input_paths = [Path(value) for value in args.inputs]
	
	# Load healthy ordering file(s) - either a single file or all CSVs in a folder
	healthy_order_path = args.healthy_order_path
	if not healthy_order_path.exists():
		raise FileNotFoundError(f"Healthy order path does not exist: {healthy_order_path}")
	
	if healthy_order_path.is_file():
		healthy_order_files = [healthy_order_path]
	else:
		healthy_order_files = collect_csv_files([str(healthy_order_path)])
		if not healthy_order_files:
			raise FileNotFoundError(f"No CSV files found in healthy order folder: {healthy_order_path}")
	
	selected_problem_ids = None
	if args.problem_selection_path is not None:
		selected_problem_ids = load_problem_selection(args.problem_selection_path)
		print(
			f"Loaded {len(selected_problem_ids)} selected problems from {args.problem_selection_path.name}: "
			f"{', '.join(sorted(selected_problem_ids))}"
		)

	metric_tag = "ex_mov" if args.metric_mode == "extra_number_of_moves" else args.metric_mode
	base_output_dir = args.output_root / args.folder_name
	base_output_dir.mkdir(parents=True, exist_ok=True)

	print(f"Output root: {base_output_dir}")
	print(f"Processing {len(healthy_order_files)} healthy ordering file(s)")

	for healthy_order_file in healthy_order_files:
		print(f"\n--- Processing reference ordering: {healthy_order_file.name} ---")
		healthy_rank_map = load_healthy_rank_map(healthy_order_file)
		output_dir = base_output_dir / healthy_order_file.stem
		output_dir.mkdir(parents=True, exist_ok=True)
		print(f"Output directory: {output_dir}")
		
		for input_path in input_paths:
			if input_path.is_dir():
				csv_files = collect_csv_files([str(input_path)])
			elif input_path.is_file() and input_path.suffix.lower() == ".csv":
				csv_files = [input_path.resolve()]
			else:
				raise FileNotFoundError(f"Unsupported input path: {input_path}")

			process_collection(
				input_path,
				csv_files=csv_files,
				healthy_rank_map=healthy_rank_map,
				output_dir=output_dir,
				image_tag=args.image_tag,
				metric_mode=args.metric_mode,
				metric_tag=metric_tag,
				permutation_runs=args.permutation_runs,
				seed=args.seed,
				selected_problem_ids=selected_problem_ids,
				moves_till_failure=args.moves_till_failure,
			)


if __name__ == "__main__":
	main()
    # Get-ChildItem "src\tower_of_london_human_data\problems_ordered\*.csv" | ForEach-Object { python "src\domip\tower_of_london_move_gridsearch\evaluation\po_pipeline.py" "tower_of_london_plots\gridsearch_bf_md_scf_attresh0-1_0p05" --healthy-order-path $_.FullName --output-root "tower_of_london_plots" --folder-name "po_outputs" --image-tag $_.BaseName --metric-mode extra_number_of_moves --permutation-runs 1000 --seed 42 }