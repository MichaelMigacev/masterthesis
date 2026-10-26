"""
Aggregate ranking CSVs from an input folder by computing the average
of numeric columns grouped by the index column (TOL-ID).

Usage:
    python average_ranking.py --input <folder> --output <folder> --prefix <prefix>
"""

import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Average ranking CSVs across all files in a folder."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input folder containing individual .csv ranking files.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output folder to write the averaged CSV to.",
    )
    parser.add_argument(
        "--prefix", "-p",
        default="average_ranking",
        help="Prefix for the output filename (default: average_ranking).",
    )
    args = parser.parse_args()

    # Collect all CSV files in the input folder
    csv_pattern = os.path.join(args.input, "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"No CSV files found in {args.input}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {args.input}")

    # Read and concatenate all DataFrames
    frames = []
    for path in sorted(csv_files):
        df = pd.read_csv(path)
        frames.append(df)
        print(f"  Loaded {os.path.basename(path)}  ({len(df)} rows)")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined: {len(combined)} total rows")

    # Identify the index column (first column) and numeric columns to average
    index_col = combined.columns[0]  # e.g. "TOL-ID"
    numeric_cols = combined.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        print("No numeric columns found to average.")
        return

    # Group by TOL-ID and compute mean of numeric columns across all CSVs
    averaged = combined.groupby(index_col, sort=False)[numeric_cols].mean().reset_index()

    # Sort by exmov ascending (lowest exmov = best rank → appears first)
    sort_col = "exmov" if "exmov" in averaged.columns else numeric_cols[0]
    averaged = averaged.sort_values(by=sort_col, ascending=True).reset_index(drop=True)

    # Write output
    os.makedirs(args.output, exist_ok=True)
    out_filename = f"{args.prefix}.csv"
    out_path = os.path.join(args.output, out_filename)
    averaged.to_csv(out_path, index=False)
    print(f"\nWrote averaged file ({len(averaged)} rows) to {out_path}")


if __name__ == "__main__":
    main()
