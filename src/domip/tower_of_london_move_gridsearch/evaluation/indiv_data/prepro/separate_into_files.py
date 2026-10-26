import pandas as pd
import argparse
from pathlib import Path


def row_to_csv(row: pd.Series, idx: int) -> pd.DataFrame:
    """Convert a single row (wide format exmov01..exmov24) to long format TOL-ID, exmov."""
    tol_ids = []
    exmovs = []
    for col in row.index:
        if col.startswith("exmov"):
            num = col.replace("exmov", "")
            tol_ids.append(f"TOL-ID-{int(num)}")
            exmovs.append(row[col])
    df = pd.DataFrame({"TOL-ID": tol_ids, "exmov": exmovs})
    df = df.sort_values("exmov").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Split a CSV into per-row CSV files (long format)."
    )
    parser.add_argument("input", type=Path, help="Input CSV file")
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for per-row CSV files"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    for idx, row in df.iterrows():
        out_df = row_to_csv(row, idx)
        out_path = args.output_dir / f"{idx}.csv"
        out_df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
