"""Create per-subject ranking files from an exmov CSV.

Each row (subject) in the input produces one output CSV with columns
`TOL-ID` and `exmov`, sorted ascending by exmov (easiest problems first).
The output file is named `<row_index>.csv` so that row order from the
group-split files can still be matched later.

This file contains code written by AI.
"""

import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser(
        description="Create per-subject ranking files from an exmov CSV."
    )
    parser.add_argument("input_csv", help="Path to the input exmov CSV file.")
    parser.add_argument(
        "output_dir",
        help="Directory where the ranking CSV files will be written.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_csv, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        # Collect exmov column names (exmov01 .. exmov24)
        exmov_cols = [f for f in fieldnames if f.startswith("exmov")]
        exmov_cols.sort()  # ensure numerical order

        for row_index, row in enumerate(reader):
            # Build list of (TOL-ID, exmov) tuples
            entries = []
            for col in exmov_cols:
                problem_num = int(col.replace("exmov", ""))
                tol_id = f"TOL-ID-{problem_num}"
                exmov = int(row[col])
                entries.append((tol_id, exmov))

            # Sort by exmov ascending (least extra moves first)
            entries.sort(key=lambda x: x[1])

            # Write per-subject ranking file
            out_path = os.path.join(args.output_dir, f"{row_index}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(["TOL-ID", "exmov"])
                for tol_id, exmov in entries:
                    writer.writerow([tol_id, exmov])

    print(f"Wrote {row_index + 1} ranking files to {args.output_dir}")


if __name__ == "__main__":
    main()
