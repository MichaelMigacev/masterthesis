"""Extract excess-moves data for PD vs Stroke comparison.

Reads raw ANONYM_TOLCLIN_PD.csv and ANONYM_TOLCLIN_STRK.csv, extracts
moves (Z columns), solved (L columns) and age, computes excess moves,
filters to Patient rows only, and writes two CSVs with age as the first
column followed by exmov01–exmov24 and group.

Output files are written to:
  data/groups/PD_vs_Stroke/ANONYM_TOLCLIN_PD_exmov_patients.csv
  data/groups/PD_vs_Stroke/ANONYM_TOLCLIN_STRK_exmov_patients.csv

This file contains code written by AI.
"""

import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_BASE = os.path.join(
    SCRIPT_DIR,
    "..", "..", "..", "..", "..",
    "tower_of_london_human_data", "indiv_freiburg_data", "raw",
)

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "..", "data", "groups", "PD_vs_Stroke",
)

RAW_FILES = {
    "PD": os.path.join(RAW_BASE, "ANONYM_TOLCLIN_PD.csv"),
    "STRK": os.path.join(RAW_BASE, "ANONYM_TOLCLIN_STRK.csv"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_optimal_moves(problem_num: int) -> int:
    """Return the optimal number of moves for a given problem index (1-24)."""
    if 1 <= problem_num <= 8:
        return 4
    elif 9 <= problem_num <= 16:
        return 5
    elif 17 <= problem_num <= 24:
        return 6
    else:
        raise ValueError(f"Invalid problem number: {problem_num}")


def compute_exmoves(moves: int, solved: bool, problem_num: int) -> int:
    """Compute excess moves for a single problem."""
    if not solved:
        return 20
    return moves - get_optimal_moves(problem_num)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_group(raw_path: str) -> pd.DataFrame:
    """Read a raw CSV and return moves/solved/age/group DataFrame."""
    df = pd.read_csv(raw_path, sep=";", decimal=",", encoding="latin-1")

    # Extract age (keep original dtype – usually float)
    ages = df["age_exact"].copy()

    # Filter to Z (moves) and L (solved) columns for problems 05-28
    cols_to_keep = [
        col for col in df.columns
        if (col.startswith("Z") or col.startswith("L")) and col[1:].isdigit()
    ]
    df = df[cols_to_keep]

    new_df = pd.DataFrame()

    for i in range(1, 25):
        # Moves: Z05 -> moves01, ..., Z28 -> moves24
        move_src = f"Z{i + 4:02d}"
        if move_src in df.columns:
            new_df[f"moves{i:02d}"] = df[move_src].replace("#NULL!", 0)

        # Solved: L05 -> solved01, ..., L28 -> solved24
        solved_src = f"L{i + 4:02d}"
        if solved_src in df.columns:
            val = df[solved_src]
            new_df[f"solved{i:02d}"] = val.apply(
                lambda x: True if str(x).strip() in ["1", "2"] else False
            )

    # Add age and group columns
    new_df.insert(0, "age", ages.values)
    new_df["group"] = new_df.index.map(
        lambda x: "Patient" if x % 2 == 0 else "Control"
    )

    return new_df


def build_exmov_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build an exmov DataFrame from a moves/solved DataFrame.

    Columns: age, exmov01-exmov24, group
    """
    n_problems = 24
    exmov_columns = [f"exmov{str(i).zfill(2)}" for i in range(1, n_problems + 1)]

    rows = []
    for _, row in df.iterrows():
        exmov_values = [
            compute_exmoves(
                int(row[f"moves{str(i).zfill(2)}"]),
                bool(row[f"solved{str(i).zfill(2)}"]),
                i,
            )
            for i in range(1, n_problems + 1)
        ]
        rows.append([row["age"]] + exmov_values + [row["group"]])

    return pd.DataFrame(rows, columns=["age"] + exmov_columns + ["group"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label, raw_path in RAW_FILES.items():
        if not os.path.exists(raw_path):
            print(f"Warning: {raw_path} not found, skipping.", file=sys.stderr)
            continue

        print(f"Processing {label}...")

        # Step 1: extract moves/solved/age/group
        df = extract_group(raw_path)

        # Step 2: filter to patients only
        patient_df = df[df["group"] == "Patient"].reset_index(drop=True)
        print(f"  {label}: {len(patient_df)} patients extracted")

        # Step 3: compute excess moves
        exmov_df = build_exmov_df(patient_df)

        # Step 4: save
        out_filename = f"ANONYM_TOLCLIN_{label}_exmov_patients.csv"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        exmov_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
