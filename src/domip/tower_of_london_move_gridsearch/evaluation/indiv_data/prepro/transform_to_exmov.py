"""Transform Tower of London CSV into excess moves (exmov) format.

For each problem (01-24):
  - If solved is False, exmov = 20 (penalty).
  - If solved is True, exmov = moves - optimal_moves
    where optimal moves are:
      problems 01-08 -> 4
      problems 09-16 -> 5
      problems 17-24 -> 6
"""

import argparse
import csv
import sys


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
    """Compute excess moves for a single problem.

    Args:
        moves: Number of moves the subject used.
        solved: Whether the problem was solved successfully.
        problem_num: Problem index (1-24).

    Returns:
        Excess moves (moves - optimal) if solved, else 20.
    """
    if not solved:
        return 20
    return moves - get_optimal_moves(problem_num)


def parse_bool(value: str) -> bool:
    """Parse a string boolean value (True/False)."""
    return value.strip().lower() == "true"


def main():
    parser = argparse.ArgumentParser(
        description="Transform Tower of London CSV to excess moves format."
    )
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("output_csv", help="Path to the output CSV file.")
    args = parser.parse_args()

    n_problems = 24
    exmov_columns = [f"exmov{str(i).zfill(2)}" for i in range(1, n_problems + 1)]

    with open(args.input_csv, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        # Validate expected columns exist
        expected = {f"moves{str(i).zfill(2)}" for i in range(1, n_problems + 1)}
        expected |= {f"solved{str(i).zfill(2)}" for i in range(1, n_problems + 1)}
        expected.add("group")
        missing = expected - set(reader.fieldnames or [])
        if missing:
            print(f"Error: missing columns in input CSV: {missing}", file=sys.stderr)
            sys.exit(1)

        with open(args.output_csv, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(exmov_columns + ["group"])

            for row in reader:
                exmov_row = []
                for i in range(1, n_problems + 1):
                    suffix = str(i).zfill(2)
                    moves = int(row[f"moves{suffix}"])
                    solved = parse_bool(row[f"solved{suffix}"])
                    exmov_row.append(compute_exmoves(moves, solved, i))
                exmov_row.append(row["group"])
                writer.writerow(exmov_row)

    print(f"Output written to {args.output_csv}")


if __name__ == "__main__":
    main()
