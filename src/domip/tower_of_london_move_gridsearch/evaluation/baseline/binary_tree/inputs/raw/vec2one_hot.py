"""
Convert problems_vector.csv (6-element vectors with values 0-3)
into problems_one_hot.csv where each vector becomes a 6x3 one-hot matrix.

Mapping:
  0 -> [0, 0, 0]  (empty / no ball)
  1 -> [1, 0, 0]  (ball 1)
  2 -> [0, 1, 0]  (ball 2)
  3 -> [0, 0, 1]  (ball 3)
"""

import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "..", "processed", "problems_vector.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "processed", "problems_one_hot.csv")


def vec_to_one_hot_matrix(vec: list[int]) -> list[list[int]]:
    """Convert a 6-element vector into a 6x3 one-hot matrix."""
    matrix = []
    for val in vec:
        if val == 0:
            matrix.append([0, 0, 0])
        elif val == 1:
            matrix.append([1, 0, 0])
        elif val == 2:
            matrix.append([0, 1, 0])
        elif val == 3:
            matrix.append([0, 0, 1])
        else:
            raise ValueError(f"Unexpected value {val} in vector (expected 0-3)")
    return matrix


def one_hot_matrix_to_str(matrix: list[list[int]]) -> str:
    """Format a 6x3 one-hot matrix as a JSON-like string for CSV."""
    return json.dumps(matrix)


def main():
    rows = []
    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_vec = json.loads(row["start_vector"])
            goal_vec = json.loads(row["goal_vector"])
            rows.append({
                "ID": row["ID"],
                "start_one_hot": one_hot_matrix_to_str(vec_to_one_hot_matrix(start_vec)),
                "goal_one_hot": one_hot_matrix_to_str(vec_to_one_hot_matrix(goal_vec)),
            })

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "start_one_hot", "goal_one_hot"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
