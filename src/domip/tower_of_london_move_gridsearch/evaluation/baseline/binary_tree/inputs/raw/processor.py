"""
Convert Tower of London problems from state representation (TOL-XXX IDs)
into a 3x3 matrix representation.

Matrix layout:
  - Row 0 = Peg 0 (height 1): [bottom, 0, 0]
  - Row 1 = Peg 1 (height 2): [bottom, top, 0]
  - Row 2 = Peg 2 (height 3): [bottom, middle, top]
  - 0 = no bead, 1 = bead 0, 2 = bead 1, 3 = bead 2
"""

import csv
import sys
import os

# Import OFFICIAL_STATES from state.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state import OFFICIAL_STATES


def state_id_to_pegs(state_id: str) -> tuple:
    """Look up a TOL state ID and return the pegs tuple."""
    for key, sid in OFFICIAL_STATES.items():
        if sid == state_id:
            return key
    raise ValueError(f"Unknown state ID: {state_id}")


def pegs_to_matrix(pegs: tuple) -> list:
    """
    Convert pegs tuple to a 3x3 matrix.

    Each peg is a list of beads (bottom to top), beads are 0-indexed.
    Matrix uses 0 for empty and bead+1 for bead presence.
    """
    matrix = []
    for peg in pegs:
        # Convert actual beads first: 0->1, 1->2, 2->3
        converted_pegs = [bead + 1 for bead in peg]
        # Then pad to 3 elements with 0 (no bead)
        row = converted_pegs + [0] * (3 - len(converted_pegs))
        matrix.append(row)
    return matrix


def matrix_to_string(matrix: list) -> str:
    """Format a 3x3 matrix as a string like '[1,0,0],[0,2,0],[3,2,1]'."""
    return ','.join('[' + ','.join(str(x) for x in row) + ']' for row in matrix)


def main():
    raw_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(raw_dir, '..', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    input_path = os.path.join(raw_dir, 'problems.csv')
    output_path = os.path.join(processed_dir, 'problems_matrix.csv')

    results = []
    with open(input_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem_id = row['id']
            start_state_id = row['start_state']
            end_state_id = row['end_state']

            start_pegs = state_id_to_pegs(start_state_id)
            end_pegs = state_id_to_pegs(end_state_id)

            start_matrix = pegs_to_matrix(start_pegs)
            end_matrix = pegs_to_matrix(end_pegs)

            results.append({
                'ID': problem_id,
                'start_matrix': matrix_to_string(start_matrix),
                'goal_matrix': matrix_to_string(end_matrix),
            })

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'start_matrix', 'goal_matrix'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Written {len(results)} problems to {output_path}")


if __name__ == '__main__':
    main()
