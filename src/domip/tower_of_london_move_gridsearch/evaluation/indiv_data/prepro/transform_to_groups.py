"""Split an exmov CSV into separate Patient and Control files.

Row order is preserved so that patients and controls can be matched by
their position (i.e. row i in the patient file corresponds to row i in
the control file).
"""

import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser(
        description="Split exmov CSV into Patient and Control files."
    )
    parser.add_argument("input_csv", help="Path to the input exmov CSV file.")
    parser.add_argument(
        "output_dir",
        help="Directory where the two output CSV files will be written.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(args.input_csv))[0]
    patient_path = os.path.join(args.output_dir, f"{basename}_patients.csv")
    control_path = os.path.join(args.output_dir, f"{basename}_controls.csv")

    with open(args.input_csv, newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader)

        with open(patient_path, "w", newline="", encoding="utf-8") as pf, \
             open(control_path, "w", newline="", encoding="utf-8") as cf:
            pw = csv.writer(pf)
            cw = csv.writer(cf)
            pw.writerow(header)
            cw.writerow(header)

            for row in reader:
                group = row[-1].strip()
                if group == "Patient":
                    pw.writerow(row)
                elif group == "Control":
                    cw.writerow(row)
                else:
                    print(f"Warning: unexpected group '{group}', skipping row.")

    print(f"Patients written to: {patient_path}")
    print(f"Controls written to: {control_path}")


if __name__ == "__main__":
    main()
