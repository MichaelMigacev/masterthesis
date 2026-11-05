"""Aggregate control subjects from PD, MCI, and STRK CSV files into a single deduplicated file.

This file contains code written by AI.
"""

import os
import pandas as pd

# Paths to the source files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "..", "..", "..", "..",
                       "tower_of_london_human_data", "indiv_freiburg_data", "raw")

pd_path = os.path.join(RAW_DIR, "ANONYM_TOLCLIN_PD.csv")
mci_path = os.path.join(RAW_DIR, "ANONYM_TOLCLIN_MCI.csv")
strk_path = os.path.join(RAW_DIR, "ANONYM_TOLCLIN_STRK.csv")

# Output file
output_path = os.path.join(RAW_DIR, "ANONYM_TOLCLIN_CONTROL.csv")

# Read all three CSV files (semicolon-separated)
df_pd = pd.read_csv(pd_path, sep=";")
df_mci = pd.read_csv(mci_path, sep=";")
df_strk = pd.read_csv(strk_path, sep=";", encoding="latin-1")

# Extract controls from each file (first column starts with 'C')
# The first column name may differ across files, so grab column 0
first_col = df_pd.columns[0]

controls_pd = df_pd[df_pd[first_col].astype(str).str.startswith("C")].copy()
controls_mci_df = df_mci[df_mci[df_mci.columns[0]].astype(str).str.startswith("C")].copy()
controls_strk_df = df_strk[df_strk[df_strk.columns[0]].astype(str).str.startswith("C")].copy()

# Concatenate all control rows
all_controls = pd.concat([controls_pd, controls_mci_df, controls_strk_df], ignore_index=True)

# Remove duplicate control entries based on the first column (PCODE / MCode)
all_controls = all_controls.drop_duplicates(subset=[all_controls.columns[0]], keep="first")

# Sort by the control ID (e.g., C1, C2, ..., C10, ...) for readability
all_controls["sort_key"] = all_controls[all_controls.columns[0]].str.extract(r"(\d+)").astype(int)
all_controls = all_controls.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

# Save to CSV
all_controls.to_csv(output_path, sep=";", index=False)
print(f"Saved {len(all_controls)} unique control entries to {output_path}")
