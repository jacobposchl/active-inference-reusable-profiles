"""Inspect CSV files in a folder and print their format summary.

Run from repo root (Windows cmd) e.g.:

.venv\\Scripts\\python.exe two_step_task\\scripts\\inspect_csv_format.py --path two_step_task\\data\\raw

The script lists CSV files in the folder, prints column names, dtypes,
and shows the first 5 rows for each file. It also checks for the
required two-step columns and reports any missing columns.
"""
import argparse
import os
import pandas as pd

REQUIRED_COLS = [
    "participant_id",
    "trial",
    "stage1_choice",
    "transition",
    "planet",
    "stage2_choice",
    "reward",
]


def inspect_file(path: str, nrows: int = 5):
    print(f"\n--- File: {path} ---")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    print("Shape:", df.shape)
    print("Columns:")
    for c in df.columns:
        print(f"  - {c} : {df[c].dtype}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
    else:
        print("All required columns present.")

    print("\nFirst %d rows:" % nrows)
    with pd.option_context('display.max_rows', nrows, 'display.max_columns', None):
        print(df.head(nrows))


def main(folder: str):
    if not os.path.isdir(folder):
        print(f"Path is not a directory: {folder}")
        return

    files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    if not files:
        print(f"No CSV files found in {folder}")
        return

    for f in files:
        inspect_file(os.path.join(folder, f))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--path', '-p', required=True, help='Path to folder containing CSV files')
    args = p.parse_args()
    main(args.path)
