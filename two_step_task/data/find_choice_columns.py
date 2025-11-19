"""Heuristic inspector to find which columns record choices or planet names.

Usage (repo root):
.venv\Scripts\python.exe two_step_task\data\find_choice_columns.py --file two_step_task\data\raw\mbmf_sub98.csv
"""
import argparse
from pathlib import Path
import pandas as pd

CANDIDATE_CHOICE_COLS = [
    'choice', 'chosen_text', 'chosen', 'button_pressed', 'key_press',
    'right_text', 'left_text', 'planet_text', 'center_text'
]


def inspect_file(path: Path, trial_col='trial_index'):
    df = pd.read_csv(path)
    if trial_col not in df.columns:
        trial_col = 'trial' if 'trial' in df.columns else None
    if trial_col is None:
        raise ValueError('No trial column found')

    print('File:', path)
    print('Rows:', len(df))
    print('\nCandidate columns non-null counts (overall):')
    for c in CANDIDATE_CHOICE_COLS:
        if c in df.columns:
            cnt = df[c].notna().sum()
            sample = df[c].dropna().unique()[:5]
            print(f'  - {c}: non-null={cnt}, sample_values={sample}')
        else:
            print(f'  - {c}: NOT PRESENT')

    print('\nPer-trial non-null counts (first 20 trials):')
    grouped = df.groupby(trial_col)
    trials = list(grouped.groups.keys())[:20]
    for t in trials:
        rows = grouped.get_group(t)
        out = {c: int(rows[c].notna().sum()) if c in rows.columns else 0 for c in CANDIDATE_CHOICE_COLS}
        print(f' trial {t}:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--file', '-f', required=True)
    args = p.parse_args()
    inspect_file(Path(args.file))
