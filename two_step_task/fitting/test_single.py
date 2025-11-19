"""Quick smoke tests for all three models across a few processed subjects.

This script runs a very small grid search per model to ensure the fitter
and models integrate correctly. It uses processed CSVs under
`two_step_task/data/processed/` and prints a short report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from two_step_task.fitting.parameter_search import parameter_search
from two_step_task.models.m1_static import M1_StaticPrecision
from two_step_task.models.m2_entropy import M2_EntropyCoupled
from two_step_task.models.m3_profiles import M3_ProfileBased


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure zero-based stage choices exist
    if 'stage1_choice' not in df.columns:
        def map_choice(v):
            if pd.isna(v):
                return np.nan
            try:
                iv = int(float(v))
            except Exception:
                return np.nan
            if iv in (1, 2):
                return iv - 1
            if iv in (0, 1):
                return iv
            return np.nan
        df['stage1_choice'] = df['choice_1'].apply(map_choice)
        df['stage2_choice'] = df['choice_2'].apply(map_choice)

    # Ensure planet column exists
    if 'planet' not in df.columns:
        def infer_planet(row):
            tr = row.get('transition')
            sc = row.get('stage1_choice')
            if pd.isna(sc) or pd.isna(tr):
                return np.nan
            trl = str(tr).lower()
            return 'red' if (trl.startswith('c') and int(sc) == 0) or (trl.startswith('r') and int(sc) == 1) else 'purple'
        df['planet'] = df.apply(infer_planet, axis=1)

    # Keep only rows with required fields
    df_model = df.dropna(subset=['stage1_choice', 'stage2_choice', 'planet', 'reward']).copy()
    df_model['reward'] = df_model['reward'].astype(int)
    return df_model


def quick_test_on_subject(subject_file: Path):
    print(f"\nSubject: {subject_file.name}")
    df = load_and_prepare(subject_file)
    print(f" Usable trials: {len(df)}")

    models = [
        ("M1_StaticPrecision", M1_StaticPrecision()),
        ("M2_EntropyCoupled", M2_EntropyCoupled()),
        ("M3_ProfileBased", M3_ProfileBased()),
    ]

    for name, model in models:
        try:
            best_params, best_ll = parameter_search(model, df, n_points=3, top_k=2)
        except Exception as e:
            print(f"  {name}: ERROR during fit: {e}")
            continue
        print(f"  {name}: best_ll={best_ll:.3f}, params={None if best_params is None else np.round(best_params,3).tolist()}")


def main():
    proc_dir = Path(__file__).parent.parent / 'data' / 'processed'
    # pick a small sample of subjects (first 3 files)
    files = sorted(proc_dir.glob('participant_*.csv'))[:3]
    if not files:
        print('No processed participant files found in', proc_dir)
        sys.exit(1)
    for f in files:
        quick_test_on_subject(f)


if __name__ == '__main__':
    main()
