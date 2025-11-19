"""Convert Pavlovia-export CSVs (per-subject event logs) into a pivoted
trial-level table compatible with the provided R/MATLAB workflow.

This script produces per-subject CSVs and a single combined text file
matching the structure produced by `format_online_data_for_matlab.R`.

Canonical output columns (per trial):
- `subject_id`, `trial`, `choice_1`, `choice_2`, `rt_1`, `rt_2`,
  `transition`, `reward`, `state`

Notes and heuristics:
- We assign event-order-based trial grouping: two sequential rows -> one trial.
- If `trial_stage` exists we use it to assign stage 1 vs stage 2; otherwise
  the first row within each paired event is stage 1 and the second stage 2.
- We keep missing numeric choices as NaN (do NOT coerce to 0).
- `state` is computed from `choice_1` and `transition` using the same rules
  as the R script so that downstream MATLAB code is compatible.
"""

from pathlib import Path
import pandas as pd
import numpy as np


RAW_DIR = Path(__file__).parent / "raw"
OUT_DIR = Path(__file__).parent / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_stage(col_val):
    """Return stage number 1 or 2 given a trial_stage value (string or int)."""
    if pd.isna(col_val):
        return None
    try:
        # numeric-like
        iv = int(col_val)
        if iv in (1, 2):
            return iv
    except Exception:
        pass
    s = str(col_val).lower()
    if '1' in s and '2' not in s:
        return 1
    if '2' in s and '1' not in s:
        return 2
    if 'first' in s or 'stage1' in s or 'stage_1' in s or 'stage-1' in s:
        return 1
    if 'second' in s or 'stage2' in s or 'stage_2' in s or 'stage-2' in s:
        return 2
    return None


def process_file(path: Path):
    df = pd.read_csv(path)
    # identify participant id
    pid_col = 'subject_id' if 'subject_id' in df.columns else ('participant_id' if 'participant_id' in df.columns else None)
    if pid_col is None:
        raise ValueError(f'No participant id column in {path}')
    pid = df[pid_col].dropna().unique()
    pid = pid[0] if len(pid) > 0 else path.stem

    # Optionally filter to real/practice trials if column exists
    if 'practice_trial' in df.columns:
        try:
            df = df[df['practice_trial'].astype(str).str.lower() == 'real']
        except Exception:
            pass

    # We create an event-order index (preserve file order) and group rows into
    # trials by pairing sequential rows (two events per trial). This mirrors
    # the R script which uses ceiling(rank(event_index)/2).
    df = df.reset_index(drop=True)
    df['_event_order'] = np.arange(len(df)) + 1
    df['_trial_pair'] = np.ceil(df['_event_order'] / 2).astype(int)

    # Determine a stage number per row: prefer an explicit trial_stage column
    if 'trial_stage' in df.columns:
        df['_stage_num'] = df['trial_stage'].apply(_normalize_stage)
    else:
        # assign by order within paired trial (1 then 2)
        df['_stage_num'] = df.groupby('_trial_pair').cumcount() + 1

    # Build stage-specific frames
    extract_cols = ['choice', 'rt', 'transition', 'reward', 'planet_text', 'left_text', 'right_text', 'chosen_text']
    # ensure cols exist
    for c in extract_cols:
        if c not in df.columns:
            df[c] = np.nan

    stage1 = df[df['_stage_num'] == 1].set_index('_trial_pair')
    stage2 = df[df['_stage_num'] == 2].set_index('_trial_pair')

    def safe_col(frame, col, suffix):
        if col in frame.columns:
            return frame[col].rename(f"{col}_{suffix}")
        return pd.Series(index=frame.index, dtype=object, name=f"{col}_{suffix}")

    # Prepare a DataFrame indexed by trial number
    trials = pd.Index(sorted(df['_trial_pair'].unique()), name='trial')
    out = pd.DataFrame(index=trials)

    # grab first non-null value for each desired column on each stage
    for col in ['choice', 'rt', 'transition', 'reward', 'planet_text', 'left_text', 'right_text', 'chosen_text']:
        if not stage1.empty:
            out[f"{col}_1"] = stage1.groupby(stage1.index)[col].first()
        else:
            out[f"{col}_1"] = np.nan
        if not stage2.empty:
            out[f"{col}_2"] = stage2.groupby(stage2.index)[col].first()
        else:
            out[f"{col}_2"] = np.nan

    # Reset trial index to a 1..N trial numbering
    out = out.reset_index().rename(columns={'_trial_pair': 'trial'})

    # Map columns to canonical names used in R script
    # choice_* may be numeric (1/2) or text; keep as-is but convert numeric 1/2 -> 1/2 (no zero-basing here)
    # We'll produce `choice_1`, `choice_2`, `rt_1`, `rt_2`, `transition` (from stage2), `reward` (from stage2)
    def to_float_or_nan(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    out['choice_1'] = out['choice_1'].apply(lambda x: to_float_or_nan(x) if pd.notna(x) else np.nan)
    out['choice_2'] = out['choice_2'].apply(lambda x: to_float_or_nan(x) if pd.notna(x) else np.nan)
    out['rt_1'] = out['rt_1'].apply(lambda x: to_float_or_nan(x) if pd.notna(x) else np.nan)
    out['rt_2'] = out['rt_2'].apply(lambda x: to_float_or_nan(x) if pd.notna(x) else np.nan)

    # transition and reward: prefer stage-2 values (mirrors R script)
    out['transition'] = out['transition_2'].replace({np.nan: None})
    out['reward'] = out['reward_2'].apply(lambda x: int(x) if pd.notna(x) else np.nan)

    # compute state using the same mapping as the R script
    # R mapping expects choice_1 coded as 1 or 2. If we detect 0/1, shift by +1.
    def compute_state(row):
        c1 = row['choice_1']
        tr = row['transition']
        if pd.isna(c1) or pd.isna(tr):
            return np.nan
        try:
            c1v = int(c1)
        except Exception:
            return np.nan
        if c1v in (0, 1):
            c1v = c1v + 1
        trl = str(tr).lower()
        if c1v == 1 and trl.startswith('c'):
            return 2
        if c1v == 1 and trl.startswith('r'):
            return 3
        if c1v == 2 and trl.startswith('c'):
            return 3
        if c1v == 2 and trl.startswith('r'):
            return 2
        return np.nan

    out['state'] = out.apply(compute_state, axis=1)

    # Put subject id column
    out.insert(0, 'subject_id', pid)

    # Also produce zero-based stage choice columns expected by the fitter
    def map_choice_to_zero_based(v):
        if pd.isna(v):
            return np.nan
        try:
            iv = int(float(v))
        except Exception:
            return np.nan
        # upstream may be 1/2 -> convert to 0/1; if already 0/1 keep
        if iv in (1, 2):
            return iv - 1
        if iv in (0, 1):
            return iv
        return np.nan

    out['stage1_choice'] = out['choice_1'].apply(map_choice_to_zero_based)
    out['stage2_choice'] = out['choice_2'].apply(map_choice_to_zero_based)

    # Infer planet (string) from transition + stage1_choice if not explicit
    def infer_planet_from_row(row):
        tr = row.get('transition')
        sc = row.get('stage1_choice')
        if pd.isna(sc) or pd.isna(tr):
            return np.nan
        trl = str(tr).lower()
        try:
            scv = int(sc)
        except Exception:
            return np.nan
        if trl.startswith('c'):
            return 'red' if scv == 0 else 'purple'
        else:
            return 'purple' if scv == 0 else 'red'

    out['planet'] = out.apply(infer_planet_from_row, axis=1)

    # Keep only the canonical columns (similar to R output order)
    final_cols = ['subject_id', 'trial', 'choice_1', 'choice_2', 'stage1_choice', 'stage2_choice', 'rt_1', 'rt_2', 'transition', 'reward', 'state', 'planet']
    final_df = out[final_cols]

    # write per-subject CSV and return
    out_path = OUT_DIR / f'participant_{pid}.csv'
    final_df.to_csv(out_path, index=False)
    return final_df, 0


def convert_all(raw_dir: Path = RAW_DIR):
    files = sorted(raw_dir.glob('*.csv'))
    if not files:
        print('No CSV files found in', raw_dir)
        return None

    combined = []
    for f in files:
        try:
            out_df, _ = process_file(f)
            print(f'Processed {f.name}: {len(out_df)} trials')
            combined.append(out_df)
        except Exception as e:
            print(f'Failed to process {f.name}: {e}')

    if combined:
        all_df = pd.concat(combined, ignore_index=True)
        # write combined file similar to R's output
        combined_path = OUT_DIR / 'online_data_for_matlab.txt'
        all_df.to_csv(combined_path, index=False)
        return all_df
    return None


if __name__ == '__main__':
    res = convert_all()
    print('Done')
