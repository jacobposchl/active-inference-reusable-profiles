"""
Data loading helpers for the changepoint behavioral dataset.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .constants import (
    RAW_DATA_ROOT,
    EVENTS_GLOB,
    NOISE_LABELS,
    PARTICIPANTS_FILE,
    SCREEN_MAX,
    SCREEN_MIN,
)

RUN_PATTERN = re.compile(r"run-(\d+)")


def list_subject_dirs(root: Path = RAW_DATA_ROOT) -> List[Path]:
    """Return sorted list of subject directories under the dataset root."""
    return sorted(p for p in root.glob("sub-*") if p.is_dir())


def load_participants(participants_file: Path = PARTICIPANTS_FILE) -> pd.DataFrame:
    """Load participant demographics."""
    if not participants_file.exists():
        raise FileNotFoundError(f"participants file missing: {participants_file}")
    participants = pd.read_csv(participants_file, sep="\t")
    participants.rename(columns={"participant_id": "subject_id"}, inplace=True)
    return participants


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename verbose BIDS columns into snake_case equivalents."""
    rename_map = {
        "outcome noise": "noise_sd",
        "outcome value": "reward_value",
        "isChangeTrial": "is_changepoint",
        "current state": "state_mean",
        "current outcome": "outcome",
        "current prediction": "prediction",
    }
    cleaned = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})
    return cleaned


def _extract_run_id(path: Path) -> int:
    match = RUN_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Unable to parse run id from {path.name}")
    return int(match.group(1))


def load_events(root: Path = RAW_DATA_ROOT) -> pd.DataFrame:
    """
    Load every task-changepoint events file into a single tidy DataFrame.
    """
    all_records: List[pd.DataFrame] = []
    for sub_dir in list_subject_dirs(root):
        subject_id = sub_dir.name
        func_dir = sub_dir / "func"
        if not func_dir.exists():
            continue
        for events_path in sorted(func_dir.glob("*task-changepoint_run-*_events.tsv")):
            run_id = _extract_run_id(events_path)
            df = pd.read_csv(events_path, sep="\t")
            df = _normalize_columns(df)
            df["subject_id"] = subject_id
            df["run_id"] = run_id
            df["events_path"] = str(events_path.relative_to(root))
            df["trial_index"] = range(len(df))
            all_records.append(df)

    if not all_records:
        raise RuntimeError(f"No events files found under {root / EVENTS_GLOB}")

    events = pd.concat(all_records, ignore_index=True)

    # Basic typing
    numeric_cols = ["noise_sd", "reward_value", "state_mean", "outcome", "prediction"]
    for col in numeric_cols:
        events[col] = pd.to_numeric(events[col], errors="coerce")
    events["is_changepoint"] = events["is_changepoint"].fillna(0).astype(int)
    events["noise_condition"] = events["noise_sd"].map(NOISE_LABELS).fillna("unknown")
    events["noise_sd"] = events["noise_sd"].astype(float)

    # Keep track of display bounds for quick QC
    events["prediction_in_bounds"] = events["prediction"].between(SCREEN_MIN, SCREEN_MAX)
    events["outcome_in_bounds"] = events["outcome"].between(SCREEN_MIN, SCREEN_MAX)

    return events


def load_dataset(root: Path = RAW_DATA_ROOT) -> pd.DataFrame:
    """Load events merged with participant metadata."""
    events = load_events(root)
    try:
        participants = load_participants()
    except FileNotFoundError:
        return events
    merged = events.merge(participants, on="subject_id", how="left")
    return merged


