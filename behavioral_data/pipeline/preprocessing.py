"""
Preprocessing utilities for behavioral changepoint trials.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .constants import (
    LEARNING_RATE_CLIP,
    MAX_INVALID_TRIAL_FRACTION,
    SCREEN_MAX,
    SCREEN_MIN,
    UPDATE_Z_THRESHOLD,
)


@dataclass
class QCSummary:
    dropped_subjects: Tuple[str, ...]
    invalid_fraction: Dict[str, float]


def _compute_trial_level_features(events: pd.DataFrame) -> pd.DataFrame:
    """Compute deltas, updates, learning rates, and helper columns."""
    trials = events.sort_values(["subject_id", "run_id", "trial_index"]).copy()
    trials["trial_number"] = (
        trials.groupby(["subject_id", "run_id"]).cumcount().astype(int)
    )
    trials["is_first_trial"] = trials["trial_number"] == 0
    max_idx = trials.groupby(["subject_id", "run_id"])["trial_number"].transform("max")
    trials["is_last_trial"] = trials["trial_number"] == max_idx

    trials["prediction_next"] = (
        trials.groupby(["subject_id", "run_id"])["prediction"].shift(-1)
    )
    trials["delta"] = trials["outcome"] - trials["prediction"]
    trials["update"] = trials["prediction_next"] - trials["prediction"]

    # Learning rate with safe division
    delta = trials["delta"].to_numpy()
    update = trials["update"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(
            np.isclose(delta, 0.0),
            0.0,
            update / np.where(delta == 0.0, 1.0, delta),
        )
    trials["learning_rate_raw"] = lr
    trials["learning_rate"] = np.clip(
        np.nan_to_num(lr, nan=0.0, posinf=LEARNING_RATE_CLIP[1], neginf=LEARNING_RATE_CLIP[0]),
        LEARNING_RATE_CLIP[0],
        LEARNING_RATE_CLIP[1],
    )

    return trials


def _apply_basic_filters(trials: pd.DataFrame) -> pd.DataFrame:
    """Drop first and last trial per run (no PE/update)."""
    mask = ~(trials["is_first_trial"] | trials["is_last_trial"])
    return trials.loc[mask].copy()


def _flag_invalid(trials: pd.DataFrame) -> pd.DataFrame:
    """Add boolean columns flagging QC failures."""
    trials["prediction_in_bounds"] = trials["prediction"].between(SCREEN_MIN, SCREEN_MAX)
    trials["outcome_in_bounds"] = trials["outcome"].between(SCREEN_MIN, SCREEN_MAX)

    update_std = (
        trials.groupby("subject_id")["update"]
        .transform("std")
        .replace(0.0, np.nan)
    )
    trials["update_z"] = trials["update"] / update_std
    trials["update_z"] = trials["update_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    trials["update_outlier"] = trials["update_z"].abs() > UPDATE_Z_THRESHOLD
    trials["lr_outlier"] = trials["learning_rate"].abs() > LEARNING_RATE_CLIP[1]

    # Reaction times are not always present, but respect them when they are.
    if "response_time" in trials.columns:
        trials["rt_outlier"] = ~trials["response_time"].between(0.2, 5.0)
    else:
        trials["rt_outlier"] = False

    trials["valid_trial"] = (
        trials["prediction_in_bounds"]
        & trials["outcome_in_bounds"]
        & ~trials["update_outlier"]
        & ~trials["lr_outlier"]
        & ~trials["rt_outlier"]
    )
    return trials


def _drop_invalid_subjects(trials: pd.DataFrame) -> QCSummary:
    valid_rates = trials.groupby("subject_id")["valid_trial"].mean()
    invalid_fraction = {subj: float(1.0 - rate) for subj, rate in valid_rates.items()}
    dropped = tuple(
        subj for subj, frac in invalid_fraction.items() if frac > MAX_INVALID_TRIAL_FRACTION
    )
    return QCSummary(dropped_subjects=dropped, invalid_fraction=invalid_fraction)


def prepare_trials(events: pd.DataFrame) -> Tuple[pd.DataFrame, QCSummary]:
    """
    Full preprocessing pipeline: compute derived metrics, filter invalid trials,
    and drop participants with excessive invalid data.
    """
    trials = _compute_trial_level_features(events)
    trials = _apply_basic_filters(trials)
    trials = _flag_invalid(trials)
    qc = _drop_invalid_subjects(trials)

    valid_mask = trials["valid_trial"] & ~trials["subject_id"].isin(qc.dropped_subjects)
    clean = trials.loc[valid_mask].copy()

    # Final housekeeping
    clean["subject_run_trial"] = (
        clean["subject_id"]
        + "_run-"
        + clean["run_id"].astype(str)
        + "_trial-"
        + clean["trial_number"].astype(str)
    )
    clean.drop(
        columns=["prediction_next", "learning_rate_raw", "is_first_trial", "is_last_trial"],
        inplace=True,
        errors="ignore",
    )

    return clean, qc


def inject_belief_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add belief-state columns derived from CPP (stable vs. volatile contexts).
    """
    if not {"cpp", "ru"}.issubset(df.columns):
        raise ValueError("CPP/RU columns missing. Run normative signal computation first.")
    enriched = df.copy()
    enriched["belief_volatile"] = enriched["cpp"].clip(0.0, 1.0)
    enriched["belief_stable"] = (1.0 - enriched["belief_volatile"]).clip(0.0, 1.0)
    return enriched


