"""
Compute normative CPP/RU/entropy signals using a discrete-state Bayesian filter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .constants import (
    DATA_ROOT,
    HAZARD_RATE,
    SIGNALS_DIR,
    STATE_GRID,
    UNIFORM_PRIOR,
)


def _gaussian_likelihood(x: float, states: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate N(x | states, sigma^2) for each candidate state."""
    var = max(sigma, 1e-6) ** 2
    norm = np.sqrt(2.0 * np.pi * var)
    diff = x - states
    return np.exp(-0.5 * (diff ** 2) / var) / norm


def _run_filter_for_run(
    run_df: pd.DataFrame,
    hazard_rate: float,
    state_grid: np.ndarray,
    prior: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Return dictionaries keyed by trial index with normative signals."""
    belief = prior.copy()
    uniform = UNIFORM_PRIOR
    results: Dict[int, Dict[str, float]] = {}

    for idx, row in run_df.iterrows():
        sigma = float(row["noise_sd"])
        trial_key = int(row["trial_index"])
        stay_component = (1.0 - hazard_rate) * belief
        change_component = hazard_rate * uniform
        prior_belief = stay_component + change_component

        likelihood = _gaussian_likelihood(float(row["outcome"]), state_grid, sigma)
        joint = likelihood * prior_belief
        evidence = joint.sum()

        if evidence <= 0 or not np.isfinite(evidence):
            posterior = uniform.copy()
            evidence = 1.0
        else:
            posterior = joint / evidence

        cpp_numerator = (likelihood * change_component).sum()
        cpp = float(np.clip(cpp_numerator / evidence, 0.0, 1.0))

        mean = float((posterior * state_grid).sum())
        variance = float((posterior * (state_grid - mean) ** 2).sum())
        entropy = float(-(posterior * np.log(posterior + 1e-12)).sum())
        ru = float(variance / (variance + max(sigma, 1e-6) ** 2))

        results[trial_key] = {
            "cpp": cpp,
            "ru": ru,
            "belief_mean": mean,
            "belief_var": variance,
            "belief_entropy": entropy,
        }

        belief = posterior

    return results


def _compute_subject_signals(
    subject_df: pd.DataFrame,
    hazard_rate: float,
    state_grid: np.ndarray,
) -> pd.DataFrame:
    subject_df = subject_df.sort_values(["subject_id", "run_id", "trial_index"])
    records = []
    for run_id, run_df in subject_df.groupby("run_id"):
        signals = _run_filter_for_run(
            run_df, hazard_rate, state_grid, UNIFORM_PRIOR.copy()
        )
        for trial_index, values in signals.items():
            row = {
                "subject_id": run_df["subject_id"].iloc[0],
                "run_id": run_id,
                "trial_index": trial_index,
            }
            row.update(values)
            records.append(row)
    return pd.DataFrame(records)


def compute_normative_signals(
    trials: pd.DataFrame,
    hazard_rate: float = HAZARD_RATE,
    state_grid: np.ndarray = STATE_GRID,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Attach CPP, RU, and entropy signals to the provided trials DataFrame.
    """
    augmented_frames = []
    for subject_id, subject_df in trials.groupby("subject_id"):
        cache_path = SIGNALS_DIR / f"{subject_id}_normative.parquet"
        if cache and cache_path.exists():
            signals = pd.read_parquet(cache_path)
        else:
            signals = _compute_subject_signals(subject_df, hazard_rate, state_grid)
            if cache:
                signals.to_parquet(cache_path, index=False)
        augmented_frames.append(signals)

    merged_signals = pd.concat(augmented_frames, ignore_index=True)
    enriched = trials.merge(
        merged_signals,
        on=["subject_id", "run_id", "trial_index"],
        how="left",
        validate="one_to_one",
    )
    return enriched


