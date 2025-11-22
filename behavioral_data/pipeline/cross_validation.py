"""
Cross-validation utilities for evaluating models on the behavioral dataset.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .model_wrappers import MODEL_BUILDERS, PARAMETER_GRIDS


@dataclass
class FitResult:
    model: str
    subject_id: str
    params: Dict[str, float]
    train_ll: float
    test_ll: float
    num_trials: int


def _expand_grid(grid: Dict[str, Sequence]) -> List[Dict[str, float]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for product in itertools.product(*values):
        combos.append(dict(zip(keys, product)))
    return combos


def _compute_sigma(update_series: pd.Series) -> float:
    sigma = float(update_series.std(ddof=1))
    if not np.isfinite(sigma) or sigma < 1e-3:
        sigma = 1.0
    return sigma


def _gaussian_log_likelihood(delta: np.ndarray, update: np.ndarray, alpha: np.ndarray, sigma: float) -> float:
    pred = alpha * delta
    resid = update - pred
    const = -0.5 * np.log(2.0 * np.pi * sigma ** 2)
    ll = const * len(resid) - 0.5 * np.sum((resid ** 2) / (sigma ** 2))
    return float(ll)


def _score(df: pd.DataFrame, builder, params: Dict[str, float], sigma: float) -> float:
    adapter = builder(df, **params)
    alpha = adapter.predict_alpha()
    return _gaussian_log_likelihood(
        df["delta"].to_numpy(dtype=float),
        df["update"].to_numpy(dtype=float),
        alpha,
        sigma,
    )


def _grid_search(train_df: pd.DataFrame, model: str) -> Tuple[Dict[str, float], float]:
    builder = MODEL_BUILDERS[model]
    sigma = _compute_sigma(train_df["update"])
    best_params = None
    best_ll = -math.inf
    for params in _expand_grid(PARAMETER_GRIDS[model]):
        ll = _score(train_df, builder, params, sigma)
        if ll > best_ll:
            best_ll = ll
            best_params = params
    return best_params or {}, best_ll


def _evaluate(df: pd.DataFrame, model: str, params: Dict[str, float]) -> float:
    sigma = _compute_sigma(df["update"])
    builder = MODEL_BUILDERS[model]
    return _score(df, builder, params, sigma)


def loso_cv(trials: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-subject-out CV."""
    records = []
    subjects = sorted(trials["subject_id"].unique())
    for subject in subjects:
        train_df = trials[trials["subject_id"] != subject]
        test_df = trials[trials["subject_id"] == subject]
        for model in MODEL_BUILDERS:
            best_params, train_ll = _grid_search(train_df, model)
            test_ll = _evaluate(test_df, model, best_params)
            records.append(
                FitResult(
                    model=model,
                    subject_id=subject,
                    params=best_params,
                    train_ll=train_ll,
                    test_ll=test_ll,
                    num_trials=len(test_df),
                ).__dict__
            )
    return pd.DataFrame(records)


def temporal_split_cv(trials: pd.DataFrame) -> pd.DataFrame:
    """Within-subject split: first two runs vs last two runs."""
    records = []
    for subject, subject_df in trials.groupby("subject_id"):
        run_ids = sorted(subject_df["run_id"].unique())
        midpoint = len(run_ids) // 2
        train_runs = run_ids[:midpoint]
        test_runs = run_ids[midpoint:]
        train_df = subject_df[subject_df["run_id"].isin(train_runs)]
        test_df = subject_df[subject_df["run_id"].isin(test_runs)]
        for model in MODEL_BUILDERS:
            best_params, train_ll = _grid_search(train_df, model)
            test_ll = _evaluate(test_df, model, best_params)
            records.append(
                FitResult(
                    model=model,
                    subject_id=subject,
                    params=best_params,
                    train_ll=train_ll,
                    test_ll=test_ll,
                    num_trials=len(test_df),
                ).__dict__
            )
    return pd.DataFrame(records)


