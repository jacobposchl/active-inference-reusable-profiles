"""
Baseline reinforcement learning models for the changepoint experiment.

Implements:
    - Rescorla-Wagner with fixed learning rate
    - Rescorla-Wagner with dynamic learning rate (alpha depends on |delta|)
    - Q-learning with epsilon-greedy policy over discretized predictions
    - Q-learning with softmax policy over discretized predictions

All models are evaluated with leave-one-subject-out cross-validation and
results are saved to `behavioral_data/derivatives/analysis/rl_loso_results.csv`.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import data_io, preprocessing

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "derivatives" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def compute_sigma(series: pd.Series) -> float:
    sigma = float(series.std(ddof=1))
    if not np.isfinite(sigma) or sigma < 1e-3:
        sigma = 1.0
    return sigma


def gaussian_ll(delta: np.ndarray, update: np.ndarray, alpha: np.ndarray, sigma: float) -> float:
    pred_update = alpha * delta
    resid = update - pred_update
    const = -0.5 * np.log(2.0 * np.pi * sigma ** 2)
    ll = const * len(resid) - 0.5 * np.sum((resid ** 2) / (sigma ** 2))
    return float(ll)


class RescorlaWagnerFixed:
    name = "RW_fixed"
    param_grid = {"alpha": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]}

    @staticmethod
    def alpha_sequence(df: pd.DataFrame, params: Dict[str, float]) -> np.ndarray:
        return np.full(len(df), float(params["alpha"]), dtype=float)

    @classmethod
    def log_likelihood(cls, df: pd.DataFrame, params: Dict[str, float]) -> float:
        sigma = compute_sigma(df["update"])
        alpha = cls.alpha_sequence(df, params)
        return gaussian_ll(df["delta"].to_numpy(), df["update"].to_numpy(), alpha, sigma)


class RescorlaWagnerDynamic:
    name = "RW_dynamic"
    param_grid = {
        "alpha_low": [0.01, 0.05, 0.1],
        "alpha_high": [0.5, 0.7, 0.9, 1.1],
        "beta": [0.5, 1.0, 1.5, 2.0],
    }

    @staticmethod
    def alpha_sequence(df: pd.DataFrame, params: Dict[str, float]) -> np.ndarray:
        delta = np.abs(df["delta"].to_numpy())
        alpha_low = float(params["alpha_low"])
        alpha_high = float(params["alpha_high"])
        beta = float(params["beta"])
        alpha_range = max(alpha_high - alpha_low, 1e-6)
        scaled = 1.0 - np.exp(-beta * delta)
        alphas = np.clip(alpha_low + alpha_range * scaled, 0.0, 2.0)
        return alphas

    @classmethod
    def log_likelihood(cls, df: pd.DataFrame, params: Dict[str, float]) -> float:
        sigma = compute_sigma(df["update"])
        alpha = cls.alpha_sequence(df, params)
        return gaussian_ll(df["delta"].to_numpy(), df["update"].to_numpy(), alpha, sigma)


class QLearningBase:
    def __init__(self, name: str, policy: str):
        self.name = name
        self.policy = policy
        self.num_bins = 31
        self.bin_edges = np.linspace(0.0, 300.0, self.num_bins + 1)
        centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_centers = centers

    def prediction_to_action(self, prediction: float) -> int:
        prediction = np.clip(prediction, 0.0, 300.0)
        idx = np.searchsorted(self.bin_edges, prediction, side="right") - 1
        return int(np.clip(idx, 0, self.num_bins - 1))

    def reward(self, action: int, outcome: float) -> float:
        # Negative absolute error so higher reward => better predictions
        return -abs(outcome - self.bin_centers[action]) / 300.0

    def _epsilon_greedy_probs(self, q: np.ndarray, epsilon: float) -> np.ndarray:
        best_value = np.max(q)
        greedy_actions = np.flatnonzero(np.isclose(q, best_value))
        probs = np.full_like(q, epsilon / self.num_bins, dtype=float)
        probs[greedy_actions] += (1.0 - epsilon) / max(len(greedy_actions), 1)
        return probs

    def _softmax_probs(self, q: np.ndarray, beta: float) -> np.ndarray:
        logits = beta * q
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()
        return probs

    def log_likelihood(self, df: pd.DataFrame, params: Dict[str, float]) -> float:
        ll = 0.0
        for subject_id, subj_df in df.groupby("subject_id"):
            q_values = np.zeros(self.num_bins, dtype=float)
            for row in subj_df.sort_values(["run_id", "trial_index"]).itertuples():
                action_idx = self.prediction_to_action(row.prediction)
                if self.policy == "epsilon":
                    probs = self._epsilon_greedy_probs(q_values, float(params["epsilon"]))
                else:
                    probs = self._softmax_probs(q_values, float(params["beta"]))
                ll += np.log(probs[action_idx] + 1e-12)
                reward = self.reward(action_idx, float(row.outcome))
                alpha = float(params["alpha"])
                q_values[action_idx] += alpha * (reward - q_values[action_idx])
        return float(ll)


class QLearningEpsilonGreedy(QLearningBase):
    param_grid = {
        "alpha": [0.05, 0.1, 0.2, 0.3, 0.5],
        "epsilon": [0.01, 0.05, 0.1, 0.2, 0.3],
    }

    def __init__(self):
        super().__init__(name="QL_eps_greedy", policy="epsilon")


class QLearningSoftmax(QLearningBase):
    param_grid = {
        "alpha": [0.05, 0.1, 0.2, 0.3, 0.5],
        "beta": [1.0, 2.0, 4.0, 6.0, 8.0],
    }

    def __init__(self):
        super().__init__(name="QL_softmax", policy="softmax")


def expand_grid(grid: Dict[str, Sequence]) -> List[Dict[str, float]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for product_values in product(*values):
        combos.append(dict(zip(keys, product_values)))
    return combos


def fit_and_score(model, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    best_params = None
    best_ll = -np.inf
    for params in expand_grid(model.param_grid):
        ll = model.log_likelihood(train_df, params)
        if ll > best_ll:
            best_ll = ll
            best_params = params
    test_ll = model.log_likelihood(test_df, best_params)
    return {"best_params": best_params, "train_ll": best_ll, "test_ll": test_ll}


def run_rl_baselines(trials: Optional[pd.DataFrame] = None, save: bool = True) -> pd.DataFrame:
    if trials is None:
        events = data_io.load_dataset()
        trials, _ = preprocessing.prepare_trials(events)

    models = [
        RescorlaWagnerFixed(),
        RescorlaWagnerDynamic(),
        QLearningEpsilonGreedy(),
        QLearningSoftmax(),
    ]

    subjects = sorted(trials["subject_id"].unique())
    records = []
    for test_subject in subjects:
        train_df = trials[trials["subject_id"] != test_subject]
        test_df = trials[trials["subject_id"] == test_subject]
        for model in models:
            result = fit_and_score(model, train_df, test_df)
            records.append(
                {
                    "model": model.name,
                    "subject_id": test_subject,
                    "params": result["best_params"],
                    "train_ll": result["train_ll"],
                    "test_ll": result["test_ll"],
                    "num_trials": len(test_df),
                }
            )

    results = pd.DataFrame(records)
    if save:
        results.to_csv(ANALYSIS_DIR / "rl_loso_results.csv", index=False)
    return results


def summarize(results: pd.DataFrame) -> None:
    stats = (
        results.groupby("model")["test_ll"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mean_test_ll", "std": "std_test_ll"})
    )
    winners = (
        results.loc[results.groupby("subject_id")["test_ll"].idxmax(), "model"]
        .value_counts()
        .sort_index()
    )

    print("=== RL Baseline LOSO summary ===")
    print(stats.to_string(float_format="{:.2f}".format))
    print("\nWinner counts:")
    print(winners.to_string())


def main():
    results = run_rl_baselines(save=True)
    summarize(results)


if __name__ == "__main__":
    main()


