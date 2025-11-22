"""
Adapters that reuse the existing value-function implementations (M1/M2/M3)
to produce learning-rate predictions for the changepoint data.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

# Ensure repo root is importable so we can reuse src.models implementations.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.models.value_functions import make_value_fn  # type: ignore  # pylint: disable=import-error


def _belief_matrix(df):
    return df[["belief_stable", "belief_volatile"]].to_numpy(dtype=float)


@dataclass
class ValueFunctionAdapter:
    value_fn: callable
    belief_matrix: np.ndarray

    def predict_alpha(self) -> np.ndarray:
        alphas = np.empty(len(self.belief_matrix), dtype=float)
        for idx, belief in enumerate(self.belief_matrix):
            _, _, gamma = self.value_fn(belief, idx)
            alphas[idx] = float(gamma)
        return alphas


def build_m1_adapter(df, alpha: float):
    value_fn = make_value_fn(
        "M1",
        C_reward_logits=[0.0, 0.0, 0.0],
        gamma=float(alpha),
    )
    return ValueFunctionAdapter(value_fn=value_fn, belief_matrix=_belief_matrix(df))


def build_m2_adapter(df, alpha_base: float, driver: str = "combined"):
    cpp = df["cpp"].to_numpy(dtype=float)
    ru = df["ru"].to_numpy(dtype=float)

    def gamma_schedule(q_context, t, driver=driver, alpha_base=alpha_base):
        if driver == "cpp":
            raw = cpp[t]
        elif driver == "ru":
            raw = ru[t]
        elif driver == "combined":
            raw = cpp[t] + ru[t] * (1.0 - cpp[t])
        else:
            raise ValueError(f"Unknown driver '{driver}'")
        raw = np.clip(raw, 0.0, 1.0)
        return float(alpha_base * raw)

    value_fn = make_value_fn(
        "M2",
        C_reward_logits=[0.0, 0.0, 0.0],
        gamma_schedule=gamma_schedule,
    )
    return ValueFunctionAdapter(value_fn=value_fn, belief_matrix=_belief_matrix(df))


def build_m3_adapter(
    df,
    alpha_stable: float,
    alpha_volatile: float,
    soft_assign: bool = False,
):
    profiles = [
        {"phi_logits": [0.0, 0.0, 0.0], "gamma": float(alpha_stable)},
        {"phi_logits": [0.0, 0.0, 0.0], "gamma": float(alpha_volatile)},
    ]
    if soft_assign:
        Z = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float)
    else:
        Z = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    value_fn = make_value_fn("M3", profiles=profiles, Z=Z)
    return ValueFunctionAdapter(value_fn=value_fn, belief_matrix=_belief_matrix(df))


MODEL_BUILDERS = {
    "M1": build_m1_adapter,
    "M2": build_m2_adapter,
    "M3": build_m3_adapter,
}

PARAMETER_GRIDS: Dict[str, Dict[str, Sequence]] = {
    "M1": {
        "alpha": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    },
    "M2": {
        "alpha_base": [0.5, 1.0, 1.5, 2.0],
        "driver": ["cpp", "ru", "combined"],
    },
    "M3": {
        "alpha_stable": [0.05, 0.1, 0.2, 0.3],
        "alpha_volatile": [0.4, 0.6, 0.8, 1.0],
        "soft_assign": [False, True],
    },
}


