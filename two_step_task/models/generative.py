"""Builders for A/B matrices moved into `models` package."""
from typing import List
import numpy as np


def build_A_matrices(num_contexts: int = 4) -> List[np.ndarray]:
    n_stages = 5
    A_spaceship = np.zeros((2, num_contexts, n_stages))
    A_spaceship[0, :, :] = 1.0
    A_spaceship[1, :, 1] = 1.0
    A_spaceship[0, :, 1] = 0.0

    A_planet = np.zeros((3, num_contexts, n_stages))
    A_planet[0, :, [0, 1, 4]] = 1.0
    A_planet[:, :, 2] = np.array([1.0, 0.0, 0.0])[:, None]
    A_planet[:, :, 3] = np.array([1.0, 0.0, 0.0])[:, None]

    A_aliens = np.zeros((2, num_contexts, n_stages))
    A_aliens[0, :, :] = 1.0
    A_aliens[1, :, 3] = 1.0
    A_aliens[0, :, 3] = 0.0

    A_reward = np.zeros((3, num_contexts, n_stages))
    A_reward[0, :, [0, 1, 2, 3]] = 1.0
    A_reward[:, :, 4] = np.array([0.0, 0.5, 0.5])[:, None]

    return [A_spaceship, A_planet, A_aliens, A_reward]


def build_B_matrices(context_volatility: float = 0.01) -> List[np.ndarray]:
    n_contexts = 4
    n_stages = 5
    n_actions_stage = 3

    B_context = np.zeros((n_contexts, n_contexts))
    for i in range(n_contexts):
        B_context[i, i] = 1.0 - context_volatility
        for j in range(n_contexts):
            if j != i:
                B_context[j, i] = context_volatility / (n_contexts - 1)

    B_stage = np.zeros((n_stages, n_stages, n_actions_stage))
    B_stage[1, 0, :] = 1.0
    B_stage[2, 1, 1:] = 1.0
    B_stage[3, 2, :] = 1.0
    B_stage[4, 3, 1:] = 1.0
    B_stage[0, 4, :] = 1.0

    return [B_context, B_stage]
