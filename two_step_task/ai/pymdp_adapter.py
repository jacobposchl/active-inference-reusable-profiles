"""Compatibility shim: thin wrapper exposing a small API backed by
`AgentRunner` and the upstream `pymdp.Agent`.

This module provides `PymdpAgent` with the same public methods the
fitting code expects, but internally uses `AgentRunner` with
`learning_mode='dirichlet'` (no alpha/beta bookkeeping).
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from pymdp.agent import Agent
from pymdp import utils

from ..models.generative import build_A_matrices, build_B_matrices
from .agent_runner import AgentRunnerWithLL
from ..models.value_functions import make_value_fn_for_model


class PymdpAgent:
    """Thin adapter that constructs a `pymdp.Agent` and an `AgentRunner`.

    Implements: `action_probs_stage1`, `action_probs_stage2`,
    `set_current_planet`, and `update_after_feedback`.
    """

    def __init__(self, model, policy_prior: Optional[np.ndarray] = None):
        self.model = model

        A_list = build_A_matrices(num_contexts=4)
        B_list = build_B_matrices(context_volatility=0.01)

        # initial priors
        n_contexts = A_list[1].shape[1]
        D_context = np.ones(n_contexts) / float(n_contexts)
        n_stages = A_list[0].shape[2]
        D_stage = np.zeros(n_stages)
        D_stage[0] = 1.0
        D = [D_context, D_stage]

        C0 = utils.obj_array_zeros([(A_list[m].shape[0],) for m in range(len(A_list))])

        agent = Agent(
            A=A_list,
            B=B_list,
            C=C0,
            D=D,
            policy_len=2,
            inference_horizon=1,
            control_fac_idx=[1],
            use_utility=True,
            use_states_info_gain=True,
            action_selection="stochastic",
            gamma=1.0,
        )

        value_fn = make_value_fn_for_model(model)
        self._runner = AgentRunnerWithLL(agent, reward_mod_idx=len(A_list) - 1, value_fn=value_fn, learning_mode="dirichlet")
        self._n_modalities = len(A_list)
        self.current_planet: Optional[str] = None

    def reset(self):
        # Reset runner internal state by recreating its concentrations from agent
        # (AgentRunner currently handles its own initialization; simple approach
        # is to recreate the wrapper entirely if needed.)
        pass

    def action_probs_stage1(self) -> np.ndarray:
        obs_ids = [0] * self._n_modalities
        q_pi, _ = self._runner.infer_policy_posterior(obs_ids, t=0)
        p_left = 0.0
        p_right = 0.0
        for pol_idx, policy in enumerate(self._runner.agent.policies):
            first_action = int(policy[0, 1])
            if first_action == 0:
                p_left += float(q_pi[pol_idx])
            else:
                p_right += float(q_pi[pol_idx])
        s = p_left + p_right
        if s <= 0:
            return np.array([0.5, 0.5])
        return np.array([p_left / s, p_right / s])

    def action_probs_stage2(self, planet: str) -> np.ndarray:
        planet_idx = 1 if str(planet).lower().startswith("r") else 2
        obs_ids = [0] * self._n_modalities
        obs_ids[1] = int(planet_idx)
        q_pi, _ = self._runner.infer_policy_posterior(obs_ids, t=0)
        p_alien0 = 0.0
        p_alien1 = 0.0
        for pol_idx, policy in enumerate(self._runner.agent.policies):
            second_action = int(policy[1, 1])
            if second_action == 0:
                p_alien0 += float(q_pi[pol_idx])
            else:
                p_alien1 += float(q_pi[pol_idx])
        s = p_alien0 + p_alien1
        if s <= 0:
            return np.array([0.5, 0.5])
        return np.array([p_alien0 / s, p_alien1 / s])

    def set_current_planet(self, planet: str):
        self.current_planet = planet

    def update_after_feedback(self, chosen_alien: int, reward: int, planet: str):
        self._runner.update_after_feedback(chosen_alien, reward, planet)
