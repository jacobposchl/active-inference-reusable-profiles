"""Agent runner wrapper for two-step task using `pymdp.Agent`.

Provides a small API for computing action log-likelihoods given
observations and a `value_fn` that returns trial-specific values.
"""
from typing import List, Tuple
import numpy as np

from pymdp.agent import Agent
from pymdp import utils


class AgentRunner:
    """Wraps a `pymdp.Agent` for trial stepping with a `value_fn`.

    `value_fn` should be a callable: value_fn(q_context, t) -> (C_eff, E_t, gamma_t)
    where `C_eff` is a scalar preference strength for the reward outcome
    (AgentRunner maps that scalar to the reward modality C vector), `E_t`
    is an optional array of policy priors, and `gamma_t` is the policy precision.
    """

    def __init__(self, agent: Agent, reward_mod_idx: int, value_fn, learning_mode: str = "dirichlet"):
        self.agent = agent
        self.value_fn = value_fn
        self.reward_mod_idx = int(reward_mod_idx)
        self.learning_mode = str(learning_mode)

        # For Dirichlet learning we maintain concentration arrays per modality
        self.A_conc = None
        if self.learning_mode == "dirichlet":
            # initialize concentration arrays from agent.A shapes
            try:
                self.A_conc = [np.ones_like(np.asarray(a), dtype=float) for a in self.agent.A]
            except Exception:
                # fallback: create ones with the same shape as agent.A entries
                self.A_conc = [np.ones(np.asarray(a).shape, dtype=float) for a in self.agent.A]

        # store last inferred posterior over states (qs) for updates
        self._last_qs = None

    def _make_C_from_scalar(self, C_eff: float) -> np.ndarray:
        n_outcomes = self.agent.A[self.reward_mod_idx].shape[0]
        logits = np.zeros(n_outcomes, dtype=float)
        logits[-1] = float(C_eff)
        logits = logits - np.max(logits)
        expx = np.exp(logits)
        return expx / np.sum(expx)

    def compute_context_beliefs(self) -> np.ndarray:
        """Return current posterior over contexts.

        Prefer the most recently inferred posterior from the agent; if not
        available, return a uniform context belief.
        """
        if self._last_qs is not None:
            return np.asarray(self._last_qs[0], dtype=float)
        # fallback: uniform over 4 contexts
        return np.ones(4, dtype=float) / 4.0

    def _prepare_agent_for_trial(self, q_context: np.ndarray, t: int):
        C_eff, E_t, gamma_t = self.value_fn(q_context, t)
        C_vec = self._make_C_from_scalar(C_eff)
        self.agent.C[self.reward_mod_idx] = C_vec
        if E_t is not None and len(E_t) == len(self.agent.policies):
            self.agent.E = np.asarray(E_t, dtype=float)
        self.agent.gamma = float(gamma_t)

    def infer_policy_posterior(self, obs_ids: List[int], t: int) -> Tuple[np.ndarray, np.ndarray]:
        qs = self.agent.infer_states(obs_ids)
        q_context = qs[0]
        # save for later learning updates
        self._last_qs = qs
        # prepare agent params for trial
        self._prepare_agent_for_trial(q_context, t)
        q_pi, efe = self.agent.infer_policies()
        return q_pi, efe

    def log_likelihood_of_action(self, obs_ids: List[int], action_stage: int, action_choice: int, t: int) -> float:
        """Return log-probability of `action_choice` at `action_stage` (0-based) given `obs_ids`."""
        q_pi, _ = self.infer_policy_posterior(obs_ids, t)
        action_ll = -np.inf
        for pi_idx, policy in enumerate(self.agent.policies):
            if int(policy[action_stage, 1]) == int(action_choice):
                if action_ll == -np.inf:
                    action_ll = np.log(q_pi[pi_idx] + 1e-16)
                else:
                    action_ll = np.logaddexp(action_ll, np.log(q_pi[pi_idx] + 1e-16))
        return float(action_ll)

    def update_after_feedback(self, chosen_alien: int, reward: int, planet: str = None, obs_ids: List[int] = None):
        """Update agent likelihoods using Dirichlet learning.

        Parameters
        - chosen_alien, reward, planet: legacy args kept for backward compat.
        - obs_ids: optional list of modality observation indices (preferred).

        This function increments concentration parameters for the reward
        modality according to the posterior over hidden states inferred in
        the previous call to `infer_policy_posterior`.
        """
        if self.learning_mode != "dirichlet":
            raise ValueError("Only 'dirichlet' learning_mode is supported; remove alpha/beta usage.")

        if self.A_conc is None:
            raise RuntimeError("A_conc not initialized for dirichlet learning")

        # decide observed outcome index for reward modality (assume 0=null, 1=no_reward, 2=reward)
        if obs_ids is not None:
            try:
                obs_idx = int(obs_ids[self.reward_mod_idx])
            except Exception:
                obs_idx = 1 + int(bool(int(reward)))
        else:
            obs_idx = 1 + int(bool(int(reward)))

        # require last inferred qs to attribute responsibility across hidden states
        if self._last_qs is None:
            # no posterior available; fall back to uniform update across contexts and stages
            for m_idx, conc in enumerate(self.A_conc):
                if m_idx == self.reward_mod_idx:
                    conc[obs_idx, ...] += 1.0
        else:
            qs = self._last_qs
            # assume two factors: context (qs[0]) and stage (qs[1])
            q_context = np.asarray(qs[0], dtype=float)
            q_stage = np.asarray(qs[1], dtype=float) if len(qs) > 1 else np.array([1.0])

            conc = self.A_conc[self.reward_mod_idx]
            # conc shape expected: (n_outcomes, n_contexts, n_stages)
            for ci, qc in enumerate(q_context):
                for si, qsval in enumerate(q_stage):
                    conc[obs_idx, ci, si] += float(qc * qsval)

        # Normalize concentrations to produce new likelihoods for agent.A
        for m_idx, conc in enumerate(self.A_conc):
            denom = conc.sum(axis=0, keepdims=True)
            denom[denom == 0] = 1.0
            newA = conc / denom
            try:
                self.agent.A[m_idx] = newA
            except Exception:
                self.agent.A[m_idx] = np.asarray(newA)


class AgentRunnerWithLL(AgentRunner):
    """Thin subclass kept for API parity; uses AgentRunner.log_likelihood_of_action."""

    pass
