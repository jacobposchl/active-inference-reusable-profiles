import numpy as np

from config import experiment_config as cfg
from src.models.generative_model import build_A, build_B, build_D
from src.models.agent_wrapper import AgentRunner, AgentRunnerWithLL


def build_generative_components():
    A = build_A(
        cfg.NUM_MODALITIES,
        cfg.STATE_CONTEXTS,
        cfg.STATE_CHOICES,
        cfg.OBSERVATION_HINTS,
        cfg.OBSERVATION_REWARDS,
        cfg.OBSERVATION_CHOICES,
        cfg.PROBABILITY_HINT,
        cfg.PROBABILITY_REWARD,
    )
    B = build_B(
        cfg.STATE_CONTEXTS,
        cfg.STATE_CHOICES,
        cfg.ACTION_CONTEXTS,
        cfg.ACTION_CHOICES,
        context_volatility=cfg.DEFAULT_CONTEXT_VOLATILITY,
    )
    D = build_D(cfg.STATE_CONTEXTS, cfg.STATE_CHOICES)
    return A, B, D


def constant_value_fn(gamma=2.5):
    C_vec = np.array([0.05, 0.1, 0.85])

    def fn(q_context, t):
        return C_vec, None, gamma

    return fn, C_vec


def test_agent_runner_updates_preferences_and_gamma():
    A, B, D = build_generative_components()
    value_fn, expected_C = constant_value_fn(gamma=4.2)
    runner = AgentRunner(
        A,
        B,
        D,
        value_fn,
        cfg.OBSERVATION_HINTS,
        cfg.OBSERVATION_REWARDS,
        cfg.OBSERVATION_CHOICES,
        cfg.ACTION_CHOICES,
        reward_mod_idx=1,
        policy_len=2,
        inference_horizon=1,
    )

    obs_ids = runner.obs_labels_to_ids(['null', 'null', 'observe_start'])
    np.random.seed(0)
    action_label, qs, q_pi, efe, gamma_t = runner.step(obs_ids, t=0)

    assert gamma_t == 4.2
    assert np.allclose(runner.agent.C[runner.reward_mod_idx], expected_C)
    assert isinstance(action_label, str)
    assert len(qs) == cfg.NUM_FACTORS
    assert len(q_pi) == len(runner.agent.policies)
    assert np.isfinite(efe).all()


def test_agent_runner_with_ll_returns_log_prob():
    A, B, D = build_generative_components()
    value_fn, _ = constant_value_fn(gamma=3.0)
    runner = AgentRunnerWithLL(
        A,
        B,
        D,
        value_fn,
        cfg.OBSERVATION_HINTS,
        cfg.OBSERVATION_REWARDS,
        cfg.OBSERVATION_CHOICES,
        cfg.ACTION_CHOICES,
        reward_mod_idx=1,
    )

    obs_ids = runner.obs_labels_to_ids(['null', 'null', 'observe_start'])
    np.random.seed(1)
    action_label, qs, q_pi, efe, gamma_t, ll = runner.step_with_ll(obs_ids, t=0)

    assert isinstance(action_label, str)
    assert np.isfinite(ll)
    assert gamma_t == 3.0

    # action_logprob should match log-likelihood when conditioning on same obs/action
    obs_ids = runner.obs_labels_to_ids(['null', 'null', 'observe_start'])
    ll_again = runner.action_logprob(obs_ids, action_label, t=0)
    assert np.isfinite(ll_again)

