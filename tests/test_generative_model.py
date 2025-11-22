import numpy as np

from config import experiment_config as cfg
from src.models.generative_model import build_A, build_B, build_D


def test_build_A_shapes_and_modalities():
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
    assert len(A) == cfg.NUM_MODALITIES
    hint_mod = A[0]
    reward_mod = A[1]
    choice_mod = A[2]
    assert hint_mod.shape == (len(cfg.OBSERVATION_HINTS), len(cfg.STATE_CONTEXTS), len(cfg.STATE_CHOICES))
    assert reward_mod.shape == (len(cfg.OBSERVATION_REWARDS), len(cfg.STATE_CONTEXTS), len(cfg.STATE_CHOICES))
    assert choice_mod.shape == (len(cfg.OBSERVATION_CHOICES), len(cfg.STATE_CONTEXTS), len(cfg.STATE_CHOICES))


def test_build_A_hint_structure():
    p_hint = 0.9
    A = build_A(
        cfg.NUM_MODALITIES,
        cfg.STATE_CONTEXTS,
        cfg.STATE_CHOICES,
        cfg.OBSERVATION_HINTS,
        cfg.OBSERVATION_REWARDS,
        cfg.OBSERVATION_CHOICES,
        p_hint,
        cfg.PROBABILITY_REWARD,
    )
    hint_mod = A[0]
    idx_hint = cfg.STATE_CHOICES.index('hint')
    # column sums should be 1
    assert np.allclose(hint_mod[:, :, idx_hint].sum(axis=0), 1.0)
    left_context = cfg.STATE_CONTEXTS.index('left_better')
    right_context = cfg.STATE_CONTEXTS.index('right_better')
    # Observing hint in left context should favor left hint
    left_hint_idx = cfg.OBSERVATION_HINTS.index('observe_left_hint')
    right_hint_idx = cfg.OBSERVATION_HINTS.index('observe_right_hint')
    assert np.isclose(hint_mod[left_hint_idx, left_context, idx_hint], p_hint)
    assert np.isclose(hint_mod[right_hint_idx, left_context, idx_hint], 1 - p_hint)
    assert np.isclose(hint_mod[right_hint_idx, right_context, idx_hint], p_hint)


def test_build_A_reward_structure():
    p_reward = 0.95
    A = build_A(
        cfg.NUM_MODALITIES,
        cfg.STATE_CONTEXTS,
        cfg.STATE_CHOICES,
        cfg.OBSERVATION_HINTS,
        cfg.OBSERVATION_REWARDS,
        cfg.OBSERVATION_CHOICES,
        cfg.PROBABILITY_HINT,
        p_reward,
    )
    reward_mod = A[1]
    loss_idx = cfg.OBSERVATION_REWARDS.index('observe_loss')
    reward_idx = cfg.OBSERVATION_REWARDS.index('observe_reward')
    left_choice = cfg.STATE_CHOICES.index('left')
    right_choice = cfg.STATE_CHOICES.index('right')
    left_context = cfg.STATE_CONTEXTS.index('left_better')
    right_context = cfg.STATE_CONTEXTS.index('right_better')
    # Choosing left in left_better context should give reward prob p_reward
    assert np.isclose(reward_mod[reward_idx, left_context, left_choice], p_reward)
    assert np.isclose(reward_mod[loss_idx, left_context, left_choice], 1 - p_reward)
    # Choosing right in left_better context should flip probabilities
    assert np.isclose(reward_mod[reward_idx, left_context, right_choice], 1 - p_reward)
    assert np.isclose(reward_mod[loss_idx, left_context, right_choice], p_reward)
    # columns should sum to 1
    assert np.allclose(reward_mod[:, :, left_choice].sum(axis=0), 1.0)
    assert np.allclose(reward_mod[:, :, right_choice].sum(axis=0), 1.0)


def test_build_A_choice_modality_is_identity():
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
    choice_mod = A[2]
    # For each state choice, the observation should be deterministic (identity mapping)
    num_choices = len(cfg.STATE_CHOICES)
    for choice_idx in range(num_choices):
        column = choice_mod[:, :, choice_idx]
        expected_row = choice_idx  # observation choice labels match state choice ordering
        assert np.allclose(column[expected_row], np.ones(len(cfg.STATE_CONTEXTS)))
        mask = np.ones_like(column, dtype=bool)
        mask[expected_row] = False
        assert np.allclose(column[mask], 0.0)


def test_build_B_context_volatility_zero_identity():
    B = build_B(
        cfg.STATE_CONTEXTS,
        cfg.STATE_CHOICES,
        cfg.ACTION_CONTEXTS,
        cfg.ACTION_CHOICES,
        context_volatility=0.0,
    )
    B_context = B[0]
    identity = np.eye(len(cfg.STATE_CONTEXTS))
    assert np.allclose(B_context[:, :, 0], identity)


def test_build_B_context_volatility_nonzero():
    vol = 0.2
    B = build_B(
        cfg.STATE_CONTEXTS,
        cfg.STATE_CHOICES,
        cfg.ACTION_CONTEXTS,
        cfg.ACTION_CHOICES,
        context_volatility=vol,
    )
    B_context = B[0][:, :, 0]
    assert np.allclose(B_context.sum(axis=0), 1.0)
    for i in range(B_context.shape[0]):
        assert np.isclose(B_context[i, i], 1 - vol)


def test_build_B_choice_mapping():
    B = build_B(
        cfg.STATE_CONTEXTS,
        cfg.STATE_CHOICES,
        cfg.ACTION_CONTEXTS,
        cfg.ACTION_CHOICES,
        context_volatility=0.0,
    )
    B_choice = B[1]
    num_choices = len(cfg.STATE_CHOICES)
    for action_idx in range(num_choices):
        column = B_choice[:, :, action_idx]
        expected_next = np.zeros(num_choices)
        expected_next[action_idx] = 1.0
        assert np.allclose(column, expected_next[:, None])


def test_build_D_defaults():
    D = build_D(cfg.STATE_CONTEXTS, cfg.STATE_CHOICES)
    D_context = D[0]
    D_choice = D[1]
    assert np.allclose(D_context, 0.5)
    start_idx = cfg.STATE_CHOICES.index('start')
    expected = np.zeros(len(cfg.STATE_CHOICES))
    expected[start_idx] = 1.0
    assert np.allclose(D_choice, expected)

