import numpy as np

from config import experiment_config as cfg
from src.models.generative_model import build_A, build_B, build_D
from src.utils.ll_eval import (
    compute_sequence_ll_for_model,
    evaluate_ll_with_valuefn,
    evaluate_ll_with_valuefn_masked,
)
from src.models.value_functions import make_values_M1


def build_components():
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


def simple_ref_logs(num_trials=5):
    return {
        'action': ['act_start'] + ['act_left'] * (num_trials - 1),
        'context': ['left_better'] * num_trials,
        'reward_label': ['null'] + ['observe_reward'] * (num_trials - 1),
        'choice_label': ['observe_start'] + ['observe_left'] * (num_trials - 1),
        'hint_label': ['null'] * num_trials,
    }


def test_compute_sequence_ll_for_model_m1():
    A, B, D = build_components()
    logs = simple_ref_logs()
    ll_seq = compute_sequence_ll_for_model('M1', A, B, D, logs)
    assert isinstance(ll_seq, list)
    assert len(ll_seq) == len(logs['action'])
    assert all(np.isfinite(ll) for ll in ll_seq)


def test_evaluate_ll_with_valuefn_total_and_sequence():
    A, B, D = build_components()
    logs = simple_ref_logs()
    value_fn = make_values_M1(C_reward_logits=[0.0, -5.0, 5.0], gamma=2.5)
    total_ll, ll_seq = evaluate_ll_with_valuefn(value_fn, A, B, D, logs)
    assert len(ll_seq) == len(logs['action'])
    assert np.isclose(total_ll, np.sum(ll_seq))


def test_evaluate_ll_with_valuefn_masked():
    A, B, D = build_components()
    logs = simple_ref_logs(num_trials=6)
    value_fn = make_values_M1(C_reward_logits=[0.0, -5.0, 5.0], gamma=2.5)
    mask_indices = [1, 3, 5]
    masked_sum, ll_seq = evaluate_ll_with_valuefn_masked(value_fn, A, B, D, logs, mask_indices)
    assert len(ll_seq) == len(logs['action'])
    expected = sum(ll_seq[idx] for idx in mask_indices)
    assert np.isclose(masked_sum, expected)

