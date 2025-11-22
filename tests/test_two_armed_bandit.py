import pytest

from src.environment.two_armed_bandit import TwoArmedBandit


def make_env(**overrides):
    """Utility to construct deterministic environments for testing."""
    params = dict(
        context='left_better',
        probability_hint=1.0,
        probability_reward=1.0,
        reversal_schedule=[],
        observation_hints=['null', 'observe_left_hint', 'observe_right_hint'],
        observation_rewards=['null', 'observe_loss', 'observe_reward'],
    )
    params.update(overrides)
    return TwoArmedBandit(**params)


def test_start_action_produces_null_observations():
    env = make_env()
    obs = env.step('act_start')
    assert obs == ['null', 'null', 'observe_start']


def test_hint_observation_matches_context_when_probability_one():
    env = make_env(context='left_better')
    obs = env.step('act_hint')
    assert obs[0] == 'observe_left_hint'
    assert obs[1] == 'null'
    assert obs[2] == 'observe_hint'


def test_reward_observation_matches_context_when_probability_one():
    env = make_env(context='left_better')
    obs = env.step('act_left')
    assert obs[1] == 'observe_reward'
    assert obs[2] == 'observe_left'

    env = make_env(context='right_better')
    obs = env.step('act_right')
    assert obs[1] == 'observe_reward'
    assert obs[2] == 'observe_right'


def test_context_reversal_occurs_on_schedule():
    env = make_env(context='left_better', reversal_schedule=[0])
    env.step('act_start')  # reversal occurs before first action
    obs = env.step('act_hint')
    assert obs[0] == 'observe_right_hint', "Context should flip to right_better after reversal"

