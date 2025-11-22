import numpy as np
import pytest

from pymdp.maths import softmax

from src.models.value_functions import (
    make_values_M1,
    make_values_M2,
    make_values_M3,
    map_action_prefs_to_policy_prefs,
    make_value_fn,
)


def test_make_values_m1_returns_fixed_preferences_and_gamma():
    C_logits = np.array([0.0, -5.0, 5.0])
    E_logits = np.array([0.0, 1.0])
    gamma = 3.5
    fn = make_values_M1(C_reward_logits=C_logits, gamma=gamma, E_logits=E_logits)

    q = np.array([0.7, 0.3])
    C_t, E_t, gamma_t = fn(q, t=0)
    assert np.allclose(C_t, softmax(C_logits))
    assert np.allclose(E_t, softmax(E_logits))
    assert gamma_t == pytest.approx(gamma)


def test_make_values_m2_uses_entropy_schedule():
    C_logits = np.array([0.0, 0.0, 0.0])
    fn = make_values_M2(C_reward_logits=C_logits)
    q_certain = np.array([1.0, 0.0])
    q_uncertain = np.array([0.5, 0.5])

    _, _, gamma_certain = fn(q_certain, t=0)
    _, _, gamma_uncertain = fn(q_uncertain, t=0)
    assert gamma_certain > gamma_uncertain


def test_make_values_m2_custom_schedule():
    def schedule(q, t):
        return 10.0 * q[0]

    fn = make_values_M2(C_reward_logits=[0, 0, 0], gamma_schedule=schedule)
    _, _, gamma_val = fn(np.array([0.6, 0.4]), t=0)
    assert gamma_val == pytest.approx(6.0)


def dummy_policies():
    # Two policies, each length 1 with control factors [context, choice].
    return [
        np.array([[0, 2]]),  # choose action index 2
        np.array([[0, 3]]),  # choose action index 3
    ]


def test_map_action_prefs_to_policy_prefs():
    xi_logits = np.array([0.0, 0.0, 1.0, -1.0])
    policies = dummy_policies()
    policy_logits = map_action_prefs_to_policy_prefs(xi_logits, policies, [1, 4])
    assert policy_logits.shape == (2,)
    assert policy_logits[0] == pytest.approx(1.0)
    assert policy_logits[1] == pytest.approx(-1.0)


def test_make_values_m3_mixes_profiles():
    profiles = [
        {'phi_logits': [0.0, 0.0, 10.0], 'xi_logits': [0, 0, 1, -1], 'gamma': 2.0},
        {'phi_logits': [0.0, 0.0, -10.0], 'xi_logits': [0, 0, -1, 1], 'gamma': 1.0},
    ]
    Z = np.array([[1.0, 0.0], [0.0, 1.0]])
    policies = dummy_policies()
    fn = make_values_M3(
        profiles=profiles,
        Z=Z,
        policies=policies,
        num_actions_per_factor=[1, 4],
    )
    q_context = np.array([0.75, 0.25])
    C_t, E_t, gamma_t = fn(q_context, t=5)

    # Outcome preference should lean toward profile 0 (strong positive reward)
    assert C_t[2] > C_t[1]
    # Policy precision should be weighted average of profile gammas
    assert gamma_t == pytest.approx(0.75 * 2.0 + 0.25 * 1.0)
    # Policy priors mix the xi logits
    assert E_t is not None
    assert np.isclose(E_t.sum(), 1.0)


def test_make_values_m3_requires_policies_if_xi_present():
    profiles = [{'phi_logits': [0, 0, 0], 'xi_logits': [0, 0, 0, 0], 'gamma': 1.0}]
    with pytest.raises(ValueError):
        make_values_M3(profiles=profiles, Z=[[1.0]])


def test_make_value_fn_dispatches_models():
    m1_fn = make_value_fn('M1', C_reward_logits=[0, 0, 0], gamma=1.0)
    assert callable(m1_fn)
    m2_fn = make_value_fn('M2', C_reward_logits=[0, 0, 0])
    assert callable(m2_fn)
    profiles = [{'phi_logits': [0, 0, 0], 'gamma': 1.0}]
    m3_fn = make_value_fn('M3', profiles=profiles, Z=[[1.0]], policies=[np.array([[0, 0]])], num_actions_per_factor=[1, 1])
    assert callable(m3_fn)
    with pytest.raises(ValueError):
        make_value_fn('M4')

