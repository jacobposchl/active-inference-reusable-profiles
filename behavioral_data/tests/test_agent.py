"""
Tests for changepoint agent using pymdp.
"""
import pytest
import numpy as np

from behavioral_data.aif_models.agent_pymdp import (
    ChangepointAgentPymdp,
    AgentConfig,
)
from behavioral_data.aif_models.value_functions import (
    make_values_M1,
    make_values_M2,
    make_values_M3_with_volatility,
)


def test_agent_initialization():
    """Test agent initializes correctly."""
    value_fn = make_values_M1(gamma=3.0)
    config = AgentConfig(n_position_bins=10, noise_sd=10.0)
    agent = ChangepointAgentPymdp(value_fn, config)
    
    assert agent.agent is not None
    assert agent.position_space.n_bins == 10
    assert len(agent.agent.qs) == 1  # One state factor
    assert agent.agent.qs[0].shape == (10,)  # Position beliefs


def test_agent_predict():
    """Test agent generates predictions."""
    value_fn = make_values_M1(gamma=3.0)
    config = AgentConfig(n_position_bins=10, noise_sd=10.0)
    agent = ChangepointAgentPymdp(value_fn, config)
    
    pred_mean, pred_std = agent.predict()
    
    # Check prediction is in valid range
    assert 0.0 <= pred_mean <= 300.0
    assert pred_std > 0
    assert len(agent.prediction_history) == 1
    assert len(agent.gamma_history) == 1


def test_agent_observe():
    """Test agent updates beliefs from observations."""
    value_fn = make_values_M1(gamma=3.0)
    config = AgentConfig(n_position_bins=10, noise_sd=10.0)
    agent = ChangepointAgentPymdp(value_fn, config)
    
    # Make initial prediction
    agent.predict()
    
    # Observe outcome
    agent.observe(150.0)
    
    # Check beliefs updated
    assert len(agent.outcome_history) == 1
    assert agent.outcome_history[0] == 150.0
    assert len(agent.prediction_error_history) == 1


def test_agent_reset():
    """Test agent resets correctly."""
    value_fn = make_values_M1(gamma=3.0)
    config = AgentConfig(n_position_bins=10, noise_sd=10.0)
    agent = ChangepointAgentPymdp(value_fn, config)
    
    # Make some predictions and observations
    agent.predict()
    agent.observe(100.0)
    agent.predict()
    agent.observe(200.0)
    
    # Reset
    agent.reset()
    
    # Check history cleared
    assert len(agent.prediction_history) == 0
    assert len(agent.outcome_history) == 0
    assert len(agent.prediction_error_history) == 0


def test_agent_m2_adaptation():
    """Test M2 agent adapts precision."""
    value_fn = make_values_M2(gamma_base=5.0, entropy_k=1.0)
    config = AgentConfig(n_position_bins=10, noise_sd=10.0)
    agent = ChangepointAgentPymdp(value_fn, config)
    
    # Initial prediction (high uncertainty)
    pred1, _ = agent.predict()
    gamma1 = agent.gamma_history[0]
    
    # Observe and update
    agent.observe(150.0)
    
    # Second prediction (should have updated beliefs)
    pred2, _ = agent.predict()
    gamma2 = agent.gamma_history[1]
    
    # Gamma might change if entropy changes
    assert isinstance(gamma1, float)
    assert isinstance(gamma2, float)


def test_agent_m3_profile_mixing():
    """Test M3 agent mixes profiles."""
    n_bins = 10
    profiles = [
        {"phi_logits": np.zeros(n_bins), "gamma": 2.0},
        {"phi_logits": np.zeros(n_bins), "gamma": 8.0},
    ]
    Z = np.array([[0.0, 1.0], [1.0, 0.0]])
    
    # Need policies for M3
    from pymdp.agent import Agent
    from pymdp import utils as pymdp_utils
    from behavioral_data.aif_models.generative_model import build_generative_model
    from behavioral_data.aif_models.state_space import create_state_space
    
    position_space, _ = create_state_space(n_bins)
    A, B, D = build_generative_model(position_space, noise_sd=10.0)
    C_temp = pymdp_utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
    temp_agent = Agent(A=A, B=B, C=C_temp, D=D, policy_len=1, inference_horizon=1,
                      control_fac_idx=[0], use_utility=True, use_states_info_gain=True,
                      action_selection="stochastic", gamma=16.0)
    
    value_fn = make_values_M3_with_volatility(
        profiles=profiles,
        Z=Z,
        policies=temp_agent.policies,
        num_actions_per_factor=[n_bins],
    )
    
    config = AgentConfig(n_position_bins=n_bins, noise_sd=10.0)
    agent = ChangepointAgentPymdp(value_fn, config)
    
    # Make prediction
    pred, _ = agent.predict()
    
    # Check gamma is set
    assert len(agent.gamma_history) == 1
    assert 2.0 <= agent.gamma_history[0] <= 8.0  # Should be between profiles


def test_agent_full_sequence():
    """Test agent on a full sequence of trials."""
    value_fn = make_values_M1(gamma=3.0)
    config = AgentConfig(n_position_bins=10, noise_sd=10.0)
    agent = ChangepointAgentPymdp(value_fn, config)
    
    outcomes = np.array([100.0, 150.0, 200.0, 180.0, 160.0])
    predictions = np.array([90.0, 140.0, 190.0, 170.0, 150.0])
    
    for outcome, prediction in zip(outcomes, predictions):
        pred_mean, pred_std = agent.predict()
        agent.observe(outcome)
    
    # Check history
    assert len(agent.prediction_history) == len(outcomes)
    assert len(agent.outcome_history) == len(outcomes)
    assert len(agent.gamma_history) == len(outcomes)


def test_agent_belief_statistics():
    """Test agent belief statistics."""
    value_fn = make_values_M1(gamma=3.0)
    config = AgentConfig(n_position_bins=10, noise_sd=10.0)
    agent = ChangepointAgentPymdp(value_fn, config)
    
    stats = agent.get_belief_statistics()
    
    assert "belief_mean" in stats
    assert "belief_variance" in stats
    assert "belief_entropy" in stats
    assert "volatility_stable" in stats
    assert "volatility_volatile" in stats
    
    # Check values are reasonable
    assert 0.0 <= stats["belief_mean"] <= 300.0
    assert stats["belief_variance"] >= 0
    assert stats["belief_entropy"] >= 0
    assert abs(stats["volatility_stable"] + stats["volatility_volatile"] - 1.0) < 1e-6

