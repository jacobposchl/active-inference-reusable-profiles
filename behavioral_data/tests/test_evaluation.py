"""
Tests for evaluation utilities.
"""
import pytest
import numpy as np
import pandas as pd

from behavioral_data.aif_models.evaluation import (
    compute_prediction_ll,
    evaluate_agent,
    evaluate_agent_on_dataframe,
)
from behavioral_data.aif_models.agent_pymdp import (
    ChangepointAgentPymdp,
    AgentConfig,
)
from behavioral_data.aif_models.value_functions import make_values_M1


def test_compute_prediction_ll():
    """Test prediction log-likelihood computation."""
    # Perfect prediction
    ll = compute_prediction_ll(predicted_mean=100.0, predicted_std=10.0, observed_prediction=100.0)
    assert isinstance(ll, float)
    assert ll > -10.0  # Should be reasonable (not too negative)
    
    # Prediction far from observed
    ll_far = compute_prediction_ll(predicted_mean=100.0, predicted_std=10.0, observed_prediction=200.0)
    assert ll_far < ll  # Should be lower likelihood
    
    # Higher uncertainty should give higher likelihood for same distance
    ll_wide = compute_prediction_ll(predicted_mean=100.0, predicted_std=50.0, observed_prediction=150.0)
    ll_narrow = compute_prediction_ll(predicted_mean=100.0, predicted_std=10.0, observed_prediction=150.0)
    assert ll_wide > ll_narrow  # Wider std = higher likelihood for distant predictions


def test_evaluate_agent():
    """Test agent evaluation on sequence."""
    value_fn = make_values_M1(gamma=3.0)
    config = AgentConfig(n_position_bins=10, noise_sd=10.0)
    agent = ChangepointAgentPymdp(value_fn, config)
    
    outcomes = np.array([100.0, 150.0, 200.0])
    predictions = np.array([90.0, 140.0, 190.0])
    
    total_ll, trial_lls = evaluate_agent(agent, outcomes, predictions, reset_between_runs=True)
    
    assert isinstance(total_ll, float)
    assert len(trial_lls) == len(outcomes)
    assert abs(total_ll - trial_lls.sum()) < 1e-6


def test_evaluate_agent_on_dataframe():
    """Test agent evaluation on DataFrame."""
    value_fn = make_values_M1(gamma=3.0)
    config = AgentConfig(n_position_bins=10, noise_sd=10.0)
    agent = ChangepointAgentPymdp(value_fn, config)
    
    # Create test DataFrame
    df = pd.DataFrame({
        "outcome": [100.0, 150.0, 200.0],
        "prediction": [90.0, 140.0, 190.0],
        "run_id": [1, 1, 1],
    })
    
    total_ll, trial_lls = evaluate_agent_on_dataframe(
        agent, df, outcome_col="outcome", prediction_col="prediction"
    )
    
    assert isinstance(total_ll, float)
    assert len(trial_lls) == len(df)


def test_evaluate_agent_multiple_runs():
    """Test agent evaluation with multiple runs."""
    value_fn = make_values_M1(gamma=3.0)
    config = AgentConfig(n_position_bins=10, noise_sd=10.0)
    agent = ChangepointAgentPymdp(value_fn, config)
    
    # Create DataFrame with multiple runs
    df = pd.DataFrame({
        "outcome": [100.0, 150.0, 200.0, 120.0, 180.0],
        "prediction": [90.0, 140.0, 190.0, 110.0, 170.0],
        "run_id": [1, 1, 1, 2, 2],
    })
    
    total_ll, trial_lls = evaluate_agent_on_dataframe(
        agent, df, outcome_col="outcome", prediction_col="prediction", group_by="run_id"
    )
    
    assert isinstance(total_ll, float)
    assert len(trial_lls) == len(df)
    # Agent should reset between runs, so we should get valid results

