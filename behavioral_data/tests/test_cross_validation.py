"""
Tests for cross-validation utilities.
"""
import pytest
import numpy as np
import pandas as pd

from behavioral_data.pipeline.cross_validation_aif import (
    loso_cv,
    temporal_split_cv,
    _create_agent,
    _score,
    PARAMETER_GRIDS,
)
from behavioral_data.aif_models.agent_pymdp import ChangepointAgentPymdp


def create_synthetic_trials(n_subjects=3, n_runs=2, n_trials_per_run=10):
    """Create synthetic trial data for testing."""
    records = []
    for subj_idx in range(n_subjects):
        subject_id = f"sub-{subj_idx:02d}"
        for run_id in range(1, n_runs + 1):
            for trial_idx in range(n_trials_per_run):
                records.append({
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "trial_index": trial_idx,
                    "outcome": np.random.uniform(0, 300),
                    "prediction": np.random.uniform(0, 300),
                    "noise_sd": 10.0,
                })
    return pd.DataFrame(records)


def test_create_agent_m1():
    """Test creating M1 agent."""
    params = {"gamma": 3.0}
    agent = _create_agent("M1", params, noise_sd=10.0)
    
    assert isinstance(agent, ChangepointAgentPymdp)
    assert agent.position_space.n_bins == 30


def test_create_agent_m2():
    """Test creating M2 agent."""
    params = {"gamma_base": 4.0, "entropy_k": 1.0}
    agent = _create_agent("M2", params, noise_sd=10.0)
    
    assert isinstance(agent, ChangepointAgentPymdp)


def test_create_agent_m3():
    """Test creating M3 agent."""
    params = {
        "gamma_stable": 8.0,
        "gamma_volatile": 2.0,
        "soft_assign": False,
    }
    agent = _create_agent("M3", params, noise_sd=10.0)
    
    assert isinstance(agent, ChangepointAgentPymdp)


def test_score():
    """Test scoring function."""
    df = create_synthetic_trials(n_subjects=1, n_runs=1, n_trials_per_run=5)
    params = {"gamma": 3.0}
    
    ll = _score(df, "M1", params)
    
    assert isinstance(ll, float)
    assert not np.isnan(ll)
    assert not np.isinf(ll)


def test_loso_cv():
    """Test leave-one-subject-out cross-validation."""
    trials = create_synthetic_trials(n_subjects=3, n_runs=2, n_trials_per_run=5)
    
    # Use smaller parameter grid for speed
    results = loso_cv(trials)
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert "model" in results.columns
    assert "subject_id" in results.columns
    assert "test_ll" in results.columns
    assert "train_ll" in results.columns
    
    # Check all models were tested
    models_tested = set(results["model"].unique())
    assert "M1" in models_tested
    assert "M2" in models_tested
    assert "M3" in models_tested


def test_temporal_split_cv():
    """Test temporal split cross-validation."""
    trials = create_synthetic_trials(n_subjects=2, n_runs=4, n_trials_per_run=5)
    
    results = temporal_split_cv(trials)
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert "model" in results.columns
    assert "subject_id" in results.columns
    assert "test_ll" in results.columns


def test_parameter_grids():
    """Test parameter grids are valid."""
    for model in ["M1", "M2", "M3"]:
        assert model in PARAMETER_GRIDS
        grid = PARAMETER_GRIDS[model]
        assert isinstance(grid, dict)
        assert len(grid) > 0
        
        # Check all values are sequences
        for key, values in grid.items():
            assert hasattr(values, "__iter__")
            assert len(values) > 0

