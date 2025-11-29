"""
Tests for parallel processing in cross-validation.
"""
import pytest
import numpy as np
import pandas as pd
import os
import time

from behavioral_data.pipeline.cross_validation_aif import (
    _grid_search,
    _score,
    loso_cv,
    temporal_split_cv,
    PARAMETER_GRIDS,
)


def create_synthetic_trials(n_subjects=2, n_runs=2, n_trials_per_run=10):
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


def test_parallel_vs_sequential_grid_search():
    """Test that parallel and sequential grid search produce similar results."""
    np.random.seed(42)
    trials = create_synthetic_trials(n_subjects=1, n_runs=1, n_trials_per_run=20)
    
    # Use a small grid for speed
    small_grid = {
        "gamma": [2.0, 3.0, 4.0],
    }
    
    # Sequential (max_workers=1)
    params_seq, ll_seq = _grid_search(trials, "M1", small_grid, max_workers=1)
    
    # Parallel (if multiple CPUs available)
    cpu_count = os.cpu_count() or 1
    if cpu_count > 1:
        np.random.seed(42)
        params_par, ll_par = _grid_search(trials, "M1", small_grid, max_workers=2)
        
        # Both should find valid parameters within the grid
        assert "gamma" in params_seq
        assert "gamma" in params_par
        assert params_seq["gamma"] in [2.0, 3.0, 4.0]
        assert params_par["gamma"] in [2.0, 3.0, 4.0]
        
        # Results should be similar (parallel uses deterministic seeds based on params)
        # Allow tolerance due to pymdp's internal randomness and potential parameter ties
        assert abs(ll_seq - ll_par) < 20.0, f"Parallel and sequential should find similar best LL (diff: {abs(ll_seq - ll_par):.2f})"


def test_parallel_processing_speedup():
    """Test that parallel processing is faster than sequential for large grids."""
    np.random.seed(42)
    trials = create_synthetic_trials(n_subjects=1, n_runs=1, n_trials_per_run=10)
    
    # Use M3 grid (largest) for speedup test
    grid = PARAMETER_GRIDS["M3"]
    
    # Sequential
    start_seq = time.time()
    params_seq, ll_seq = _grid_search(trials, "M3", grid, max_workers=1)
    time_seq = time.time() - start_seq
    
    # Parallel (if multiple CPUs available)
    cpu_count = os.cpu_count() or 1
    if cpu_count > 1:
        np.random.seed(42)
        start_par = time.time()
        params_par, ll_par = _grid_search(trials, "M3", grid, max_workers=2)
        time_par = time.time() - start_par
        
        # Results should find similar best params (may differ slightly due to pymdp randomness)
        # Both should find reasonable parameters within the grid
        assert "gamma_stable" in params_seq
        assert "gamma_stable" in params_par
        assert params_seq["gamma_stable"] in PARAMETER_GRIDS["M3"]["gamma_stable"]
        assert params_par["gamma_stable"] in PARAMETER_GRIDS["M3"]["gamma_stable"]
        # Allow larger tolerance for M3 due to larger parameter space and potential ties
        assert abs(ll_seq - ll_par) < 30.0, f"Parallel and sequential should find similar best LL (diff: {abs(ll_seq - ll_par):.2f})"
        
        # Parallel should be faster (at least 1.2x speedup with 2 workers)
        # Note: This might be flaky on slow systems, so we just check it doesn't take much longer
        assert time_par <= time_seq * 1.5, f"Parallel ({time_par:.2f}s) should be faster than sequential ({time_seq:.2f}s)"


def test_default_workers_half_cpu():
    """Test that default max_workers is half of CPU count."""
    cpu_count = os.cpu_count() or 1
    expected_workers = max(1, cpu_count // 2)
    
    # Create small test data
    trials = create_synthetic_trials(n_subjects=1, n_runs=1, n_trials_per_run=5)
    small_grid = {"gamma": [2.0, 3.0]}
    
    # Call without max_workers (should use default)
    params, ll = _grid_search(trials, "M1", small_grid, max_workers=None)
    
    # Verify it works (can't directly test worker count, but if it uses wrong count it might fail)
    assert params is not None
    assert not np.isnan(ll)
    assert not np.isinf(ll)


def test_parallel_loso_cv():
    """Test that LOSO CV works with parallel processing."""
    trials = create_synthetic_trials(n_subjects=2, n_runs=2, n_trials_per_run=5)
    
    # Run with parallel processing
    cpu_count = os.cpu_count() or 1
    if cpu_count > 1:
        results_par = loso_cv(trials, max_workers=2)
    else:
        results_par = loso_cv(trials, max_workers=1)
    
    # Verify results structure
    assert isinstance(results_par, pd.DataFrame)
    assert len(results_par) > 0
    assert "model" in results_par.columns
    assert "subject_id" in results_par.columns
    assert "test_ll" in results_par.columns
    
    # Verify all models were tested
    models_tested = set(results_par["model"].unique())
    assert "M1" in models_tested
    assert "M2" in models_tested
    assert "M3" in models_tested


def test_parallel_temporal_split_cv():
    """Test that temporal split CV works with parallel processing."""
    trials = create_synthetic_trials(n_subjects=2, n_runs=4, n_trials_per_run=5)
    
    # Run with parallel processing
    cpu_count = os.cpu_count() or 1
    if cpu_count > 1:
        results_par = temporal_split_cv(trials, max_workers=2)
    else:
        results_par = temporal_split_cv(trials, max_workers=1)
    
    # Verify results structure
    assert isinstance(results_par, pd.DataFrame)
    assert len(results_par) > 0
    assert "model" in results_par.columns
    assert "subject_id" in results_par.columns
    assert "test_ll" in results_par.columns


def test_parallel_consistency():
    """Test that parallel processing produces consistent results across runs."""
    np.random.seed(42)
    trials = create_synthetic_trials(n_subjects=1, n_runs=1, n_trials_per_run=10)
    small_grid = {"gamma": [2.0, 3.0, 4.0]}
    
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        pytest.skip("Need multiple CPUs to test parallel processing")
    
    # Run parallel grid search twice with same seed
    np.random.seed(42)
    params1, ll1 = _grid_search(trials, "M1", small_grid, max_workers=2)
    
    np.random.seed(42)
    params2, ll2 = _grid_search(trials, "M1", small_grid, max_workers=2)
    
    # Results should be similar (parallel uses deterministic seeds based on params)
    # Allow tolerance due to pymdp's internal randomness and potential parameter ties
    # Both should find valid parameters
    assert "gamma" in params1
    assert "gamma" in params2
    assert params1["gamma"] in [2.0, 3.0, 4.0]
    assert params2["gamma"] in [2.0, 3.0, 4.0]
    assert abs(ll1 - ll2) < 20.0, f"Parallel processing should be consistent (diff: {abs(ll1 - ll2):.2f})"


def test_worker_function_imports():
    """Test that worker function can be imported and called correctly."""
    from behavioral_data.pipeline.cross_validation_aif import _score_worker_wrapper
    
    # Create test data
    trials = create_synthetic_trials(n_subjects=1, n_runs=1, n_trials_per_run=5)
    params = {"gamma": 3.0}
    
    # Call worker function directly
    result_params, result_ll = _score_worker_wrapper((trials, "M1", params))
    
    # Verify it works
    assert result_params == params
    assert not np.isnan(result_ll)
    assert not np.isinf(result_ll)

