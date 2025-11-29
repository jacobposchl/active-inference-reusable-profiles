"""
Tests for generative model construction.
"""
import pytest
import numpy as np
from pymdp import utils

from behavioral_data.aif_models.generative_model import (
    build_A,
    build_B,
    build_D,
    build_generative_model,
)
from behavioral_data.aif_models.state_space import create_state_space


def test_build_A_shape():
    """Test A matrix has correct shape and is pymdp format."""
    position_space, _ = create_state_space(n_position_bins=10)
    A = build_A(position_space, noise_sd=10.0)
    
    # Should be pymdp object array
    assert isinstance(A, np.ndarray)
    assert len(A) == 1  # One observation modality
    assert A[0].shape == (10, 10)  # (n_obs_bins, n_state_bins)


def test_build_A_normalization():
    """Test A matrix columns sum to 1."""
    position_space, _ = create_state_space(n_position_bins=10)
    A = build_A(position_space, noise_sd=10.0)
    
    # Each column should sum to 1 (likelihood over observations)
    col_sums = A[0].sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, rtol=1e-6)


def test_build_B_shape():
    """Test B matrix has correct shape and is pymdp format."""
    position_space, _ = create_state_space(n_position_bins=10)
    B = build_B(position_space, hazard_rate=0.1)
    
    # Should be pymdp object array
    assert isinstance(B, np.ndarray)
    assert len(B) == 1  # One state factor
    assert B[0].shape == (10, 10, 10)  # (n_new_states, n_old_states, n_actions)


def test_build_B_normalization():
    """Test B matrix columns sum to 1."""
    position_space, _ = create_state_space(n_position_bins=10)
    B = build_B(position_space, hazard_rate=0.1)
    
    # Each column should sum to 1 (transition probabilities)
    for a in range(10):
        col_sums = B[0][:, :, a].sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, rtol=1e-6)


def test_build_B_hazard_rate():
    """Test B matrix respects hazard rate."""
    position_space, _ = create_state_space(n_position_bins=10)
    B = build_B(position_space, hazard_rate=0.2)
    
    # After normalization, diagonal should be approximately (1 - hazard_rate)
    # but slightly higher due to uniform component added to all elements
    # Diagonal = (1 - h) + (h / n_bins), then normalized
    # For h=0.2, n_bins=10: diagonal ≈ 0.8 + 0.02 = 0.82 before normalization
    # After normalization to sum to 1, it's approximately 0.82
    for a in range(10):
        diagonal = np.diag(B[0][:, :, a])
        # Check that diagonal is close to expected (allowing for normalization)
        expected = 0.8 + (0.2 / 10)  # 0.82 before normalization
        np.testing.assert_allclose(diagonal, expected, rtol=0.05)  # Allow 5% tolerance


def test_build_D_shape():
    """Test D matrix has correct shape and is pymdp format."""
    position_space, _ = create_state_space(n_position_bins=10)
    D = build_D(position_space)
    
    # Should be pymdp object array
    assert isinstance(D, np.ndarray)
    assert len(D) == 1  # One state factor
    assert D[0].shape == (10,)  # Prior over initial states


def test_build_D_normalization():
    """Test D vector sums to 1."""
    position_space, _ = create_state_space(n_position_bins=10)
    D = build_D(position_space)
    
    # Should sum to 1 (probability distribution)
    assert abs(D[0].sum() - 1.0) < 1e-6


def test_build_generative_model():
    """Test complete generative model construction."""
    position_space, _ = create_state_space(n_position_bins=10)
    A, B, D = build_generative_model(position_space, noise_sd=10.0, hazard_rate=0.1)
    
    # Check all are pymdp object arrays
    assert isinstance(A, np.ndarray)
    assert isinstance(B, np.ndarray)
    assert isinstance(D, np.ndarray)
    
    # Check shapes
    assert len(A) == 1
    assert len(B) == 1
    assert len(D) == 1
    
    # Check dimensions match
    n_bins = position_space.n_bins
    assert A[0].shape == (n_bins, n_bins)
    assert B[0].shape == (n_bins, n_bins, n_bins)
    assert D[0].shape == (n_bins,)

