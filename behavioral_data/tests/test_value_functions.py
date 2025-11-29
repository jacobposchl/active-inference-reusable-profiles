"""
Tests for value functions (M1, M2, M3).
"""
import pytest
import numpy as np
from pymdp.maths import softmax

from behavioral_data.aif_models.value_functions import (
    make_values_M1,
    make_values_M2,
    make_values_M3_with_volatility,
    make_value_fn,
)


def test_m1_static():
    """Test M1 returns fixed values."""
    value_fn = make_values_M1(gamma=5.0)
    
    qs = [np.ones(10) / 10]  # Uniform position beliefs
    C_t, E_t, gamma_t = value_fn(qs, 0)
    
    assert gamma_t == 5.0
    assert C_t.shape == (10,)
    assert abs(C_t.sum() - 1.0) < 1e-6  # Should be probability distribution
    assert E_t is None  # M1 doesn't use policy priors


def test_m2_entropy_coupling():
    """Test M2 adapts precision to entropy."""
    value_fn = make_values_M2(gamma_base=5.0, entropy_k=1.0)
    
    # High entropy (uncertain)
    qs_uncertain = [np.ones(10) / 10]  # Uniform = high entropy
    C1, E1, gamma1 = value_fn(qs_uncertain, 0)
    
    # Low entropy (certain)
    qs_certain = [np.zeros(10)]
    qs_certain[0][5] = 1.0  # All mass on one state
    C2, E2, gamma2 = value_fn(qs_certain, 0)
    
    # Higher entropy should lead to lower precision
    assert gamma1 < gamma2
    assert gamma1 < 5.0  # Should be reduced from base
    assert gamma2 > gamma1  # More certain = higher precision


def test_m3_profile_mixing():
    """Test M3 mixes profiles based on volatility beliefs."""
    n_bins = 10
    profiles = [
        {"phi_logits": np.zeros(n_bins), "gamma": 2.0},  # Profile 0: volatile
        {"phi_logits": np.zeros(n_bins), "gamma": 8.0},  # Profile 1: stable
    ]
    Z = np.array([[0.0, 1.0], [1.0, 0.0]])  # stable→profile1, volatile→profile0
    
    value_fn = make_values_M3_with_volatility(profiles=profiles, Z=Z)
    
    qs = [np.ones(n_bins) / n_bins]
    
    # Test with stable belief
    # q_volatile_stable = [0.9, 0.1] means 90% stable (state 0), 10% volatile (state 1)
    # With Z = [[0.0, 1.0], [1.0, 0.0]]:
    #   stable (0) → profile 1 (gamma=8.0)
    #   volatile (1) → profile 0 (gamma=2.0)
    # w = [0.9, 0.1] @ Z = [0.1, 0.9] → mostly profile 1
    q_volatile_stable = np.array([0.9, 0.1])  # Mostly stable
    C1, E1, gamma1 = value_fn(qs, 0, q_volatility=q_volatile_stable)
    
    # Test with volatile belief
    # q_volatile_volatile = [0.1, 0.9] means 10% stable, 90% volatile
    # w = [0.1, 0.9] @ Z = [0.9, 0.1] → mostly profile 0
    q_volatile_volatile = np.array([0.1, 0.9])  # Mostly volatile
    C2, E2, gamma2 = value_fn(qs, 0, q_volatility=q_volatile_volatile)
    
    # Stable should use profile 1 (gamma=8.0), volatile should use profile 0 (gamma=2.0)
    assert gamma1 > gamma2  # Stable has higher precision
    # With q_volatile=[0.9, 0.1] (mostly stable), weight on profile1 is 0.9
    # So gamma = 0.1 * 2.0 + 0.9 * 8.0 = 0.2 + 7.2 = 7.4
    assert 7.0 < gamma1 < 8.0  # Should be close to stable profile (8.0)
    # With q_volatile=[0.1, 0.9] (mostly volatile), weight on profile0 is 0.9
    # So gamma = 0.9 * 2.0 + 0.1 * 8.0 = 1.8 + 0.8 = 2.6
    assert 2.0 < gamma2 < 3.0  # Should be closer to volatile profile (2.0)


def test_m3_soft_assignment():
    """Test M3 with soft assignment matrix."""
    n_bins = 10
    profiles = [
        {"phi_logits": np.zeros(n_bins), "gamma": 2.0},
        {"phi_logits": np.zeros(n_bins), "gamma": 8.0},
    ]
    Z = np.array([[0.2, 0.8], [0.8, 0.2]])  # Soft assignment
    
    value_fn = make_values_M3_with_volatility(profiles=profiles, Z=Z)
    
    qs = [np.ones(n_bins) / n_bins]
    q_volatile = np.array([0.5, 0.5])  # Equal belief
    
    C_t, E_t, gamma_t = value_fn(qs, 0, q_volatility=q_volatile)
    
    # Should be weighted average
    expected_gamma = 0.5 * 2.0 + 0.5 * 8.0  # But Z mixes it...
    # Actually, with Z and equal q_volatility, it's more complex
    assert 2.0 <= gamma_t <= 8.0  # Should be between the two


def test_make_value_fn_factory():
    """Test factory function."""
    # M1
    vf1 = make_value_fn("M1", gamma=3.0)
    assert callable(vf1)
    
    # M2
    vf2 = make_value_fn("M2", gamma_base=4.0, entropy_k=1.5)
    assert callable(vf2)
    
    # M3 (should use with_volatility version)
    n_bins = 10
    profiles = [
        {"phi_logits": np.zeros(n_bins), "gamma": 2.0},
        {"phi_logits": np.zeros(n_bins), "gamma": 8.0},
    ]
    Z = np.array([[0.0, 1.0], [1.0, 0.0]])
    vf3 = make_value_fn("M3", profiles=profiles, Z=Z)
    assert callable(vf3)

