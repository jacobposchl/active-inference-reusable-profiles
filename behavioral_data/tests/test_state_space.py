"""
Tests for state space definitions.
"""
import pytest
import numpy as np

from behavioral_data.aif_models.state_space import (
    PositionStateSpace,
    VolatilityStateSpace,
    create_state_space,
)
from behavioral_data.pipeline.constants import SCREEN_MIN, SCREEN_MAX


def test_position_state_space_creation():
    """Test creating a position state space."""
    space = PositionStateSpace.create(n_bins=30)
    
    assert space.n_bins == 30
    assert space.min_pos == SCREEN_MIN
    assert space.max_pos == SCREEN_MAX
    assert len(space.bin_centers) == 30
    assert len(space.bin_edges) == 31


def test_position_to_bin():
    """Test converting continuous positions to bins."""
    space = PositionStateSpace.create(n_bins=10, min_pos=0.0, max_pos=100.0)
    
    # Test exact bin centers
    assert space.position_to_bin(5.0) == 0  # First bin center
    assert space.position_to_bin(95.0) == 9  # Last bin center
    
    # Test clipping
    assert space.position_to_bin(-10.0) == 0
    assert space.position_to_bin(200.0) == 9


def test_bin_to_position():
    """Test converting bin indices to positions."""
    space = PositionStateSpace.create(n_bins=10, min_pos=0.0, max_pos=100.0)
    
    assert space.bin_to_position(0) == 5.0  # First bin center
    assert space.bin_to_position(9) == 95.0  # Last bin center


def test_volatility_state_space():
    """Test volatility state space."""
    space = VolatilityStateSpace()
    
    assert space.n_states == 2
    assert space.state_to_idx("stable") == 0
    assert space.state_to_idx("volatile") == 1
    assert space.idx_to_state(0) == "stable"
    assert space.idx_to_state(1) == "volatile"


def test_create_state_space():
    """Test factory function."""
    pos_space, vol_space = create_state_space(n_position_bins=20)
    
    assert isinstance(pos_space, PositionStateSpace)
    assert isinstance(vol_space, VolatilityStateSpace)
    assert pos_space.n_bins == 20
    assert vol_space.n_states == 2

