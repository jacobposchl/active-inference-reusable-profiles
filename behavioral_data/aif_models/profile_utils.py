"""
Utilities for creating dynamic outcome preferences (phi_logits) for M3 profiles.
"""
from __future__ import annotations

import numpy as np

from .state_space import PositionStateSpace


def create_exploit_phi_logits(
    qs: list,
    profile_idx: int,
    position_space: PositionStateSpace,
    preference_width: float = 20.0,
) -> np.ndarray:
    """
    Create peaked outcome preferences for exploit profile centered on current belief mean.
    
    Parameters
    ----------
    qs : list of arrays
        Beliefs over state factors [q_position]
    profile_idx : int
        Profile index (0 = explore, 1 = exploit) - only exploit (1) uses this
    position_space : PositionStateSpace
        Position state space for bin information
    preference_width : float
        Width of preference peak (standard deviation of Gaussian)
    
    Returns
    -------
    phi_logits : np.ndarray
        Outcome preference logits (peaked around belief mean for exploit, flat for explore)
    """
    if profile_idx == 0:
        # Explore profile: flat preferences (uniform)
        n_bins = position_space.n_bins
        return np.zeros(n_bins)
    
    elif profile_idx == 1:
        # Exploit profile: peaked preferences around current belief mean
        qs_position = qs[0] if len(qs) > 0 else np.ones(position_space.n_bins) / position_space.n_bins
        
        # Compute current belief mean
        belief_mean = np.sum(qs_position * position_space.bin_centers)
        
        # Create Gaussian preferences centered on belief mean
        n_bins = position_space.n_bins
        phi_logits = np.zeros(n_bins)
        
        for i in range(n_bins):
            bin_center = position_space.bin_to_position(i)
            # Gaussian logit: higher for bins closer to belief mean
            # Using negative squared distance (so closer = higher preference)
            phi_logits[i] = -0.5 * ((bin_center - belief_mean) / preference_width) ** 2
        
        return phi_logits
    
    else:
        raise ValueError(f"Invalid profile_idx: {profile_idx}, must be 0 or 1")


def create_dynamic_phi_generator(preference_width: float = 20.0) -> callable:
    """
    Create a function that generates phi_logits dynamically for M3 profiles.
    
    Parameters
    ----------
    preference_width : float
        Width of preference peak for exploit profile
    
    Returns
    -------
    generator : callable
        Function(qs, profile_idx, position_space) -> phi_logits
    """
    def generator(qs: list, profile_idx: int, position_space: PositionStateSpace) -> np.ndarray:
        return create_exploit_phi_logits(qs, profile_idx, position_space, preference_width)
    
    return generator

