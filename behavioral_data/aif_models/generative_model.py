"""
Generative model construction for the changepoint task using pymdp.

Builds A (likelihood), B (transition), and D (prior) matrices in pymdp format.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm
from pymdp import utils

from .state_space import PositionStateSpace, VolatilityStateSpace

# Import constants
try:
    from ..pipeline.constants import HAZARD_RATE
except ImportError:
    from behavioral_data.pipeline.constants import HAZARD_RATE


def build_A(
    position_space: PositionStateSpace,
    noise_sd: float,
) -> np.ndarray:
    """
    Build likelihood matrix A: p(observed_bag_position | helicopter_position).
    
    Uses pymdp format: A is an object array with one matrix per observation modality.
    For changepoint task, we have one modality: bag position observations.
    
    Parameters
    ----------
    position_space : PositionStateSpace
        Position state space
    noise_sd : float
        Standard deviation of observation noise
    
    Returns
    -------
    A : object array
        Likelihood matrices [A_observation]
        A[0] shape: (n_bins, n_bins) where A[0][obs_bin, heli_bin] = p(obs_bin | heli_bin)
    """
    n_bins = position_space.n_bins
    
    # Create object array for pymdp (one modality)
    A = utils.obj_array(1)
    
    # A[0]: p(observed_bag_bin | helicopter_position_bin)
    A_obs = np.zeros((n_bins, n_bins))
    
    # For each helicopter position bin
    for heli_bin in range(n_bins):
        heli_center = position_space.bin_to_position(heli_bin)
        
        # For each possible observation bin
        for obs_bin in range(n_bins):
            # Compute probability mass in observation bin given helicopter position
            # Use bin edges for more accurate integration
            obs_min, obs_max = position_space.bin_to_continuous_range(obs_bin)
            
            # Cumulative probability at edges
            prob_min = norm.cdf(obs_min, loc=heli_center, scale=noise_sd)
            prob_max = norm.cdf(obs_max, loc=heli_center, scale=noise_sd)
            
            A_obs[obs_bin, heli_bin] = prob_max - prob_min
    
    # Normalize columns (each column should sum to 1)
    A_obs = A_obs / (A_obs.sum(axis=0, keepdims=True) + 1e-12)
    
    A[0] = A_obs
    
    return A


def build_B(
    position_space: PositionStateSpace,
    hazard_rate: float = HAZARD_RATE,
) -> np.ndarray:
    """
    Build transition matrix B: p(helicopter_position_t+1 | helicopter_position_t, action_t).
    
    Uses pymdp format: B is an object array with one matrix per state factor.
    For changepoint task, we have one state factor: position.
    
    Implements changepoint dynamics:
    - With probability (1 - hazard_rate): stay in same position
    - With probability hazard_rate: jump to any position (uniform)
    - Actions don't affect transitions (helicopter moves independently)
    
    Parameters
    ----------
    position_space : PositionStateSpace
        Position state space
    hazard_rate : float
        Probability of changepoint (helicopter jumps)
    
    Returns
    -------
    B : object array
        Transition matrices [B_position]
        B[0] shape: (n_bins, n_bins, n_actions)
    """
    n_bins = position_space.n_bins
    n_actions = n_bins  # Actions are bucket position predictions (same as position bins)
    
    # Create object array for pymdp (one state factor)
    B = utils.obj_array(1)
    
    # B[0]: p(position_t+1 | position_t, action_t)
    B_pos = np.zeros((n_bins, n_bins, n_actions))
    
    # For all actions, same transition dynamics (actions don't affect helicopter)
    for a in range(n_actions):
        # Stay in place with probability (1 - hazard_rate)
        B_pos[:, :, a] = np.eye(n_bins) * (1.0 - hazard_rate)
        # Jump uniformly with probability hazard_rate
        B_pos[:, :, a] += (hazard_rate / n_bins)
    
    # Normalize columns
    B_pos = B_pos / (B_pos.sum(axis=0, keepdims=True) + 1e-12)
    
    B[0] = B_pos
    
    return B


def build_D(
    position_space: PositionStateSpace,
) -> np.ndarray:
    """
    Build prior over initial states D: p(s_0).
    
    Uses pymdp format: D is an object array with one vector per state factor.
    For changepoint task, we have one state factor: position.
    
    Parameters
    ----------
    position_space : PositionStateSpace
        Position state space
    
    Returns
    -------
    D : object array
        Prior distributions [D_position]
        D[0] shape: (n_bins,)
    """
    n_bins = position_space.n_bins
    
    # Create object array for pymdp (one state factor)
    D = utils.obj_array(1)
    
    # D[0]: Uniform prior over initial helicopter position
    D_position = np.ones(n_bins) / n_bins
    
    D[0] = D_position
    
    return D


def build_generative_model(
    position_space: PositionStateSpace,
    noise_sd: float,
    hazard_rate: float = HAZARD_RATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build complete generative model (A, B, D matrices) in pymdp format.
    
    Parameters
    ----------
    position_space : PositionStateSpace
        Position state space
    noise_sd : float
        Standard deviation of observation noise
    hazard_rate : float
        Probability of changepoint
    
    Returns
    -------
    A : object array
        Likelihood matrices p(observation | position)
    B : object array
        Transition matrices p(position_t+1 | position_t, action_t)
    D : object array
        Prior beliefs p(position_0)
    """
    A = build_A(position_space, noise_sd)
    B = build_B(position_space, hazard_rate)
    D = build_D(position_space)
    
    return A, B, D
