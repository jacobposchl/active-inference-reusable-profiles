"""
State space definitions for the changepoint active inference model.

Defines discretized state spaces for helicopter position and volatility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Import constants
try:
    from ..pipeline.constants import SCREEN_MIN, SCREEN_MAX, HAZARD_RATE
except ImportError:
    from behavioral_data.pipeline.constants import SCREEN_MIN, SCREEN_MAX, HAZARD_RATE


@dataclass
class PositionStateSpace:
    """Discretized position state space for helicopter location."""
    
    n_bins: int
    min_pos: float
    max_pos: float
    bin_centers: np.ndarray
    bin_edges: np.ndarray
    
    @classmethod
    def create(cls, n_bins: int = 30, min_pos: float = SCREEN_MIN, max_pos: float = SCREEN_MAX) -> PositionStateSpace:
        """Create a position state space with evenly spaced bins."""
        bin_edges = np.linspace(min_pos, max_pos, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return cls(
            n_bins=n_bins,
            min_pos=min_pos,
            max_pos=max_pos,
            bin_centers=bin_centers,
            bin_edges=bin_edges,
        )
    
    def position_to_bin(self, position: float) -> int:
        """Convert continuous position to bin index."""
        position = np.clip(position, self.min_pos, self.max_pos)
        idx = np.searchsorted(self.bin_edges, position, side="right") - 1
        return int(np.clip(idx, 0, self.n_bins - 1))
    
    def bin_to_position(self, bin_idx: int) -> float:
        """Get center position of a bin."""
        return float(self.bin_centers[bin_idx])
    
    def bin_to_continuous_range(self, bin_idx: int) -> Tuple[float, float]:
        """Get continuous range [min, max) for a bin."""
        return (float(self.bin_edges[bin_idx]), float(self.bin_edges[bin_idx + 1]))


@dataclass
class VolatilityStateSpace:
    """Volatility state space (stable vs volatile contexts)."""
    
    n_states: int = 2
    state_labels: Tuple[str, ...] = ("stable", "volatile")
    
    def state_to_idx(self, state: str) -> int:
        """Convert state label to index."""
        return self.state_labels.index(state)
    
    def idx_to_state(self, idx: int) -> str:
        """Convert index to state label."""
        return self.state_labels[idx]


def create_state_space(
    n_position_bins: int = 30,
    min_pos: float = SCREEN_MIN,
    max_pos: float = SCREEN_MAX,
) -> Tuple[PositionStateSpace, VolatilityStateSpace]:
    """
    Create state spaces for position and volatility.
    
    Parameters
    ----------
    n_position_bins : int
        Number of bins for discretizing helicopter position
    min_pos : float
        Minimum position value
    max_pos : float
        Maximum position value
    
    Returns
    -------
    position_space : PositionStateSpace
        Position state space
    volatility_space : VolatilityStateSpace
        Volatility state space
    """
    position_space = PositionStateSpace.create(n_bins=n_position_bins, min_pos=min_pos, max_pos=max_pos)
    volatility_space = VolatilityStateSpace()
    return position_space, volatility_space

