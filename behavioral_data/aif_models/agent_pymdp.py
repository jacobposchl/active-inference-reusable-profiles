"""
Active inference agent for changepoint task using pymdp.

Follows the same pattern as AgentRunner in the bandit task.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from pymdp.agent import Agent
from pymdp import utils

from .state_space import PositionStateSpace, VolatilityStateSpace
from .generative_model import build_generative_model
from .value_functions import make_value_fn, make_values_M3_with_volatility
from typing import Callable, Optional

# Import constants
try:
    from ..pipeline.constants import HAZARD_RATE
except ImportError:
    from behavioral_data.pipeline.constants import HAZARD_RATE


@dataclass
class AgentConfig:
    """Configuration for changepoint agent."""
    
    n_position_bins: int = 30
    noise_sd: float = 10.0  # Will be overridden per run
    hazard_rate: float = HAZARD_RATE
    volatility_window_size: int = 10
    volatility_error_threshold: float = 20.0
    policy_len: int = 1  # For changepoint, we only need 1-step policies
    inference_horizon: int = 1
    outcome_mod_idx: int = 0  # Index of outcome observation modality


class ChangepointAgentPymdp:
    """
    Active inference agent for changepoint task using pymdp.
    
    Follows the same pattern as AgentRunner:
    - infer_states() for belief updating
    - value_fn() returns (C_t, E_t, gamma_t)
    - infer_policies() for policy inference (EFE)
    - sample_action() for action selection
    """
    
    def __init__(
        self,
        value_fn,
        config: AgentConfig,
        seed: Optional[int] = None,
    ):
        """
        Initialize agent.
        
        Parameters
        ----------
        value_fn : callable
            Value function that returns (C_t, E_t, gamma_t)
            For M3, should be created with make_values_M3_with_volatility
        config : AgentConfig
            Agent configuration
        seed : int, optional
            Random seed
        """
        self.value_fn = value_fn
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.outcome_mod_idx = config.outcome_mod_idx
        
        # Create state spaces
        self.position_space, self.volatility_space = self._create_state_spaces()
        
        # Store M3 profile configuration if using dynamic profiles
        self.m3_use_dynamic_profiles = False
        self.m3_profile_config = None  # Will store gamma values, Z matrix, etc.
        
        # Build generative model in pymdp format
        A, B, D = build_generative_model(
            self.position_space,
            self.config.noise_sd,
            self.config.hazard_rate,
        )
        
        # Initialize C vector (will be updated each trial)
        C0 = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
        
        # Create pymdp Agent
        # State factors: [position] (single factor)
        # Only position is controllable (we predict bucket position)
        self.agent = Agent(
            A=A,
            B=B,
            C=C0,
            D=D,
            policy_len=config.policy_len,
            inference_horizon=config.inference_horizon,
            control_fac_idx=[0],  # Position is controllable (we predict it)
            use_utility=True,
            use_states_info_gain=True,
            action_selection="stochastic",
            gamma=16.0  # Will be overridden by value function
        )
        
        # History tracking
        self.prediction_history: List[float] = []
        self.prediction_std_history: List[float] = []
        self.outcome_history: List[float] = []
        self.prediction_error_history: List[float] = []
        self.gamma_history: List[float] = []
        self.belief_mean_history: List[float] = []
        self.belief_entropy_history: List[float] = []
        self.gamma_t = None
    
    def _create_state_spaces(self) -> Tuple[PositionStateSpace, VolatilityStateSpace]:
        """Create state spaces from config."""
        from .state_space import create_state_space
        return create_state_space(n_position_bins=self.config.n_position_bins)
    
    def _continuous_to_observation_bin(self, continuous_position: float) -> int:
        """Convert continuous bag position to observation bin index."""
        return self.position_space.position_to_bin(continuous_position)
    
    def _action_bin_to_continuous(self, action_bin: int) -> float:
        """Convert action bin index to continuous bucket position."""
        return self.position_space.bin_to_position(action_bin)
    
    def _get_volatility_beliefs(self) -> np.ndarray:
        """Get current volatility beliefs (inferred from prediction errors)."""
        if len(self.prediction_error_history) == 0:
            return np.array([0.5, 0.5])
        
        # Simple heuristic: large recent errors → volatile
        recent_errors = np.array(self.prediction_error_history[-self.config.volatility_window_size:])
        mean_error = np.mean(np.abs(recent_errors))
        
        # Compute entropy of position beliefs (high entropy = uncertain = volatile)
        qs_position = self.agent.qs[0]
        entropy = -np.sum(qs_position * np.log(qs_position + 1e-12))
        max_entropy = np.log(len(qs_position))
        normalized_entropy = entropy / max_entropy
        
        # Combine error-based and entropy-based signals
        error_signal = min(mean_error / self.config.volatility_error_threshold, 1.0)
        entropy_signal = normalized_entropy
        
        volatile_prob = 0.5 * error_signal + 0.5 * entropy_signal
        volatile_prob = np.clip(volatile_prob, 0.0, 1.0)
        
        return np.array([1.0 - volatile_prob, volatile_prob])
    
    def predict(self) -> Tuple[float, float]:
        """
        Generate bucket position prediction using pymdp's active inference.
        
        Follows AgentRunner.step() pattern:
        1. Get current beliefs (already updated from last observation)
        2. Compute value function (C_t, E_t, gamma_t)
        3. Update agent parameters
        4. Infer policies
        5. Sample action
        
        Returns
        -------
        prediction_mean : float
            Predicted bucket position (mean)
        prediction_std : float
            Prediction uncertainty (standard deviation)
        """
        # Get current beliefs
        qs = self.agent.qs
        
        # Get volatility beliefs
        q_volatility = self._get_volatility_beliefs()
        
        # Compute value profile for this trial (pass full beliefs)
        # For M3, value_fn needs q_volatility as additional parameter
        # Check if value_fn accepts q_volatility parameter
        import inspect
        sig = inspect.signature(self.value_fn)
        
        # Check if value_fn accepts position_space (for dynamic profiles)
        if 'position_space' in sig.parameters:
            C_t, E_t, gamma_t = self.value_fn(
                qs, 
                len(self.prediction_history), 
                q_volatility=q_volatility,
                position_space=self.position_space
            )
        elif 'q_volatility' in sig.parameters:
            C_t, E_t, gamma_t = self.value_fn(qs, len(self.prediction_history), q_volatility=q_volatility)
        else:
            C_t, E_t, gamma_t = self.value_fn(qs, len(self.prediction_history))
        
        # Update agent parameters
        self.agent.C[self.outcome_mod_idx] = C_t
        
        if E_t is not None:
            if len(E_t) == len(self.agent.policies):
                self.agent.E = E_t
        
        self.gamma_t = float(gamma_t)
        self.agent.gamma = self.gamma_t
        
        # Infer policy posterior (this computes Expected Free Energy)
        q_pi, efe = self.agent.infer_policies()
        
        # Sample action (bucket position prediction)
        chosen_action_ids = self.agent.sample_action()
        action_bin = int(chosen_action_ids[0])
        pred_mean = self._action_bin_to_continuous(action_bin)
        
        # Compute prediction uncertainty from policy posterior
        policy_entropy = -np.sum(q_pi * np.log(q_pi + 1e-12))
        max_entropy = np.log(len(q_pi))
        normalized_entropy = policy_entropy / max_entropy
        
        # Prediction std scales with uncertainty
        qs_position = qs[0]
        belief_variance = np.sum(qs_position * (self.position_space.bin_centers - pred_mean) ** 2)
        # Base std from belief uncertainty, scaled by policy entropy
        pred_std = np.sqrt(belief_variance * (1.0 + normalized_entropy))
        pred_std = np.clip(pred_std, 1.0, 50.0)
        
        # Store history
        self.prediction_history.append(pred_mean)
        self.prediction_std_history.append(pred_std)
        self.gamma_history.append(gamma_t)
        
        # Store belief statistics
        belief_mean = np.sum(qs_position * self.position_space.bin_centers)
        belief_entropy = -np.sum(qs_position * np.log(qs_position + 1e-12))
        self.belief_mean_history.append(belief_mean)
        self.belief_entropy_history.append(belief_entropy)
        
        return pred_mean, pred_std
    
    def observe(self, observed_outcome: float):
        """
        Observe outcome and update beliefs using pymdp's inference.
        
        Parameters
        ----------
        observed_outcome : float
            Observed bag position
        """
        # Convert continuous observation to bin index
        obs_bin = self._continuous_to_observation_bin(observed_outcome)
        obs_ids = [obs_bin]
        
        # Update beliefs using pymdp's state inference
        self.agent.infer_states(obs_ids)
        
        # Update volatility beliefs based on prediction errors
        if len(self.prediction_history) > 0:
            last_pred = self.prediction_history[-1]
            prediction_error = observed_outcome - last_pred
            self.prediction_error_history.append(prediction_error)
        else:
            self.prediction_error_history.append(0.0)
        
        # Store outcome
        self.outcome_history.append(observed_outcome)
    
    def reset(self):
        """Reset agent to initial state."""
        self.agent.reset()
        self.prediction_history.clear()
        self.prediction_std_history.clear()
        self.outcome_history.clear()
        self.prediction_error_history.clear()
        self.gamma_history.clear()
        self.belief_mean_history.clear()
        self.belief_entropy_history.clear()
        self.gamma_t = None
    
    def get_belief_statistics(self) -> dict:
        """Get current belief statistics."""
        qs_position = self.agent.qs[0]
        mean = np.sum(qs_position * self.position_space.bin_centers)
        variance = np.sum(qs_position * (self.position_space.bin_centers - mean) ** 2)
        entropy = -np.sum(qs_position * np.log(qs_position + 1e-12))
        
        q_volatility = self._get_volatility_beliefs()
        
        return {
            "belief_mean": float(mean),
            "belief_variance": float(variance),
            "belief_entropy": float(entropy),
            "volatility_stable": float(q_volatility[0]),
            "volatility_volatile": float(q_volatility[1]),
        }
