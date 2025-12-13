"""
Two-armed bandit environment with context-dependent volatility.
"""

import numpy as np
from pymdp import utils

class TwoArmedBandit:
    """
    Two-armed bandit environment with context-dependent arm dynamics.
    
    The hidden context determines the volatility regime:
    - 'volatile': Which arm is better switches every N trials (micro-reversals)
    - 'stable': Which arm is better stays fixed
    
    Context switches occur at specified reversal_schedule trials.
    """
    
    def __init__(self, context=None, probability_hint=0.85, 
                 volatile_reward_better=0.70, volatile_reward_worse=0.30,
                 stable_reward_better=0.90, stable_reward_worse=0.10,
                 volatile_switch_interval=10,
                 reversal_schedule=None,
                 observation_hints=None, observation_rewards=None):
        """
        Initialize environment.
        
        Parameters:
        -----------
        context : str or None
            Initial context ('volatile' or 'stable'). 
            If None, randomly sampled.
        probability_hint : float
            Accuracy of hint observations (0.0 to 1.0)
        volatile_reward_better : float
            Reward probability for better arm in volatile context
        volatile_reward_worse : float
            Reward probability for worse arm in volatile context
        stable_reward_better : float
            Reward probability for better arm in stable context
        stable_reward_worse : float
            Reward probability for worse arm in stable context
        volatile_switch_interval : int
            Number of trials between arm switches in volatile context
        reversal_schedule : list or None
            Trial indices where CONTEXT reverses (volatile <-> stable)
        observation_hints : list or None
            Hint observation labels
        observation_rewards : list or None
            Reward observation labels
        """

        self.context_names = ['volatile', 'stable']
        
        # Sample context if not provided
        if context is None:
            self.context = self.context_names[utils.sample(np.array([0.5, 0.5]))]
        else:
            assert context in self.context_names, f"Context must be in {self.context_names}, got {context}"
            self.context = context
        
        self.probability_hint = probability_hint
        
        # Store reward probabilities for each context
        self.volatile_reward_better = volatile_reward_better
        self.volatile_reward_worse = volatile_reward_worse
        self.stable_reward_better = stable_reward_better
        self.stable_reward_worse = stable_reward_worse
        
        self.volatile_switch_interval = volatile_switch_interval
        
        # Set observation labels (defaults to config if not provided)
        if observation_hints is None:
            from config.experiment_config import OBSERVATION_HINTS
            observation_hints = OBSERVATION_HINTS
        if observation_rewards is None:
            from config.experiment_config import OBSERVATION_REWARDS
            observation_rewards = OBSERVATION_REWARDS
            
        self.observation_hints = observation_hints
        self.observation_rewards = observation_rewards
        
        self.reversal_schedule = reversal_schedule or []
        self.trial_count = 0
        
        # Track which arm is currently better (starts random)
        self.current_better_arm = 'left' if np.random.rand() < 0.5 else 'right'
        
        # Track when we last switched in volatile context
        self.trials_since_volatile_switch = 0
    
    def _check_volatility_switch(self):
        """Check if arms should switch in volatile context."""
        if self.context == 'volatile':
            self.trials_since_volatile_switch += 1
            if self.trials_since_volatile_switch >= self.volatile_switch_interval:
                # Switch which arm is better
                self.current_better_arm = 'right' if self.current_better_arm == 'left' else 'left'
                self.trials_since_volatile_switch = 0
                return True
        return False
    
    def _check_context_reversal(self):
        """Check if context should reverse (volatile <-> stable)."""
        if self.trial_count in self.reversal_schedule:
            # Switch context
            self.context = 'stable' if self.context == 'volatile' else 'volatile'
            # Reset volatile switch counter
            self.trials_since_volatile_switch = 0
            return True
        return False
    
    def _get_reward_probs(self):
        """Get current reward probabilities based on context and which arm is better."""
        if self.context == 'volatile':
            p_better = self.volatile_reward_better
            p_worse = self.volatile_reward_worse
        else:  # stable
            p_better = self.stable_reward_better
            p_worse = self.stable_reward_worse
        
        if self.current_better_arm == 'left':
            return {'left': p_better, 'right': p_worse}
        else:
            return {'left': p_worse, 'right': p_better}
    
    def step(self, action):
        """
        Execute action and return observations.
        
        Parameters:
        -----------
        action : str
            Action label (e.g., 'act_left', 'act_hint')
        
        Returns:
        --------
        observations : list of str
            [hint_obs, reward_obs, choice_obs]
            Context is now hidden - agent must infer it from reward patterns.
        """
        
        self.trial_count += 1
        
        # Check for context reversals (volatile <-> stable)
        self._check_context_reversal()
        
        # Check for volatility-driven arm switches (only in volatile context)
        self._check_volatility_switch()
        
        # Generate observations based on action
        reward_probs = self._get_reward_probs()
        
        # Are we just starting? 
        if action == "act_start":
            observed_hint = "null"
            observed_reward = "null"
            observed_choice = "observe_start"
        
        # Did we ask for a hint? 
        elif action == "act_hint":
            # Hint tells which arm is better
            if self.current_better_arm == "left":
                observed_hint = self.observation_hints[
                    utils.sample(np.array([0.0, self.probability_hint, 1.0 - self.probability_hint]))
                ]
            else:
                observed_hint = self.observation_hints[
                    utils.sample(np.array([0.0, 1.0 - self.probability_hint, self.probability_hint]))
                ]
            observed_reward = "null"
            observed_choice = "observe_hint"
            
        # Did we choose left? 
        elif action == "act_left":
            observed_hint = "null"
            # Sample reward based on left arm's current probability
            p_reward_left = reward_probs['left']
            observed_reward = self.observation_rewards[
                utils.sample(np.array([0.0, 1.0 - p_reward_left, p_reward_left]))
            ]
            observed_choice = "observe_left"

        # Did we choose right?  
        elif action == "act_right":
            observed_hint = "null"
            # Sample reward based on right arm's current probability
            p_reward_right = reward_probs['right']
            observed_reward = self.observation_rewards[
                utils.sample(np.array([0.0, 1.0 - p_reward_right, p_reward_right]))
            ]
            observed_choice = "observe_right"
        
        else:
            raise ValueError(f"Unknown action: {action}")
        
        observations = [observed_hint, observed_reward, observed_choice]
        return observations
