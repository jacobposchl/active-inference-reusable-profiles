"""
Two-armed bandit environment with reversal learning.
"""

import numpy as np
from pymdp import utils

class TwoArmedBandit:
    """
    Two-armed bandit environment with context reversals.
    
    The environment has a hidden context (left_better or right_better) that
    determines which arm has higher reward probability. Context can reverse
    at specified trial indices.
    """
    
    def __init__(self, context=None, probability_hint=0.7, 
                 probability_reward=0.8, reversal_schedule=None,
                 observation_hints=None, observation_rewards=None):
        """
        Initialize environment.
        
        Parameters:
        -----------
        context : str or None
            Initial context ('left_better' or 'right_better'). 
            If None, randomly sampled.
        probability_hint : float
            Accuracy of hint observations (0.0 to 1.0)
        probability_reward : float
            Win rate when pulling better arm (0.0 to 1.0)
        reversal_schedule : list or None
            Trial indices where context reverses
        observation_hints : list or None
            Hint observation labels
        observation_rewards : list or None
            Reward observation labels
        """

        self.context_names = ['left_better', 'right_better']
        
        # Sample context if not provided
        if context is None:
            self.context = self.context_names[utils.sample(np.array([0.5, 0.5]))]
        else:
            assert context in self.context_names
            self.context = context
        
        self.probability_hint = probability_hint
        self.probability_reward = probability_reward
        
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
    
    def step(self, action):
        """
        Execute action and return observations.
        -----------
        Parameters:
        -----------
        action : str
            Action label (e.g., 'act_left', 'act_hint')
        --------
        Returns:
        --------
        observations : list of str
            [hint_obs, reward_obs, choice_obs]
        """
        # Check for reversals
        if self.trial_count in self.reversal_schedule:
            self.context = ('right_better' if self.context == 'left_better' else 'left_better')
        
        self.trial_count += 1
        
        # Generate observations based on action

        # Are we just starting? 
        if action == "act_start":
            observed_hint = "null"
            observed_reward = "null"
            observed_choice = "observe_start"
        
        # Did we ask for a hint? 
        elif action == "act_hint":
            # Sample hint based on context and hint probability
            if self.context == "left_better":
                observed_hint = self.observation_hints[
                    utils.sample(np.array([0.0, self.probability_hint, 1.0 - self.probability_hint]))
                ]
            else:  # right_better
                observed_hint = self.observation_hints[
                    utils.sample(np.array([0.0, 1.0 - self.probability_hint, self.probability_hint]))
                ]
            observed_reward = "null"
            observed_choice = "observe_hint"
            
        # Did we choose left? 
        elif action == "act_left":
            observed_hint = "null"
            # Sample reward based on context
            if self.context == "left_better":
                observed_reward = self.observation_rewards[
                    utils.sample(np.array([0.0, 1.0 - self.probability_reward, 
                                          self.probability_reward]))
                ]
            else:
                observed_reward = self.observation_rewards[
                    utils.sample(np.array([0.0, self.probability_reward, 
                                          1.0 - self.probability_reward]))
                ]
            observed_choice = "observe_left"

        # Did we choose right?  
        elif action == "act_right":
            observed_hint = "null"
            # Sample reward based on context
            if self.context == "left_better":
                observed_reward = self.observation_rewards[
                    utils.sample(np.array([0.0, self.probability_reward, 
                                          1.0 - self.probability_reward]))
                ]
            else:
                observed_reward = self.observation_rewards[
                    utils.sample(np.array([0.0, 1.0 - self.probability_reward, 
                                          self.probability_reward]))
                ]
            observed_choice = "observe_right"
        
        observations = [observed_hint, observed_reward, observed_choice]
        return observations
