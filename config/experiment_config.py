"""
Global configuration parameters for all experiments.
"""

# Task dimensionalities
STATE_CONTEXTS = ['left_better', 'right_better']
STATE_CHOICES = ['start', 'hint', 'left', 'right']

ACTION_CONTEXTS = ['rest']
ACTION_CHOICES = ['act_start', 'act_hint', 'act_left', 'act_right']

OBSERVATION_HINTS = ['null', 'observe_left_hint', 'observe_right_hint']
OBSERVATION_REWARDS = ['null', 'observe_loss', 'observe_reward']
OBSERVATION_CHOICES = ['observe_start', 'observe_hint', 'observe_left', 'observe_right']

# Derived dimensions
NUM_STATES = [len(STATE_CONTEXTS), len(STATE_CHOICES)]
NUM_ACTIONS = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
NUM_OBS = [len(OBSERVATION_HINTS), len(OBSERVATION_REWARDS), len(OBSERVATION_CHOICES)]

NUM_FACTORS = len(NUM_STATES)
NUM_MODALITIES = len(NUM_OBS)

# Environment parameters (generative process)
PROBABILITY_HINT = 0.7
PROBABILITY_REWARD = 0.8

# Default experiment parameters
DEFAULT_TRIALS = 200
DEFAULT_REVERSAL_SCHEDULE = [30, 60, 90, 120, 150, 180]
DEFAULT_CONTEXT_VOLATILITY = 0.05

# Plotting parameters
ROLLING_WINDOW = 7
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIG_DPI = 300

# Model default parameters
M1_DEFAULTS = {
    'C_reward_logits': [0.0, -4.0, 2.0],
    'gamma': 1.2
}

M2_DEFAULTS = {
    'C_reward_logits': [0.0, -4.0, 2.0],
    'gamma_base': 1.6,
    'entropy_k': 1.0
}

M3_DEFAULTS = {
    'profiles': [
        {
            'phi_logits': [0.0, -2.0, 1.5],
            'xi_logits': [0.0, 0.5, 0.0, 0.0],
            'gamma': 0.6
        },
        {
            'phi_logits': [0.0, -8.0, 4.0],
            'xi_logits': [0.0, -1.0, 0.0, 0.0],
            'gamma': 3.0
        }
    ],
    'Z': [[0.8, 0.2],
          [0.2, 0.8]]
}
