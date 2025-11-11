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
PROBABILITY_HINT = 0.99 # percent chance of correct hint (which arm is better) --- this create observation noise
PROBABILITY_REWARD = 0.85 # percent chance of reward when choosing better arm. --- this creates outcome noise

# Default experiment parameters
DEFAULT_TRIALS = 800
DEFAULT_REVERSAL_SCHEDULE = [i for i in range(40, 800, 40)]
# probability of arm context flipping
DEFAULT_CONTEXT_VOLATILITY = 0.05


# Plotting parameters
ROLLING_WINDOW = 7
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIG_DPI = 300

# Model default parameters
M1_DEFAULTS = {
    # OBSERVATION_REWARDS = ['null', 'observe_loss', 'observe_reward']
    'C_reward_logits': [0.0, -4.0, 2.0],
    'gamma': 1.2
}

M2_DEFAULTS = {
    # OBSERVATION_REWARDS = ['null', 'observe_loss', 'observe_reward']
    'C_reward_logits': [0.0, -4.0, 2.0],
    'gamma_base': 1.6,
    'entropy_k': 1.0
}

M3_DEFAULTS = {
    'profiles': [
        {   
            # PROFILE 0: For left_better context - exploitative
            # OBSERVATION_REWARDS = ['null', 'observe_loss', 'observe_reward']
            'phi_logits': [0.0, -4.0, 2.5], # Strong outcome preferences
            # ACTION_CHOICE = ['act_start', 'act_hint', 'act_left', 'act_right']
            'xi_logits': [0.0, 0.0, 0.0, 0.0], # Neutral on actions
            'gamma': 2.5 # Moderate-high precision
        },
        {
            # PROFILE 1: For right_better context - exploitative
            # OBSERVATION_REWARDS = ['null', 'observe_loss', 'observe_reward']
            'phi_logits': [0.0, -4.0, 2.5], # Strong outcome preferences
            # ACTION_CHOICE = ['act_start', 'act_hint', 'act_left', 'act_right']
            'xi_logits': [0.0, 0.0, 0.0, 0.0], # Neutral on actions
            'gamma': 2.5 # Moderate-high precision
        }
    ],
    'Z': [[1.0, 0.0], # When context = left_better -> use Profile 0
          [0.0, 1.0]] # When context = right_better -> use Profile 1
}
