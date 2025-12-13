"""
Global configuration parameters for all experiments.
"""

# =============================================================================
# STATE SPACE
# =============================================================================
# The agent tracks 3 hidden state factors:
# 1. CONTEXT: Volatility regime (volatile vs stable) - HIDDEN STATE
#    - 'volatile': Better arm switches frequently
#    - 'stable': Better arm stays fixed for long periods
#    - Agent infers context from reward probability patterns
#
# 2. BETTER_ARM: Which arm currently has higher reward probability
#    - 'left_better': Left arm is currently better
#    - 'right_better': Right arm is currently better
#    - This is what hints reveal and what the agent should learn
#
# 3. CHOICE: The agent's current action state in the trial
#    - Tracks what action was taken (start, hint, left, right)

STATE_CONTEXTS = ['volatile', 'stable']
STATE_BETTER_ARM = ['left_better', 'right_better']
STATE_CHOICES = ['start', 'hint', 'left', 'right']

ACTION_CONTEXTS = ['rest'] # No direct control over regime context
ACTION_BETTER_ARM = ['rest']  # No direct control over which arm is better
ACTION_CHOICES = ['act_start', 'act_hint', 'act_left', 'act_right']

OBSERVATION_HINTS = ['null', 'observe_left_hint', 'observe_right_hint']
OBSERVATION_REWARDS = ['null', 'observe_loss', 'observe_reward']
OBSERVATION_CHOICES = ['observe_start', 'observe_hint', 'observe_left', 'observe_right']

# Derived dimensions - 3 state factors, 3 observation modalities
NUM_STATES = [len(STATE_CONTEXTS), len(STATE_BETTER_ARM), len(STATE_CHOICES)]
NUM_ACTIONS = [len(ACTION_CONTEXTS), len(ACTION_BETTER_ARM), len(ACTION_CHOICES)]
NUM_OBS = [len(OBSERVATION_HINTS), len(OBSERVATION_REWARDS), len(OBSERVATION_CHOICES)]

NUM_FACTORS = len(NUM_STATES)
NUM_MODALITIES = len(NUM_OBS)

# =============================================================================
# ENVIRONMENT PARAMETERS (GENERATIVE PROCESS)
# =============================================================================
PROBABILITY_HINT = 0.85  # Hint accuracy (same for both volatile and stable contexts)

# Reward probabilities differ by context to create different optimal strategies
# Volatile context: moderate discrimination (70% vs 30%) + frequent switches
VOLATILE_REWARD_BETTER = 0.70  # Better arm in volatile context
VOLATILE_REWARD_WORSE = 0.30   # Worse arm in volatile context
VOLATILE_SWITCH_INTERVAL = 10  # Arms switch every 10 trials

# Stable context: strong discrimination (90% vs 10%) with no switches
STABLE_REWARD_BETTER = 0.90    # Better arm in stable context
STABLE_REWARD_WORSE = 0.10     # Worse arm in stable context

# =============================================================================
# DEFAULT EXPERIMENT PARAMETERS
# =============================================================================

# Number of bandit trials per experimental run
DEFAULT_TRIALS = 400

# Trial indices where the context regime switches between volatile and stable.
# This controls when the environment changes from one volatility regime to another.
# Note: This is DIFFERENT from arm switches (which arm is better):
#   - Context switches: volatile ↔ stable regime changes (controlled here)
#   - Arm switches: left_better ↔ right_better changes (controlled by VOLATILE_SWITCH_INTERVAL)
DEFAULT_REVERSAL_SCHEDULE = [i for i in range(40, 400, 40)]

# Agent's generative model parameter: probability per trial that context switches.
# Used in the B matrix (transition probabilities) to encode the agent's beliefs about
# how frequently the volatility regime changes. This is the agent's model of the world,
# NOT the actual environment dynamics (which use deterministic DEFAULT_REVERSAL_SCHEDULE).
# Value of 0.05 means agent believes 5% chance of context switch per trial.
DEFAULT_CONTEXT_VOLATILITY = 0.05


# Plotting parameters
ROLLING_WINDOW = 7
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIG_DPI = 300

# Model default parameters
M1_DEFAULTS = {
    # OBSERVATION_REWARDS = ['null', 'observe_loss', 'observe_reward']
    'C_reward_logits': [0.0, -5.0, 5.0],
    'gamma': 2.5
}

M2_DEFAULTS = {
    # OBSERVATION_REWARDS = ['null', 'observe_loss', 'observe_reward']
    'C_reward_logits': [0.0, -5.0, 5.0],
    'gamma_base': 2.5,
    'entropy_k': 1.0
}

M3_DEFAULTS = {
    'profiles': [
        {   
            # PROFILE 0: For VOLATILE context - Information-seeking, exploratory
            # Strategy: Arms switch frequently, so KEEP CHECKING HINTS and EXPLORE
            # OBSERVATION_REWARDS = ['null', 'observe_loss', 'observe_reward']
            'phi_logits': [0.0, -5.0, 5.0],  # Strong outcome preferences (same as others)
            # ACTION_CHOICE = ['act_start', 'act_hint', 'act_left', 'act_right']
            'xi_logits': [0.0, 3.0, 0.0, 0.0],  # HIGH preference for hints (3.0), neutral on arms
            'gamma': 2.0  # LOWER precision (more exploratory, less committed)
        },
        {
            # PROFILE 1: For STABLE context - Exploitative, decisive
            # Strategy: Arms stay fixed, so CHECK HINT ONCE then EXPLOIT with confidence
            # OBSERVATION_REWARDS = ['null', 'observe_loss', 'observe_reward']
            'phi_logits': [0.0, -5.0, 5.0],  # Strong outcome preferences (same as others)
            # ACTION_CHOICE = ['act_start', 'act_hint', 'act_left', 'act_right']
            'xi_logits': [0.0, 0.5, 0.0, 0.0],  # LOW preference for hints (0.5), mostly rely on direct sampling
            'gamma': 4.0  # HIGHER precision (more exploitative, commit to best arm)
        }
    ],
    'Z': [[1.0, 0.0],  # When context = volatile -> use Profile 0 (info-seeking)
          [0.0, 1.0]]  # When context = stable -> use Profile 1 (exploitative)
}
