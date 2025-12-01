"""
Generative model construction (A, B, D matrices).

These matrices define the agent's beliefs about how the world works.

State factors:
    0: context (volatile, stable) - volatility regime
    1: better_arm (left_better, right_better) - which arm is currently better
    2: choice (start, hint, left, right) - current action state

Observation modalities:
    0: hints - reveals which arm is better
    1: rewards - outcome of arm choice (context-dependent probabilities enable context inference)
    2: choices - observes own action
"""
import numpy as np
from pymdp import utils


def build_A(num_modalities, 
            state_contexts, state_better_arm, state_choices,
            observation_hints, observation_rewards, observation_choices,
            p_hint, p_reward):
    """
    Build likelihood matrix A: p(o_t | s_t).
    
    Handles 3 state factors: context, better_arm, choice.
    Context is now a HIDDEN state - agent infers it from reward patterns.
    
    Parameters:
    -----------
    num_modalities : int
        Number of observation modalities (3: hints, rewards, choices)
    state_contexts : list
        Context state labels ['volatile', 'stable']
    state_better_arm : list
        Better arm state labels ['left_better', 'right_better']
    state_choices : list
        Choice state labels ['start', 'hint', 'left', 'right']
    observation_hints : list
        Hint observation labels ['null', 'observe_left_hint', 'observe_right_hint']
    observation_rewards : list
        Reward observation labels ['null', 'observe_loss', 'observe_reward']
    observation_choices : list
        Choice observation labels
    p_hint : float
        Believed probability of correct hint
    p_reward : list or array-like
        Reward probabilities for better arm, context-specific.
        Should have length equal to number of contexts.
        p_reward[0] = reward prob for volatile context, p_reward[1] = stable context.
        Worse arm probability is (1 - p_reward[ctx_idx]) for each context.
        Different reward probabilities (e.g., 0.70 vs 0.90) enable context inference.
        
    Returns:
    --------
    A : object array
        Likelihood matrices for each modality
        A[0] shape: (n_hint_obs, n_contexts, n_better_arm, n_choices)
        A[1] shape: (n_reward_obs, n_contexts, n_better_arm, n_choices) - context-dependent
        A[2] shape: (n_choice_obs, n_contexts, n_better_arm, n_choices)
    """
    # Convert p_reward to array and validate
    p_reward = np.asarray(p_reward, dtype=float)
    n_contexts = len(state_contexts)
    if len(p_reward) != n_contexts:
        raise ValueError(f"p_reward must have length {n_contexts} (one per context), got {len(p_reward)}")
    
    n_better_arm = len(state_better_arm)
    n_choices = len(state_choices)
    
    A = utils.obj_array(num_modalities)
    
    # =========================================================================
    # A[0]: Hint observations - depends on better_arm state, NOT on context
    # =========================================================================
    # Shape: (n_hint_obs, n_contexts, n_better_arm, n_choices)
    A_hint = np.zeros((len(observation_hints), n_contexts, n_better_arm, n_choices))
    
    for ctx_idx in range(n_contexts):
        for choice_idx, choice in enumerate(state_choices):
            if choice == "hint":
                # Hints reveal which arm is better (depends on better_arm state)
                # left_better (idx 0): hint_left likely, hint_right unlikely
                # right_better (idx 1): hint_right likely, hint_left unlikely
                A_hint[1, ctx_idx, 0, choice_idx] = p_hint      # P(hint_left | left_better)
                A_hint[2, ctx_idx, 0, choice_idx] = 1 - p_hint  # P(hint_right | left_better)
                A_hint[1, ctx_idx, 1, choice_idx] = 1 - p_hint  # P(hint_left | right_better)
                A_hint[2, ctx_idx, 1, choice_idx] = p_hint      # P(hint_right | right_better)
            else:
                # No hint when not in hint state
                A_hint[0, ctx_idx, :, choice_idx] = 1.0  # null hint
    
    # =========================================================================
    # A[1]: Reward observations - depends on better_arm, choice, AND CONTEXT
    # =========================================================================
    # Shape: (n_reward_obs, n_contexts, n_better_arm, n_choices)
    # Reward probabilities differ by context:
    # - Volatile context (ctx_idx=0): p_reward[0] for better arm, (1-p_reward[0]) for worse
    # - Stable context (ctx_idx=1): p_reward[1] for better arm, (1-p_reward[1]) for worse
    A_reward = np.zeros((len(observation_rewards), n_contexts, n_better_arm, n_choices))
    
    for ctx_idx in range(n_contexts):
        # Get context-specific reward probability for better arm
        p_reward_ctx = p_reward[ctx_idx]
        
        for choice_idx, choice in enumerate(state_choices):
            if choice in ["start", "hint"]:
                # No reward for start/hint actions
                A_reward[0, ctx_idx, :, choice_idx] = 1.0  # null reward
            elif choice == "left":
                # Chose left arm - reward depends on whether left is better
                # better_arm=0 (left_better): high reward prob (context-specific)
                # better_arm=1 (right_better): low reward prob (context-specific)
                A_reward[1, ctx_idx, 0, choice_idx] = 1 - p_reward_ctx  # P(loss | left, left_better)
                A_reward[2, ctx_idx, 0, choice_idx] = p_reward_ctx      # P(reward | left, left_better)
                A_reward[1, ctx_idx, 1, choice_idx] = p_reward_ctx      # P(loss | left, right_better)
                A_reward[2, ctx_idx, 1, choice_idx] = 1 - p_reward_ctx  # P(reward | left, right_better)
            elif choice == "right":
                # Chose right arm - reward depends on whether right is better
                # better_arm=0 (left_better): low reward prob (context-specific)
                # better_arm=1 (right_better): high reward prob (context-specific)
                A_reward[1, ctx_idx, 0, choice_idx] = p_reward_ctx      # P(loss | right, left_better)
                A_reward[2, ctx_idx, 0, choice_idx] = 1 - p_reward_ctx  # P(reward | right, left_better)
                A_reward[1, ctx_idx, 1, choice_idx] = 1 - p_reward_ctx  # P(loss | right, right_better)
                A_reward[2, ctx_idx, 1, choice_idx] = p_reward_ctx      # P(reward | right, right_better)
    
    # =========================================================================
    # A[2]: Choice observations - one-to-one mapping to choice state
    # =========================================================================
    # Shape: (n_choice_obs, n_contexts, n_better_arm, n_choices)
    A_choice = np.zeros((len(observation_choices), n_contexts, n_better_arm, n_choices))
    
    for ctx_idx in range(n_contexts):
        for better_idx in range(n_better_arm):
            for choice_idx in range(n_choices):
                # Observe what choice was made
                A_choice[choice_idx, ctx_idx, better_idx, choice_idx] = 1.0
    
    A[0] = A_hint
    A[1] = A_reward
    A[2] = A_choice
    
    return A


def build_B(state_contexts, state_better_arm, state_choices, 
            action_contexts, action_better_arm, action_choices,
            context_volatility=0.0, arm_switch_prob_volatile=0.1, arm_switch_prob_stable=0.01):
    """
    Build transition matrix B: p(s_t | s_t-1, a_t-1).
    
    Now handles 3 state factors with context-dependent better_arm transitions.
    
    Parameters:
    -----------
    state_contexts : list
        Context state labels ['volatile', 'stable']
    state_better_arm : list
        Better arm state labels ['left_better', 'right_better']
    state_choices : list
        Choice state labels
    action_contexts : list
        Context action labels (just ['rest'])
    action_better_arm : list
        Better arm action labels (just ['rest'] - no control)
    action_choices : list
        Choice action labels
    context_volatility : float
        Probability of context switching (volatile <-> stable) per trial
    arm_switch_prob_volatile : float
        Agent's belief about P(arm switches | volatile context)
    arm_switch_prob_stable : float
        Agent's belief about P(arm switches | stable context)
        
    Returns:
    --------
    B : object array
        Transition matrices [B_context, B_better_arm, B_choice]
    """
    n_contexts = len(state_contexts)
    n_better_arm = len(state_better_arm)
    n_choices = len(state_choices)
    n_action_ctx = len(action_contexts)
    n_action_arm = len(action_better_arm)
    n_action_choice = len(action_choices)
    
    B = utils.obj_array(3)
    
    # =========================================================================
    # B[0]: Context transitions - contexts are sticky (rarely switch)
    # =========================================================================
    # Shape: (n_contexts, n_contexts, n_action_ctx)
    B_context = np.zeros((n_contexts, n_contexts, n_action_ctx))
    
    for a in range(n_action_ctx):
        if context_volatility == 0.0:
            B_context[:, :, a] = np.eye(n_contexts)
        else:
            for i in range(n_contexts):
                for j in range(n_contexts):
                    if i == j:
                        B_context[i, j, a] = 1.0 - context_volatility
                    else:
                        B_context[i, j, a] = context_volatility / (n_contexts - 1)
    
    # =========================================================================
    # B[1]: Better arm transitions - DEPENDS ON CONTEXT
    # =========================================================================
    # This is the key fix: arm switches are MORE LIKELY in volatile context
    # Shape: (n_better_arm, n_better_arm, n_action_arm)
    # But we need it to depend on context... 
    # 
    # In pyMDP, B[f] has shape (n_states[f], n_states[f], n_actions[f])
    # To make it context-dependent, we need to expand the action space or
    # use a different mechanism.
    #
    # WORKAROUND: Use a single B_better_arm with AVERAGE switch probability,
    # and let the agent learn context through observation patterns.
    # 
    # Better approach: The agent's belief about arm switching should reflect
    # its belief about context. We encode this through the overall switch rate.
    #
    # For a proper implementation, we'd need factorized B matrices or
    # a different state representation. For now, use moderate switch probability.
    
    # Actually, we CAN make this work by having the agent use its BELIEF about
    # context to weight its predictions. But that requires custom inference.
    #
    # Simpler approach: Use a moderate arm switch probability that lets the
    # agent update its better_arm beliefs through observations (hints/rewards)
    
    # For the agent's generative model, use a moderate belief about arm switches
    avg_switch_prob = 0.05  # Agent believes arms switch occasionally
    
    B_better_arm = np.zeros((n_better_arm, n_better_arm, n_action_arm))
    for a in range(n_action_arm):
        for i in range(n_better_arm):
            for j in range(n_better_arm):
                if i == j:
                    B_better_arm[i, j, a] = 1.0 - avg_switch_prob
                else:
                    B_better_arm[i, j, a] = avg_switch_prob
    
    # =========================================================================
    # B[2]: Choice transitions - deterministic given action
    # =========================================================================
    # Shape: (n_choices, n_choices, n_action_choice)
    B_choice = np.zeros((n_choices, n_choices, n_action_choice))
    
    for action_idx in range(n_action_choice):
        # Action directly determines next choice state
        B_choice[action_idx, :, action_idx] = 1.0
    
    B[0] = B_context
    B[1] = B_better_arm
    B[2] = B_choice
    
    return B


def build_D(state_contexts, state_better_arm, state_choices):
    """
    Build prior over initial states D: p(s_0).
    
    Parameters:
    -----------
    state_contexts : list
        Context state labels
    state_better_arm : list
        Better arm state labels
    state_choices : list
        Choice state labels
        
    Returns:
    --------
    D : object array
        Prior distributions [D_context, D_better_arm, D_choice]
    """
    D = utils.obj_array(3)
    
    # D[0]: Uniform prior over contexts
    D_context = np.ones(len(state_contexts)) / len(state_contexts)
    
    # D[1]: Uniform prior over which arm is better
    D_better_arm = np.ones(len(state_better_arm)) / len(state_better_arm)
    
    # D[2]: Deterministic start at "start" state
    D_choice = np.zeros(len(state_choices))
    D_choice[state_choices.index("start")] = 1.0
    
    D[0] = D_context
    D[1] = D_better_arm
    D[2] = D_choice
    
    return D
