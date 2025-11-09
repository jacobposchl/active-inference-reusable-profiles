"""
Generative model construction (A, B, D matrices).

These matrices define the agent's beliefs about how the world works.
"""
import numpy as np
from pymdp import utils


def build_A(num_modalities, state_contexts, state_choices, 
            observation_hints, observation_rewards, observation_choices,
            p_hint, p_reward):
    """
    Build likelihood matrix A: p(o_t | s_t).
    
    Parameters:
    -----------
    num_modalities : int
        Number of observation modalities
    state_contexts : list
        Context state labels
    state_choices : list
        Choice state labels
    observation_hints : list
        Hint observation labels
    observation_rewards : list
        Reward observation labels
    observation_choices : list
        Choice observation labels
    p_hint : float
        Believed probability of correct hint
    p_reward : float
        Believed probability of reward from better arm
        
    Returns:
    --------
    A : object array
        Likelihood matrices for each modality
    """
    A = utils.obj_array(num_modalities)
    
    # A[0]: Hint observations
    A_hint = np.zeros((len(observation_hints), len(state_contexts), len(state_choices)))
    
    for choice_index, choice in enumerate(state_choices):
        if choice == "start":
            A_hint[0, :, choice_index] = 1.0  # null hint
        elif choice == "hint":
            # Left hint more likely if left_better
            A_hint[1, :, choice_index] = [p_hint, 1 - p_hint]
            A_hint[2, :, choice_index] = [1 - p_hint, p_hint]
        elif choice in ["left", "right"]:
            A_hint[0, :, choice_index] = 1.0  # null hint
    
    # A[1]: Reward observations
    A_reward = np.zeros((len(observation_rewards), len(state_contexts), len(state_choices)))
    
    for choice_index, choice in enumerate(state_choices):
        if choice in ["start", "hint"]:
            A_reward[0, :, choice_index] = 1.0  # null reward
        elif choice == "left":
            # Reward more likely if left_better
            A_reward[1, :, choice_index] = [1 - p_reward, p_reward]  # loss
            A_reward[2, :, choice_index] = [p_reward, 1 - p_reward]  # reward
        elif choice == "right":
            # Reward more likely if right_better
            A_reward[1, :, choice_index] = [p_reward, 1 - p_reward]  # loss
            A_reward[2, :, choice_index] = [1 - p_reward, p_reward]  # reward
    
    # A[2]: Choice observations (one-to-one mapping) - we observe the choices we made
    A_choice = np.zeros((len(observation_choices), len(state_contexts), len(state_choices)))
    
    for choice_index in range(len(state_choices)):
        A_choice[choice_index, :, choice_index] = 1.0
    
    A[0] = A_hint
    A[1] = A_reward
    A[2] = A_choice
    
    return A


def build_B(state_contexts, state_choices, action_contexts, action_choices, context_volatility=0.0):
    """
    Build transition matrix B: p(s_t | s_t-1, a_t-1).
    
    Parameters:
    -----------
    state_contexts : list
        Context state labels
    state_choices : list
        Choice state labels
    action_contexts : list
        Context action labels
    action_choices : list
        Choice action labels
    context_volatility : float
        Probability of context flip per trial (agent's belief about volatility)
        
    Returns:
    --------
    B : object array
        Transition matrices [B_context, B_choice]
    """

    B = utils.obj_array(2)
    
    # B[0]: Context transitions
    B_context = np.zeros((len(state_contexts), len(state_contexts), len(action_contexts)))
    
    if context_volatility == 0.0:
        # Contexts are stable
        B_context[:, :, 0] = np.eye(len(state_contexts))
    else:
        # Contexts can flip with some probability
        for i in range(len(state_contexts)):
            for j in range(len(state_contexts)):
                if i == j:
                    B_context[i, j, 0] = 1.0 - context_volatility
                else:
                    B_context[i, j, 0] = context_volatility
    
    # B[1]: Choice transitions (deterministic given action)
    B_choice = np.zeros((len(state_choices), len(state_choices), len(action_choices)))
    
    for choice_index in range(len(state_choices)):
        # One-to-one mapping: action directly determines next state
        B_choice[choice_index, :, choice_index] = 1.0
    
    B[0] = B_context
    B[1] = B_choice
    
    return B


def build_D(state_contexts, state_choices):
    """
    Build prior over initial states D: p(s_0).
    
    Parameters:
    -----------
    state_contexts : list
        Context state labels
    state_choices : list
        Choice state labels
        
    Returns:
    --------
    D : object array
        Prior distributions [D_context, D_choice]
    """
    
    D = utils.obj_array(2)
    
    # D[0]: Uniform prior over contexts
    D_context = np.array([0.5, 0.5])
    
    # D[1]: Deterministic start at "start" state
    D_choice = np.zeros(len(state_choices))
    D_choice[state_choices.index("start")] = 1.0
    
    D[0] = D_context
    D[1] = D_choice
    
    return D
