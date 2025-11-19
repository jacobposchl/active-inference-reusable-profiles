"""
Value function implementations for M1, M2, and M3 models.
"""

import numpy as np
from pymdp.maths import softmax

def make_values_M1(C_reward_logits=None, gamma=1.0, E_logits=None):
    """
    Model 1: Static global precision and preferences.
    
    No adaptation - all parameters are fixed across trials and states.
    
    Parameters:
    -----------
    C_reward_logits : array-like
        Log preferences over reward observations
    gamma : float
        Fixed policy precision
    E_logits : array-like or None
        Log preferences over policies (optional)
        
    Returns:
    --------
    value_fn : function
        Value function that returns (C_t, E_t, gamma_t)
    """

    C_logits = np.asarray(C_reward_logits, float)
    E_logits = None if E_logits is None else np.asarray(E_logits, float)
    
    def value_fn(q_context_t, t):
        """Returns fixed values regardless of beliefs or time."""
        C_t = softmax(C_logits)
        E_t = None if E_logits is None else softmax(E_logits)
        return C_t, E_t, float(gamma)
    
    return value_fn


def make_values_M2(C_reward_logits=None, gamma_schedule=None, E_logits=None):
    """
    Model 2: Dynamic global precision based on belief entropy.
    
    Policy precision adapts to uncertainty, but preferences are fixed.
    
    Parameters:
    -----------
    C_reward_logits : array-like
        Log preferences over reward observations (fixed)
    gamma_schedule : function or None
        Function(q_context_t, t) -> gamma_t
        If None, uses default entropy-coupled schedule
    E_logits : array-like or None
        Log preferences over policies (optional, fixed)
        
    Returns:
    --------
    value_fn : function
        Value function that returns (C_t, E_t, gamma_t)
    """

    if gamma_schedule is None:
        # Default: precision inversely related to entropy
        def gamma_schedule(q_context_t, t, g_base=1.5, k=1.0):
            """
            Lower precision when uncertain (higher entropy).
            
            gamma = g_base / (1 + k * H(q_context))
            """
            p = np.clip(np.asarray(q_context_t, float), 1e-12, 1.0)
            H = -(p * np.log(p)).sum()
            return g_base / (1.0 + k * H)
    
    C_logits = np.asarray(C_reward_logits, float)
    E_logits = None if E_logits is None else np.asarray(E_logits, float)
    
    def value_fn(q_context_t, t):
        """Returns fixed C and E, but dynamic gamma."""
        C_t = softmax(C_logits)
        E_t = None if E_logits is None else softmax(E_logits)
        gamma_t = float(gamma_schedule(q_context_t, t))
        return C_t, E_t, gamma_t
    
    return value_fn


def make_values_M3(profiles, Z, num_policies=None, policies=None, num_actions_per_factor=None):
    """
    Model 3: Profile model with state-conditional mixing.
    
    Multiple value profiles are mixed based on current beliefs about
    hidden states via assignment matrix Z.
    
    Parameters:
    -----------
    profiles : list of dict
        Each profile contains:
        - 'phi_logits': outcome preference logits (-> C vector)
        - 'xi_logits': action preference logits (-> E vector)
        - 'gamma': policy precision scalar
    Z : array-like, shape (num_states, num_profiles)
        Assignment matrix linking hidden states to profiles.
        Each row sums to 1.0
    policies : list of arrays (required if using xi_logits)
        Complete list of policies from pymdp agent
    num_actions_per_factor : list (required if using xi_logits)
        Number of actions for each control factor
        
    Returns:
    --------
    value_fn : function
        Value function that returns (C_t, E_t, gamma_t)
    """

    # Normalize Z
    Z = np.asarray(Z, float)
    Z /= Z.sum(axis=1, keepdims=True)
    
    K = len(profiles)  # Number of profiles
    
    # Extract profile parameters
    # Shape: (2 profiles, 3 reward outcomes)
    PHI = np.stack([np.asarray(p['phi_logits'], float) for p in profiles], axis=0)
    # Profile 1: low, Profile 2: high
    GAM = np.array([float(p['gamma']) for p in profiles])
    
    # Check if profiles have policy priors
    has_xi = all('xi_logits' in p for p in profiles)
    
    if has_xi:
        if policies is None or num_actions_per_factor is None:
            raise ValueError("Must provide 'policies' and 'num_actions_per_factor' "
                           "when using xi_logits")
        
        # Convert action-level preferences to policy-level preferences
        XI_list = []
        for p in profiles:
            policy_logits = map_action_prefs_to_policy_prefs(
                p['xi_logits'],
                policies,
                num_actions_per_factor
            )
            XI_list.append(policy_logits)
        
        XI = np.stack(XI_list, axis=0)
    else:
        XI = None
    
    def value_fn(q_context_t, t):
        """
        Compute trial-specific values by mixing profiles.
        
        Parameters:
        -----------
        q_context_t : array, shape (num_context_states,)
            Current belief over context states
        t : int
            Trial number (not used but kept for API consistency)
            
        Returns:
        --------
        C_t : array
            Mixed outcome preferences
        E_t : array or None
            Mixed policy priors
        gamma_t : float
            Mixed policy precision
        """
        # Compute profile weights from beliefs
        w = np.asarray(q_context_t, float) @ Z
        w = w / (w.sum() + 1e-12)  # Normalize
        
        # Mix outcome preferences
        phi_t = (w[:, None] * PHI).sum(axis=0)
        C_t = softmax(phi_t)
        
        # Mix policy precision
        gamma_t = float((w * GAM).sum())
        
        # Mix policy priors if available
        if XI is not None:
            xi_t = (w[:, None] * XI).sum(axis=0)
            E_t = softmax(xi_t)
        else:
            E_t = None
        
        return C_t, E_t, gamma_t
    
    return value_fn


def map_action_prefs_to_policy_prefs(xi_logits_per_action, policies, 
                                     num_actions_per_factor):
    """
    Map action-level preferences to policy-level preferences.
    
    This is needed because xi_logits are defined over individual actions,
    but pymdp needs preferences over multi-step policies.
    
    Parameters:
    -----------
    xi_logits_per_action : array-like, shape (num_actions,)
        Log preferences for each action type
    policies : list of arrays
        Policies from pymdp. Each is [policy_len, num_factors] array
    num_actions_per_factor : list
        Number of actions for each control factor
        
    Returns:
    --------
    policy_logits : np.ndarray, shape (num_policies,)
        Log preferences for each policy
    """
    xi = np.asarray(xi_logits_per_action, dtype=float)
    num_policies = len(policies)
    policy_logits = np.zeros(num_policies)
    
    for pol_idx, policy in enumerate(policies):
        # policy shape: [policy_len, num_control_factors]
        # Sum log preferences across timesteps
        log_pref = 0.0
        for t in range(policy.shape[0]):
            action_idx = int(policy[t, 1])  # Factor 1 is choice factor
            log_pref += xi[action_idx]
        
        policy_logits[pol_idx] = log_pref
    
    return policy_logits


def make_value_fn(model, **cfg):
    """
    Factory function to create value functions for different models.
    
    Parameters:
    -----------
    model : str
        Model identifier: 'M1', 'M2', or 'M3'
    **cfg : dict
        Model-specific configuration parameters
        
    Returns:
    --------
    value_fn : function
        Value function that takes (q_context_t, t) and returns (C_t, E_t, gamma_t)
    """
    if model == "M1":
        return make_values_M1(**cfg)
    elif model == "M2":
        return make_values_M2(**cfg)
    elif model == "M3":
        return make_values_M3(**cfg)
    else:
        raise ValueError(f"model must be 'M1', 'M2', or 'M3', got '{model}'")
