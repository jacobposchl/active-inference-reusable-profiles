"""
Value functions for M1, M2, and M3 models adapted for changepoint task.

These follow the same pattern as the bandit task value functions:
- Return (C_t, E_t, gamma_t) where C_t is outcome preferences, E_t is policy priors
- M3 uses profile mixing via Z matrix based on volatility beliefs
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from pymdp.maths import softmax

from .state_space import PositionStateSpace


def _compute_entropy(q):
    """Compute entropy of a probability distribution."""
    p = np.clip(np.asarray(q, float), 1e-12, 1.0)
    return -(p * np.log(p)).sum()


def make_values_M1(
    C_outcome_logits=None,
    gamma: float = 2.5,
    E_logits=None,
) -> Callable:
    """
    Model 1: Static precision and preferences.
    
    Parameters
    ----------
    C_outcome_logits : array-like, optional
        Log preferences over outcome observations (bag positions)
        If None, uses uniform preferences
    gamma : float
        Fixed policy precision
    E_logits : array-like, optional
        Log preferences over policies (optional)
    
    Returns
    -------
    value_fn : function
        Value function that returns (C_t, E_t, gamma_t)
    """
    if C_outcome_logits is None:
        # Default: prefer outcomes near center (neutral)
        C_outcome_logits = np.zeros(30)  # Will be set based on n_bins dynamically
    C_logits = np.asarray(C_outcome_logits, float)
    E_logits = None if E_logits is None else np.asarray(E_logits, float)
    
    def value_fn(qs, t):
        """
        Returns fixed values regardless of beliefs.
        
        Parameters
        ----------
        qs : list of arrays
            Beliefs over state factors [q_position]
        t : int
            Trial number (unused)
        """
        # If C_logits size doesn't match, create uniform
        n_outcomes = len(qs[0]) if len(qs) > 0 else 30
        if len(C_logits) != n_outcomes:
            C_logits_actual = np.zeros(n_outcomes)
        else:
            C_logits_actual = C_logits
        
        C_t = softmax(C_logits_actual)
        E_t = None if E_logits is None else softmax(E_logits)
        return C_t, E_t, float(gamma)
    
    return value_fn


def make_values_M2(
    C_outcome_logits=None,
    gamma_schedule=None,
    E_logits=None,
    gamma_base=None,
    entropy_k=None,
) -> Callable:
    """
    Model 2: Dynamic precision based on position belief entropy.
    
    Precision adapts to uncertainty: higher entropy → lower precision.
    
    Parameters
    ----------
    C_outcome_logits : array-like, optional
        Log preferences over outcomes (fixed)
    gamma_schedule : function or None
        Function(H_position, t) -> gamma_t where H_position is entropy
        If None, uses default entropy-coupled schedule
    E_logits : array-like, optional
        Log preferences over policies (optional, fixed)
    gamma_base : float, optional
        Base precision value (default: 2.5). Used if gamma_schedule is None.
    entropy_k : float, optional
        Entropy coupling strength (default: 1.0). Used if gamma_schedule is None.
    """
    # If gamma_base/entropy_k provided, create schedule with those values
    if gamma_schedule is None:
        g_base = gamma_base if gamma_base is not None else 2.5
        k = entropy_k if entropy_k is not None else 1.0
        def gamma_schedule(H_position, t):
            return g_base / (1.0 + k * H_position)
    
    if C_outcome_logits is None:
        C_outcome_logits = np.zeros(30)  # Will be set dynamically
    C_logits = np.asarray(C_outcome_logits, float)
    E_logits = None if E_logits is None else np.asarray(E_logits, float)
    
    def value_fn(qs, t):
        """
        Returns fixed C and E, but dynamic gamma based on position uncertainty.
        
        Parameters
        ----------
        qs : list of arrays
            Beliefs over state factors [q_position]
        t : int
            Trial number
        """
        q_position = qs[0] if len(qs) > 0 else np.ones(30) / 30
        H_position = _compute_entropy(q_position)
        
        n_outcomes = len(q_position)
        if len(C_logits) != n_outcomes:
            C_logits_actual = np.zeros(n_outcomes)
        else:
            C_logits_actual = C_logits
        
        C_t = softmax(C_logits_actual)
        E_t = None if E_logits is None else softmax(E_logits)
        gamma_t = float(gamma_schedule(H_position, t))
        
        return C_t, E_t, gamma_t
    
    return value_fn


def make_values_M3(
    profiles: List[dict],
    Z: np.ndarray,
    policies: Optional[List] = None,
    num_actions_per_factor: Optional[List] = None,
) -> Callable:
    """
    Model 3: Profile model with volatility-based mixing via Z matrix.
    
    Multiple value profiles are mixed based on beliefs about volatility
    (stable vs volatile) through the assignment matrix Z.
    
    Parameters
    ----------
    profiles : list of dict
        Each profile contains:
        - 'phi_logits': outcome preference logits (-> C vector)
        - 'xi_logits': action preference logits (-> E vector, optional)
        - 'gamma': policy precision scalar
        Profile 0 = volatile/exploratory, Profile 1 = stable/exploitative
    Z : array-like, shape (n_volatility_states, n_profiles)
        Assignment matrix mapping volatility beliefs to profile weights
        Z[0, :] = weights when stable, Z[1, :] = weights when volatile
    policies : list of arrays, optional
        Complete list of policies from pymdp agent (required if using xi_logits)
    num_actions_per_factor : list, optional
        Number of actions for each control factor (required if using xi_logits)
    
    Returns
    -------
    value_fn : function
        Value function that returns (C_t, E_t, gamma_t)
    """
    K = len(profiles)
    if K != 2:
        raise ValueError(f"M3 requires exactly 2 profiles, got {K}")
    
    # Normalize Z matrix
    Z_mat = np.asarray(Z, float)
    Z_mat = Z_mat / (Z_mat.sum(axis=1, keepdims=True) + 1e-12)
    
    # Extract profile parameters
    PHI = np.stack([np.asarray(p['phi_logits'], float) for p in profiles], axis=0)
    GAM = np.array([float(p['gamma']) for p in profiles])
    
    # Check if profiles have policy priors
    has_xi = all('xi_logits' in p for p in profiles)
    
    if has_xi:
        if policies is None or num_actions_per_factor is None:
            raise ValueError("Must provide 'policies' and 'num_actions_per_factor' when using xi_logits")
        
        # Convert action-level preferences to policy-level preferences
        XI_list = []
        for p in profiles:
            policy_logits = _map_action_prefs_to_policy_prefs(
                p['xi_logits'],
                policies,
                num_actions_per_factor
            )
            XI_list.append(policy_logits)
        
        XI = np.stack(XI_list, axis=0)
    else:
        XI = None
    
    def value_fn(qs, t):
        """
        Compute trial-specific values by mixing profiles based on volatility beliefs.
        
        Profile weights = q_volatility @ Z
        - Believe stable (q_volatility[0] high) → more weight on Profile 1
        - Believe volatile (q_volatility[1] high) → more weight on Profile 0
        
        Parameters
        ----------
        qs : list of arrays
            Beliefs over state factors [q_position]
            Note: volatility beliefs are inferred separately, not in qs
        t : int
            Trial number (not used but kept for API consistency)
        """
        # Volatility beliefs need to be passed separately or inferred
        # For now, we'll need to modify the agent to pass q_volatility
        # This is a limitation - we need q_volatility from the agent
        # For now, return a function that takes q_volatility as additional arg
        raise NotImplementedError(
            "M3 value function needs q_volatility. "
            "Use make_values_M3_with_volatility instead, or modify agent to pass q_volatility."
        )
    
    return value_fn


def make_values_M3_with_volatility(
    profiles: List[dict],
    Z: np.ndarray,
    policies: Optional[List] = None,
    num_actions_per_factor: Optional[List] = None,
    dynamic_phi_generator: Optional[Callable] = None,
) -> Callable:
    """
    Model 3: Profile model with volatility-based mixing.
    
    Same as make_values_M3 but value_fn takes q_volatility as additional parameter.
    
    Parameters
    ----------
    profiles : list of dict
        Profile definitions with 'gamma' and optionally 'phi_logits'.
        If dynamic_phi_generator is provided, 'phi_logits' in profiles is ignored.
    Z : np.ndarray
        Assignment matrix mapping volatility states to profiles
    policies : list, optional
        Policy list from pymdp agent (needed if profiles have 'xi_logits')
    num_actions_per_factor : list, optional
        Number of actions per factor (needed if profiles have 'xi_logits')
    dynamic_phi_generator : callable, optional
        Function(qs, profile_idx, position_space) -> phi_logits
        If provided, generates phi_logits dynamically from current beliefs.
        profile_idx: 0 = explore (volatile), 1 = exploit (stable)
    """
    K = len(profiles)
    if K != 2:
        raise ValueError(f"M3 requires exactly 2 profiles, got {K}")
    
    # Normalize Z matrix
    Z_mat = np.asarray(Z, float)
    Z_mat = Z_mat / (Z_mat.sum(axis=1, keepdims=True) + 1e-12)
    
    # Extract static profile parameters (gamma always static)
    GAM = np.array([float(p['gamma']) for p in profiles])
    
    # Extract static PHI if not using dynamic generator
    if dynamic_phi_generator is None:
        PHI = np.stack([np.asarray(p.get('phi_logits', np.zeros(30)), float) for p in profiles], axis=0)
    else:
        # PHI will be computed dynamically, but we need to know the size
        # Use first profile's phi_logits size as template, or default
        n_bins_template = len(profiles[0].get('phi_logits', np.zeros(30)))
        PHI = None  # Will be computed dynamically
    
    # Check if profiles have policy priors
    has_xi = all('xi_logits' in p for p in profiles)
    
    if has_xi:
        if policies is None or num_actions_per_factor is None:
            raise ValueError("Must provide 'policies' and 'num_actions_per_factor' when using xi_logits")
        
        XI_list = []
        for p in profiles:
            policy_logits = _map_action_prefs_to_policy_prefs(
                p['xi_logits'],
                policies,
                num_actions_per_factor
            )
            XI_list.append(policy_logits)
        
        XI = np.stack(XI_list, axis=0)
    else:
        XI = None
    
    def value_fn(qs, t, q_volatility=None, position_space=None):
        """
        Compute trial-specific values by mixing profiles based on volatility beliefs.
        
        Parameters
        ----------
        qs : list of arrays
            Beliefs over state factors [q_position]
        t : int
            Trial number
        q_volatility : np.ndarray, optional
            Beliefs over volatility states [stable, volatile]
            If None, uses uniform
        position_space : PositionStateSpace, optional
            Required if dynamic_phi_generator is used
        """
        if q_volatility is None:
            q_volatility = np.array([0.5, 0.5])
        
        # Compute profile weights from volatility beliefs via Z matrix
        w = np.asarray(q_volatility, float) @ Z_mat
        w = w / (w.sum() + 1e-12)  # Normalize
        
        # Get outcome preferences (dynamic or static)
        n_outcomes = len(qs[0]) if len(qs) > 0 else (PHI.shape[1] if PHI is not None else 30)
        
        if dynamic_phi_generator is not None:
            # Generate phi_logits dynamically for each profile
            if position_space is None:
                raise ValueError("position_space required when using dynamic_phi_generator")
            PHI_dynamic = np.stack([
                dynamic_phi_generator(qs, 0, position_space),  # Profile 0: explore
                dynamic_phi_generator(qs, 1, position_space),  # Profile 1: exploit
            ], axis=0)
            PHI_use = PHI_dynamic
        else:
            # Use static PHI
            if PHI.shape[1] != n_outcomes:
                # Pad or truncate PHI to match
                if PHI.shape[1] < n_outcomes:
                    pad_width = ((0, 0), (0, n_outcomes - PHI.shape[1]))
                    PHI_use = np.pad(PHI, pad_width, mode='constant', constant_values=0.0)
                else:
                    PHI_use = PHI[:, :n_outcomes]
            else:
                PHI_use = PHI
        
        phi_t = (w[:, None] * PHI_use).sum(axis=0)
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


def _map_action_prefs_to_policy_prefs(xi_logits_per_action, policies, num_actions_per_factor):
    """
    Map action-level preferences to policy-level preferences.
    
    This is needed because xi_logits are defined over individual actions,
    but pymdp needs preferences over multi-step policies.
    """
    xi = np.asarray(xi_logits_per_action, dtype=float)
    num_policies = len(policies)
    policy_logits = np.zeros(num_policies)
    
    for pol_idx, policy in enumerate(policies):
        # policy shape: [policy_len, num_action_factors]
        # For changepoint, we only have one action factor (position prediction)
        log_pref = 0.0
        for t in range(policy.shape[0]):
            action_idx = int(policy[t, 0])  # Factor 0 is position/action
            log_pref += xi[action_idx]
        
        policy_logits[pol_idx] = log_pref
    
    return policy_logits


def make_value_fn(model: str, **kwargs) -> Callable:
    """
    Factory function to create value functions for different models.
    
    Parameters
    ----------
    model : str
        Model identifier: 'M1', 'M2', or 'M3'
    **kwargs : dict
        Model-specific configuration parameters
    
    Returns
    -------
    value_fn : function
        Value function that takes (qs, t) and returns (C_t, E_t, gamma_t)
        For M3, use make_values_M3_with_volatility for volatility support
    """
    if model == "M1":
        return make_values_M1(**kwargs)
    elif model == "M2":
        return make_values_M2(**kwargs)
    elif model == "M3":
        # Check if q_volatility support is needed
        if 'q_volatility' in str(kwargs) or 'volatility' in str(kwargs).lower():
            return make_values_M3_with_volatility(**kwargs)
        else:
            return make_values_M3(**kwargs)
    else:
        raise ValueError(f"model must be 'M1', 'M2', or 'M3', got '{model}'")
