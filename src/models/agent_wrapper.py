"""
Agent wrapper classes for running simulations and tracking log-likelihoods.
"""

import numpy as np
from pymdp.agent import Agent
from pymdp import utils


class AgentRunner:
    """
    Agent wrapper that handles value profiles and episode execution.
    """
    
    def __init__(self, A, B, D, value_fn, observation_hints, observation_rewards,
                 observation_choices, action_choices, reward_mod_idx=1,
                 policy_len=2, inference_horizon=1):
        """
        Initialize agent runner.
        
        Parameters:
        -----------
        A : object array
            Likelihood matrices
        B : object array
            Transition matrices
        D : object array
            Prior over initial states
        value_fn : function
            Value function that returns (C_t, E_t, gamma_t)
        observation_hints : list
            Hint observation labels
        observation_rewards : list
            Reward observation labels
        observation_choices : list
            Choice observation labels
        action_choices : list
            Choice action labels
        reward_mod_idx : int
            Index of reward modality for C vector
        policy_len : int
            Length of policies to consider
        inference_horizon : int
            Planning horizon
        """

        self.value_fn = value_fn
        self.reward_mod_idx = reward_mod_idx
        
        self.observation_hints = observation_hints
        self.observation_rewards = observation_rewards
        self.observation_choices = observation_choices
        self.action_choices = action_choices
        self.policy_len = policy_len
        
        self.gamma_t = None
        
        # Initialize C vector (will be updated each trial)
        C0 = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
        
        # Create pymdp Agent
        self.agent = Agent(
            A=A, B=B, C=C0, D=D,
            policy_len=policy_len,
            inference_horizon=inference_horizon,
            control_fac_idx=[1],  # Only control choice factor
            use_utility=True,
            use_states_info_gain=True,
            action_selection="stochastic",
            gamma=16  # Will be overridden by value function
        )
    
    def obs_labels_to_ids(self, obs_labels):
        """Convert observation strings to indices."""
        return [
            self.observation_hints.index(obs_labels[0]),
            self.observation_rewards.index(obs_labels[1]),
            self.observation_choices.index(obs_labels[2])
        ]
    
    def action_id_to_label(self, chosen_action_ids):
        """Convert action index to string label."""
        a_idx = int(chosen_action_ids[1])
        return self.action_choices[a_idx]
    
    def step(self, obs_ids, t):
        """
        Execute one trial: infer states, update values, select action.
        
        Parameters:
        -----------
        obs_ids : list of int
            Observation indices for each modality
        t : int
            Trial number
            
        Returns:
        --------
        action_label : str
            Chosen action
        qs : list of arrays
            Posterior beliefs over states
        q_pi : array
            Posterior over policies
        efe : array
            Expected free energy for each policy
        gamma_t : float
            Policy precision used this trial
        """
        
        # Infer hidden states
        qs = self.agent.infer_states(obs_ids)
        
        # Get context beliefs
        q_context = qs[0]
        
        # Compute value profile for this trial
        C_t, E_t, gamma_t = self.value_fn(q_context, t)
        
        # Update agent parameters
        self.agent.C[self.reward_mod_idx] = C_t
        
        if E_t is not None:
            if len(E_t) == len(self.agent.policies):
                self.agent.E = E_t
        
        self.gamma_t = float(gamma_t)
        self.agent.gamma = self.gamma_t
        
        # Infer policy posterior
        q_pi, efe = self.agent.infer_policies()
        
        # Sample action
        chosen_action_ids = self.agent.sample_action()
        action_label = self.action_id_to_label(chosen_action_ids)
        
        return action_label, qs, q_pi, efe, self.gamma_t


class AgentRunnerWithLL(AgentRunner):
    """
    Extended AgentRunner that tracks action log-likelihoods.
    """
    
    def step_with_ll(self, obs_ids, t):
        """
        Enhanced step that returns log-likelihood of chosen action.
        
        Returns:
        --------
        action_label : str
        qs : list of arrays
        q_pi : array
        efe : array
        gamma_t : float
        log_likelihood : float
            Log-probability of the action actually taken
        """
        # Standard inference
        qs = self.agent.infer_states(obs_ids)
        q_context = qs[0]
        
        # Get value profile
        C_t, E_t, gamma_t = self.value_fn(q_context, t)
        
        # Update agent
        self.agent.C[self.reward_mod_idx] = C_t
        if E_t is not None and len(E_t) == len(self.agent.policies):
            self.agent.E = E_t
        
        self.gamma_t = float(gamma_t)
        self.agent.gamma = self.gamma_t
        
        # Infer policies
        q_pi, efe = self.agent.infer_policies()
        
        # Sample action
        chosen_action_ids = self.agent.sample_action()
        u_choice = int(chosen_action_ids[1])
        action_label = self.action_id_to_label(chosen_action_ids)
        
        # Compute log-likelihood of chosen action
        action_ll = -np.inf
        for pi_idx, policy in enumerate(self.agent.policies):
            if int(policy[0, 1]) == u_choice:  # First action of policy matches
                if action_ll == -np.inf:
                    action_ll = np.log(q_pi[pi_idx] + 1e-16)
                else:
                    # Log-sum-exp for numerical stability
                    action_ll = np.logaddexp(action_ll, np.log(q_pi[pi_idx] + 1e-16))
        
        return action_label, qs, q_pi, efe, self.gamma_t, action_ll

    def action_logprob(self, obs_ids, action_label, t):
        """
        Compute log-probability of a provided `action_label` under the agent's
        current policy posterior when conditioned on `obs_ids`.

        This implements teacher-forcing style evaluation: it does inference on
        the supplied observation, updates internal value parameters, computes
        the posterior over policies `q_pi`, and returns the log-probability of
        the given action (log p(a_t | o_{1:t}, a_{1:t-1})).
        """
        # Infer hidden states from supplied observations
        qs = self.agent.infer_states(obs_ids)
        q_context = qs[0]

        # Value/profile update
        C_t, E_t, gamma_t = self.value_fn(q_context, t)
        self.agent.C[self.reward_mod_idx] = C_t
        if E_t is not None and len(E_t) == len(self.agent.policies):
            self.agent.E = E_t

        self.gamma_t = float(gamma_t)
        self.agent.gamma = self.gamma_t

        # Infer policies
        q_pi, efe = self.agent.infer_policies()

        # Map action label to index for the choice factor
        try:
            u_choice = int(self.action_choices.index(action_label))
        except ValueError:
            # Fallback: if action label format differs, try to match by suffix
            u_choice = None
            for idx, lab in enumerate(self.action_choices):
                if lab == action_label:
                    u_choice = idx
                    break
            if u_choice is None:
                raise

        # Compute log-probability by summing q_pi over policies whose first
        # action matches the requested action (log-sum-exp)
        action_ll = -np.inf
        for pi_idx, policy in enumerate(self.agent.policies):
            if int(policy[0, 1]) == u_choice:
                if action_ll == -np.inf:
                    action_ll = np.log(q_pi[pi_idx] + 1e-16)
                else:
                    action_ll = np.logaddexp(action_ll, np.log(q_pi[pi_idx] + 1e-16))

        return action_ll


def run_episode(runner, env, T=200, verbose=False, initial_obs_labels=None,
                print_around_reversals=False, window=5):
    """
    Run complete episode with agent-environment interaction.
    
    Parameters:
    -----------
    runner : AgentRunner
        Agent wrapper instance
    env : TwoArmedBandit
        Environment instance
    T : int
        Number of trials
    verbose : bool
        Print every trial
    initial_obs_labels : list or None
        Starting observation
    print_around_reversals : bool
        Print details around reversals
    window : int
        Window size for reversal printing
        
    Returns:
    --------
    logs : dict
        Trial-by-trial data
    """
    from ..utils.helpers import print_trial_details
    
    if initial_obs_labels is None:
        initial_obs_labels = ['null', 'null', 'observe_start']
    
    obs_ids = runner.obs_labels_to_ids(initial_obs_labels)
    
    logs = {
        't': [],
        'context': [],
        'belief': [],
        'gamma': [],
        'action': [],
        'reward_label': [],
        'choice_label': [],
        'hint_label': []
    }
    
    reversal_trials = set(env.reversal_schedule) if env.reversal_schedule else set()
    
    for t in range(T):
        # Agent step
        action_label, qs, q_pi, efe, gamma_t = runner.step(obs_ids, t)
        
        # Environment step
        obs_labels = env.step(action_label)
        obs_ids = runner.obs_labels_to_ids(obs_labels)
        
        # Log data
        logs['t'].append(t)
        logs['context'].append(env.context)
        logs['belief'].append(qs[0].copy())
        logs['gamma'].append(gamma_t)
        logs['action'].append(action_label)
        logs['hint_label'].append(obs_labels[0])
        logs['reward_label'].append(obs_labels[1])
        logs['choice_label'].append(obs_labels[2])
        
        # Conditional printing
        if verbose:
            print_trial_details(t, env, qs, gamma_t, action_label, obs_labels, show=True)
        elif print_around_reversals:
            near_reversal = any(abs(t - r) <= window for r in reversal_trials)
            if near_reversal or t < 20:
                print_trial_details(t, env, qs, gamma_t, action_label, obs_labels, show=True)
    
    return logs


def run_episode_with_ll(runner, env, T=200, verbose=False, initial_obs_labels=None):
    """
    Run episode and track per-trial log-likelihoods.
    
    Returns logs with additional 'll' field containing per-trial log-likelihoods.
    """
    if initial_obs_labels is None:
        initial_obs_labels = ['null', 'null', 'observe_start']
    
    obs_ids = runner.obs_labels_to_ids(initial_obs_labels)
    
    logs = {
        't': [],
        'context': [],
        'belief': [],
        'gamma': [],
        'action': [],
        'reward_label': [],
        'choice_label': [],
        'hint_label': [],
        'll': []
    }
    
    for t in range(T):
        # Agent step with LL tracking
        if hasattr(runner, 'step_with_ll'):
            action_label, qs, q_pi, efe, gamma_t, ll_t = runner.step_with_ll(obs_ids, t)
        else:
            action_label, qs, q_pi, efe, gamma_t = runner.step(obs_ids, t)
            ll_t = 0.0
        
        # Environment step
        obs_labels = env.step(action_label)
        obs_ids = runner.obs_labels_to_ids(obs_labels)
        
        # Log data
        logs['t'].append(t)
        logs['context'].append(env.context)
        logs['belief'].append(qs[0].copy())
        logs['gamma'].append(gamma_t)
        logs['action'].append(action_label)
        logs['hint_label'].append(obs_labels[0])
        logs['reward_label'].append(obs_labels[1])
        logs['choice_label'].append(obs_labels[2])
        logs['ll'].append(ll_t)

        # (method moved to class scope)
    
    return logs
