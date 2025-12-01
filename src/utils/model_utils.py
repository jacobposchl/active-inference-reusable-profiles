import numpy as np
from config.experiment_config import *
from src.models import make_value_fn


def create_model(model_name, A, B, D):
    """Create value function for specified model."""
    if model_name == 'M1':
        value_fn = make_value_fn('M1', **M1_DEFAULTS)

    elif model_name == 'M2':
        # gamma_schedule receives H (entropy of better_arm beliefs) directly
        def gamma_schedule(H_better_arm, t, g_base=M2_DEFAULTS['gamma_base'], 
                          k=M2_DEFAULTS['entropy_k']):
            return g_base / (1.0 + k * H_better_arm)
        
        value_fn = make_value_fn('M2', 
                                C_reward_logits=M2_DEFAULTS['C_reward_logits'],
                                gamma_schedule=gamma_schedule)
        
    elif model_name == 'M3':
        # Get policies from temporary agent
        # With 3 state factors: [context, better_arm, choice]
        # Only choice (index 2) is controllable
        from pymdp.agent import Agent
        from pymdp import utils
        
        C_temp = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
        temp_agent = Agent(A=A, B=B, C=C_temp, D=D,
                         policy_len=2, inference_horizon=1,
                         control_fac_idx=[2],  # Choice is now factor 2
                         use_utility=True,
                         use_states_info_gain=True,
                         action_selection="stochastic", gamma=16)
        
        policies = temp_agent.policies
        # 3 action factors: [context_action, better_arm_action, choice_action]
        num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_BETTER_ARM), len(ACTION_CHOICES)]
        
        value_fn = make_value_fn('M3',
                                profiles=M3_DEFAULTS['profiles'],
                                Z=np.array(M3_DEFAULTS['Z']),
                                policies=policies,
                                num_actions_per_factor=num_actions_per_factor)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return value_fn


def get_num_parameters(model_name):
    """Get number of free parameters for each model (actually optimized in grid search).
    
    Note: This counts only parameters that are fitted during recovery, not fixed defaults.
    """
    if model_name == 'M1':
        return 1  # gamma only (C_reward_logits are fixed)
        
    elif model_name == 'M2':
        return 2  # gamma_base and entropy_k (C_reward_logits are fixed)
        
    elif model_name == 'M3':
        # M3 optimizes 4 parameters:
        # - gamma_p0: policy precision for profile 0 (volatile/exploratory)
        # - gamma_p1: policy precision for profile 1 (stable/exploitative)
        # - hint_scale: shared scaling factor for hint action preferences (applied to both profiles)
        # - arm_scale: shared scaling factor for left/right arm preferences (applied to both profiles)
        # Note: phi_logits (outcome preferences) are fixed. xi scales are shared parameters that multiply
        # each profile's base xi_logits (e.g., profile 0 base hint=3.0, profile 1 base hint=0.5).
        return 4
    return 0


def compute_metrics(logs):
    """Compute performance metrics from episode logs."""
    # Accuracy
    from src.utils.helpers import trial_accuracy
    acc = trial_accuracy(logs['action'], logs['context'])
    
    # Reversals
    from src.utils.helpers import find_reversals
    reversals = find_reversals(logs['context'])
    
    # Rewards - convert labels to numeric values
    reward_values = []
    for reward_label in logs['reward_label']:
        if reward_label == 'observe_reward':
            reward_values.append(1)
        elif reward_label == 'observe_loss':
            reward_values.append(-1)
        else:  # 'null'
            reward_values.append(0)
    total_reward = np.sum(reward_values)
    
    # Log-likelihood
    log_likelihood = np.sum(logs['ll'])
    num_trials = len(logs['ll'])
    
    # Adaptation speed (trials to recover after reversal)
    adaptation_times = []
    for rev_t in reversals:
        window = min(20, len(acc) - rev_t)
        if window > 5:
            post_rev_acc = acc[rev_t:rev_t+window]
            above_threshold = np.where(post_rev_acc > 0.7)[0]
            if len(above_threshold) > 0:
                adaptation_times.append(above_threshold[0])
    
    avg_adaptation = np.mean(adaptation_times) if adaptation_times else np.nan
    
    # Gamma statistics
    gamma_mean = np.mean(logs['gamma'])
    gamma_std = np.std(logs['gamma'])
    
    return {
        'accuracy': acc,
        'mean_accuracy': acc.mean(),
        'total_reward': total_reward,
        'reversals': reversals,
        'adaptation_time': avg_adaptation,
        'gamma_mean': gamma_mean,
        'gamma_std': gamma_std,
        'log_likelihood': log_likelihood,
        'num_trials': num_trials,
        'logs': logs
    }
