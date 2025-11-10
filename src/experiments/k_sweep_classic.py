"""
K-Sweep with Parameter Optimization

Finds optimal profile parameters for each K using parallel optimization.
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from functools import partial
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.experiment_config import *
from src.environment import TwoArmedBandit
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL, run_episode_with_ll
from src.utils import find_reversals, trial_accuracy


def params_to_profiles_and_Z(params, K):
    """
    Convert flat parameter vector to profiles and Z matrix.
    
    Parameter structure:
    - gammas: [0:K]
    - phi_strengths: [K:2K]
    - xi_hints: [2K:3K]
    - Z_params: [3K:5K-2] (2 states, K profiles, each row sums to 1)
    
    Total: 5K - 2 parameters
    """
    
    # Extract parameters
    gammas = params[0:K]
    phi_strengths = params[K:2*K]
    xi_hints = params[2*K:3*K]
    Z_raw = params[3*K:5*K-2]  # 2*(K-1) parameters for Z
    
    # Build profiles
    profiles = []
    for k in range(K):
        profile = {
            'phi_logits': [0.0, -phi_strengths[k], phi_strengths[k] / 2],
            'xi_logits': [0.0, xi_hints[k], 0.0, 0.0],
            'gamma': float(gammas[k])
        }
        profiles.append(profile)
    
    # Build Z matrix (2 states x K profiles)
    # Each row is a softmax over K-1 logits
    Z = np.zeros((2, K))
    
    if K == 1:
        Z = np.ones((2, 1))
    else:
        # Row 0 (left_better)
        Z_logits_0 = Z_raw[0:K-1]
        Z_row_0 = np.exp(np.concatenate([[0], Z_logits_0]))  # First element fixed at 0
        Z[0, :] = Z_row_0 / Z_row_0.sum()
        
        # Row 1 (right_better)
        Z_logits_1 = Z_raw[K-1:2*(K-1)]
        Z_row_1 = np.exp(np.concatenate([[0], Z_logits_1]))
        Z[1, :] = Z_row_1 / Z_row_1.sum()
    
    return profiles, Z


def objective_function(params, K, A, B, D, num_trials=100, num_runs=3, seed=None):
    """
    Objective function: negative log-likelihood (to minimize).
    
    Returns average negative LL across multiple runs.
    """
    
    try:
        # Convert params to profiles and Z
        profiles, Z = params_to_profiles_and_Z(params, K)
        
        # Get policies
        from pymdp.agent import Agent
        from pymdp import utils
        
        C_temp = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
        temp_agent = Agent(A=A, B=B, C=C_temp, D=D,
                         policy_len=2, inference_horizon=1,
                         control_fac_idx=[1], use_utility=True,
                         use_states_info_gain=True,
                         action_selection="stochastic", gamma=16)
        
        policies = temp_agent.policies
        num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
        
        # Create value function
        value_fn = make_value_fn('M3',
                                profiles=profiles,
                                Z=Z,
                                policies=policies,
                                num_actions_per_factor=num_actions_per_factor)
        
        # Run episodes and collect LL
        total_ll = 0.0
        
        for run in range(num_runs):
            run_seed = seed + run if seed is not None else None
            
            if run_seed is not None:
                np.random.seed(run_seed)
            
            # Create environment
            env = TwoArmedBandit(
                probability_hint=PROBABILITY_HINT,
                probability_reward=PROBABILITY_REWARD,
                reversal_schedule=DEFAULT_REVERSAL_SCHEDULE[:3]  # Fewer reversals for speed
            )
            
            # Create agent
            runner = AgentRunnerWithLL(A, B, D, value_fn,
                                OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                OBSERVATION_CHOICES, ACTION_CHOICES,
                                reward_mod_idx=1)
            
            # Run episode
            logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)
            
            # Accumulate LL
            total_ll += np.sum(logs['ll'])
        
        # Return negative average LL (to minimize)
        avg_ll = total_ll / num_runs
        return -avg_ll
        
    except Exception as e:
        # If anything fails, return a large penalty
        print(f"Error in objective: {e}")
        return 1e10


def optimize_K(K, A, B, D, num_trials=100, num_runs=3, seed=42, maxiter=50):
    """
    Optimize parameters for a specific K value.
    
    Uses differential evolution with parallel workers.
    """
    
    print(f"\n{'='*70}")
    print(f"Optimizing K={K} profiles")
    print(f"{'='*70}")
    
    # Define parameter bounds - WIDENED to reduce bias
    num_params = 5 * K - 2
    
    bounds = []
    
    # Gamma bounds: [0.2, 8.0] - wide range from exploratory to highly exploitative
    for k in range(K):
        bounds.append((0.2, 8.0))
    
    # Phi strength bounds: [0.5, 12.0] - very wide preference range
    for k in range(K):
        bounds.append((0.5, 12.0))
    
    # Xi hint bounds: [-2.0, 2.0] - wide information-seeking range
    for k in range(K):
        bounds.append((-2.0, 2.0))
    
    # Z matrix logit bounds: [-5.0, 5.0] - very wide assignment range
    for i in range(2 * (K - 1)):
        bounds.append((-5.0, 5.0))
    
    print(f"Number of parameters: {num_params}")
    print(f"Optimization bounds: {len(bounds)} parameters")
    
    # Create objective with fixed arguments
    obj_fn = partial(objective_function, K=K, A=A, B=B, D=D, 
                    num_trials=num_trials, num_runs=num_runs, seed=seed)
    
    # Run optimization
    print(f"Running differential evolution (maxiter={maxiter}, parallel)...")
    
    result = differential_evolution(
        obj_fn,
        bounds,
        maxiter=maxiter,
        popsize=15,  # Population size
        workers=-1,  # Use all CPU cores
        seed=seed,
        updating='deferred',  # Better for parallel
        polish=False,  # Skip local refinement for speed
        atol=0.01,
        tol=0.01,
        disp=True
    )
    
    print(f"\nOptimization complete!")
    print(f"Best negative LL: {result.fun:.2f}")
    print(f"Best LL: {-result.fun:.2f}")
    print(f"Function evaluations: {result.nfev}")
    
    # Extract best parameters
    best_params = result.x
    best_profiles, best_Z = params_to_profiles_and_Z(best_params, K)
    
    return {
        'K': K,
        'params': best_params,
        'profiles': best_profiles,
        'Z': best_Z,
        'neg_ll': result.fun,
        'll': -result.fun,
        'optimization_result': result
    }


def evaluate_optimized_model(opt_result, A, B, D, num_trials=200, num_runs=10, seed=42):
    """
    Evaluate optimized model on full task.
    """
    
    K = opt_result['K']
    profiles = opt_result['profiles']
    Z = opt_result['Z']
    
    print(f"\nEvaluating optimized K={K} model...")
    
    # Get policies
    from pymdp.agent import Agent
    from pymdp import utils
    
    C_temp = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
    temp_agent = Agent(A=A, B=B, C=C_temp, D=D,
                     policy_len=2, inference_horizon=1,
                     control_fac_idx=[1], use_utility=True,
                     use_states_info_gain=True,
                     action_selection="stochastic", gamma=16)
    
    policies = temp_agent.policies
    num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
    
    value_fn = make_value_fn('M3',
                            profiles=profiles,
                            Z=Z,
                            policies=policies,
                            num_actions_per_factor=num_actions_per_factor)
    
    # Run multiple episodes
    results = {
        'log_likelihoods': [],
        'accuracies': [],
        'total_rewards': [],
        'gamma_means': [],
    }
    
    for run in tqdm(range(num_runs), desc=f"  Evaluating K={K}"):
        run_seed = seed + run if seed is not None else None
        
        if run_seed is not None:
            np.random.seed(run_seed)
        
        env = TwoArmedBandit(
            probability_hint=PROBABILITY_HINT,
            probability_reward=PROBABILITY_REWARD,
            reversal_schedule=DEFAULT_REVERSAL_SCHEDULE
        )
        
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                            OBSERVATION_HINTS, OBSERVATION_REWARDS,
                            OBSERVATION_CHOICES, ACTION_CHOICES,
                            reward_mod_idx=1)
        
        logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)
        
        # Compute metrics
        acc = trial_accuracy(logs['action'], logs['context'])
        
        reward_values = []
        for reward_label in logs['reward_label']:
            if reward_label == 'observe_reward':
                reward_values.append(1)
            elif reward_label == 'observe_loss':
                reward_values.append(-1)
            else:
                reward_values.append(0)
        
        results['log_likelihoods'].append(np.sum(logs['ll']))
        results['accuracies'].append(acc.mean())
        results['total_rewards'].append(np.sum(reward_values))
        results['gamma_means'].append(np.mean(logs['gamma']))
    
    return results


def run_optimized_k_sweep(K_values=None, num_trials_opt=100, num_trials_eval=200, 
                          maxiter=50, seed=42):
    """
    Run K-sweep with parameter optimization for each K.
    """
    
    if K_values is None:
        K_values = [1, 2, 3, 4]  # Limit to 4 for computational efficiency
    
    print("="*70)
    print("OPTIMIZED K-SWEEP EXPERIMENT")
    print("="*70)
    print(f"K values: {K_values}")
    print(f"Optimization trials: {num_trials_opt}")
    print(f"Optimization iterations: {maxiter}")
    print(f"Evaluation trials: {num_trials_eval}")
    
    # Build shared components
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)
    
    all_results = {}
    
    # Optimize each K
    for K in K_values:
        # Optimize
        opt_result = optimize_K(K, A, B, D, 
                               num_trials=num_trials_opt, 
                               num_runs=3,
                               seed=seed,
                               maxiter=maxiter)
        
        # Evaluate on full task
        eval_results = evaluate_optimized_model(opt_result, A, B, D,
                                               num_trials=num_trials_eval,
                                               num_runs=10,
                                               seed=seed)
        
        # Combine results
        all_results[K] = {
            'optimization': opt_result,
            'evaluation': eval_results,
            'num_params': 5 * K - 2
        }
        
        # Print summary
        mean_ll = np.mean(eval_results['log_likelihoods'])
        mean_acc = np.mean(eval_results['accuracies'])
        
        print(f"\nK={K} Final Results:")
        print(f"  Mean LL: {mean_ll:.2f}")
        print(f"  Mean Accuracy: {mean_acc:.3f}")
        print(f"  Optimized profiles:")
        for i, p in enumerate(opt_result['profiles']):
            print(f"    Profile {i}: γ={p['gamma']:.2f}, φ={p['phi_logits'][1]:.2f}/{p['phi_logits'][2]:.2f}, ξ={p['xi_logits'][1]:.2f}")
        print(f"  Z matrix:")
        print(f"{opt_result['Z']}")
    
    # Print comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'K':<5} {'Params':<8} {'LL':<12} {'AIC':<12} {'BIC':<12} {'Acc':<8}")
    print("-"*70)
    
    for K in K_values:
        res = all_results[K]
        k_params = res['num_params']
        mean_ll = np.mean(res['evaluation']['log_likelihoods'])
        mean_aic = 2 * k_params - 2 * mean_ll
        mean_bic = k_params * np.log(num_trials_eval) - 2 * mean_ll
        mean_acc = np.mean(res['evaluation']['accuracies'])
        
        print(f"{K:<5} {k_params:<8} {mean_ll:<12.2f} {mean_aic:<12.2f} {mean_bic:<12.2f} {mean_acc:<8.3f}")
    
    # Find best
    bic_values = {K: all_results[K]['num_params'] * np.log(num_trials_eval) - 
                      2 * np.mean(all_results[K]['evaluation']['log_likelihoods'])
                  for K in K_values}
    
    best_K = min(bic_values, key=bic_values.get)
    
    print(f"\nBest K by BIC: {best_K}")
    
    return all_results


def main():
    """Main entry point."""
    
    # Run optimized K-sweep
    # Start with K=1,2,3,4 - can extend if needed
    results = run_optimized_k_sweep(
        K_values=[1, 2, 3, 4],
        num_trials_opt=100,
        num_trials_eval=200,
        maxiter=30,  # 30 iterations should be enough
        seed=42
    )
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
