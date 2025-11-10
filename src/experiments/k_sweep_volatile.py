"""
K-sweep optimization for Variable Volatility task.

This task has regime-dependent hint reliability:
- Stable: 55% hint accuracy, slow reversals
- Volatile: 95% hint accuracy, fast reversals

Different regimes should benefit from different profiles.
"""

import numpy as np
import sys
import os
from scipy.optimize import differential_evolution
from tqdm import tqdm
import multiprocessing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.experiment_config import *
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL
from pymdp.agent import Agent
from pymdp import utils


class RegimeDependentBandit:
    """
    Bandit with regime-dependent hint reliability AND reward probability.
    
    Key idea: Different regimes have different trade-offs between
    exploration (hints) and exploitation (direct choices).
    """
    
    def __init__(self, regime_schedule=None, **kwargs):
        from src.environment import TwoArmedBandit
        kwargs.pop('probability_hint', None)
        kwargs.pop('probability_reward', None)
        self.bandit = TwoArmedBandit(probability_hint=0.7, probability_reward=0.8, **kwargs)
        self.regime_schedule = regime_schedule or []
        self.current_regime = None
        
    def get_current_regime(self):
        """Get current regime based on trial count."""
        current = None
        for regime in self.regime_schedule:
            if self.bandit.trial_count >= regime['start']:
                current = regime
        return current
    
    def step(self, action):
        """Override step to use regime-dependent parameters and apply penalties."""
        regime = self.get_current_regime()
        if regime:
            # Update regime-specific parameters
            self.bandit.probability_hint = regime.get('hint_prob', 0.7)
            self.bandit.probability_reward = regime.get('reward_prob', 0.8)
            self.current_regime = regime['type']
            
            # Check for reversals
            if 'reversals' in regime:
                if self.bandit.trial_count in regime['reversals']:
                    self.bandit.context = ('right_better' if self.bandit.context == 'left_better' 
                                          else 'left_better')
        
        # Take the step
        obs = self.bandit.step(action)
        
        # Apply penalty for wrong choices in volatile regime
        if regime and regime.get('penalty', 0) != 0:
            # If action was a choice (left=1 or right=2), check if it was wrong
            if action in [1, 2]:
                # Check reward observation (modality 1, outcome 0=loss, 1=reward)
                if obs[1] == 0:  # Got loss (wrong choice)
                    # Apply penalty by modifying the reward observation
                    # We'll track this separately for objective function
                    pass  # Penalty handled in objective function
        
        return obs


def create_regime_schedule():
    """
    REVISED: Make regimes differ in INFORMATION SOURCE reliability
    
    STABLE regime (feedback-based learning):
    - Hint accuracy: 55% (useless)
    - Reward probability: 85% (reliable feedback signal)
    - Rare reversals: Learn from feedback
    → OPTIMAL: Ignore hints, learn from reward feedback
    
    VOLATILE regime (hint-based learning):
    - Hint accuracy: 95% (very reliable)
    - Reward probability: 0% (NO FEEDBACK - can't tell if choice was correct!)
    - Frequent reversals: Can't learn from uninformative feedback
    → OPTIMAL: Must use hints because feedback is useless
    """
    return [
        {'start': 0, 'type': 'stable', 
         'hint_prob': 0.55, 'reward_prob': 0.85, 
         'penalty': -1.0,
         'reversals': [60]},
        
        {'start': 120, 'type': 'volatile', 
         'hint_prob': 0.95, 'reward_prob': 0.0,  # ← KEY CHANGE!
         'penalty': -2.0,
         'reversals': [128, 136, 144, 152, 160, 168, 176, 184]},
        
        {'start': 200, 'type': 'stable', 
         'hint_prob': 0.55, 'reward_prob': 0.85,
         'penalty': -1.0,
         'reversals': [260]},
        
        {'start': 320, 'type': 'volatile', 
         'hint_prob': 0.95, 'reward_prob': 0.0,  # ← KEY CHANGE!
         'penalty': -2.0,
         'reversals': [328, 336, 344, 352, 360, 368, 376, 384]},
    ]


def params_to_profiles_and_Z(params, K):
    """Convert flat parameter vector to profiles and Z matrix.
    
    Parameter layout:
    - gamma: K params (policy precision per profile)
    - phi: 2K params (2 outcome preferences per profile: loss, reward)
    - xi: 3K params (3 action preferences per profile: hint, left, right)
    - Z: 2(K-1) params (assignment matrix logits)
    
    Total: K + 2K + 3K + 2(K-1) = 8K - 2 for K>1, or 6K for K=1
    """
    
    # Unpack parameters
    gamma_params = params[0:K]
    phi_params = params[K:3*K]
    xi_params = params[3*K:6*K]  # NOW 3 per profile!
    
    # Build profiles
    profiles = []
    for k in range(K):
        profile = {
            'gamma': float(gamma_params[k]),
            'phi_logits': [0.0, float(phi_params[2*k]), float(phi_params[2*k+1])],
            'xi_logits': [0.0, 
                         float(xi_params[3*k]),      # hint bias
                         float(xi_params[3*k + 1]),  # left bias
                         float(xi_params[3*k + 2])]  # right bias
        }
        profiles.append(profile)
    
    # Build Z matrix (optimized separately)
    if K == 1:
        Z = np.ones((2, 1))
    else:
        Z_raw = params[6*K:6*K+2*(K-1)]  # Adjusted offset
        Z = np.zeros((2, K))
        
        # Row 0: left_better context
        Z_logits_0 = Z_raw[0:K-1]
        Z_row_0 = np.exp(np.concatenate([[0], Z_logits_0]))
        Z[0, :] = Z_row_0 / Z_row_0.sum()
        
        # Row 1: right_better context
        Z_logits_1 = Z_raw[K-1:2*(K-1)]
        Z_row_1 = np.exp(np.concatenate([[0], Z_logits_1]))
        Z[1, :] = Z_row_1 / Z_row_1.sum()
    
    return profiles, Z


def objective_function(params, K, A, B, D, policies, num_actions_per_factor, 
                      regime_schedule, num_trials, num_runs, seed_base):
    """
    Objective: Maximize CHOICE accuracy while discouraging hint-only solutions.
    
    Returns:
    --------
    objective : float
        Combined score = -accuracy - 0.1*choice_ratio + 0.01*mean_ll
        (lower is better; optimizer minimizes this)
        
    Primary goal: maximize accuracy on left/right choices
    Secondary goal: encourage actual choices (not just hints)
    Tertiary goal: maintain reasonable log-likelihood
    """
    
    try:
        profiles, Z = params_to_profiles_and_Z(params, K)
        
        value_fn = make_value_fn('M3',
                                profiles=profiles,
                                Z=Z,
                                policies=policies,
                                num_actions_per_factor=num_actions_per_factor)
        
        total_ll = 0.0
        total_rewards = 0  # Track cumulative rewards (including penalties)
        total_choices = 0  # left/right actions
        total_trials = 0
        
        for run in range(num_runs):
            seed = seed_base + run
            np.random.seed(seed)
            
            env = RegimeDependentBandit(
                regime_schedule=regime_schedule
            )
            
            runner = AgentRunnerWithLL(A, B, D, value_fn,
                                OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                OBSERVATION_CHOICES, ACTION_CHOICES,
                                reward_mod_idx=1)
            
            # Run episode
            obs = env.step('act_start')
            for t in range(num_trials):
                obs_ids = runner.obs_labels_to_ids(obs)
                action, qs, q_pi, efe, gamma, ll = runner.step_with_ll(obs_ids, t)
                total_ll += ll
                total_trials += 1
                
                # Track choices and rewards
                if action in ['act_left', 'act_right']:
                    total_choices += 1
                
                # Take action and get observation
                obs = env.step(action)
                
                # Track rewards (with penalties in volatile regime)
                regime = env.get_current_regime()
                if obs[1] == 1:  # Reward outcome
                    total_rewards += 1
                elif obs[1] == 0 and action in ['act_left', 'act_right']:  # Loss from choice
                    # Apply penalty if in penalty regime
                    if regime and regime.get('penalty', 0) != 0:
                        total_rewards += regime['penalty']  # penalty is negative, e.g., -5.0
                    else:
                        total_rewards -= 1  # Standard loss
        
        mean_ll = total_ll / total_trials
        
        # Penalize hint-only solutions severely
        if total_choices == 0:
            return 1e6
        
        # Calculate metrics
        choice_ratio = total_choices / total_trials
        
        # Require minimum choice ratio (at least 30% of trials should be choices)
        if choice_ratio < 0.3:
            penalty = 100.0 * (0.3 - choice_ratio)
            return 1e6 + penalty
        
        # Primary objective: maximize total rewards (negate because we minimize)
        # Secondary: small LL regularization to avoid degenerate solutions
        objective = -total_rewards + 0.01 * (-mean_ll)
        
        return objective
        
    except Exception as e:
        print(f"Error in objective: {e}")
        return 1e6


def optimize_K_volatile(K, A, B, D, num_trials=400, num_runs=3, seed=42, maxiter=15):
    """Optimize K profiles for variable volatility task."""
    
    print(f"\n{'='*70}")
    print(f"Optimizing K={K} profiles for VARIABLE VOLATILITY task")
    print(f"{'='*70}")
    
    # Get policies
    C_temp = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
    temp_agent = Agent(A=A, B=B, C=C_temp, D=D,
                     policy_len=2, inference_horizon=1,
                     control_fac_idx=[1], use_utility=True,
                     use_states_info_gain=True,
                     action_selection="stochastic", gamma=16)
    
    policies = temp_agent.policies
    num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
    regime_schedule = create_regime_schedule()
    
    # Parameter bounds
    num_params = 6*K + 2*(K-1) if K > 1 else 6*K  # Updated for 3*K xi params
    
    bounds = []
    # Gamma bounds (policy precision)
    for _ in range(K):
        bounds.append((0.2, 8.0))
    # Phi bounds (outcome preferences)
    # For each profile: [phi_loss, phi_reward]
    # Force phi_reward to be non-negative so the agent values reward over null/loss
    for _ in range(K):
        bounds.append((-12.0, 0.0))  # phi_loss
        bounds.append((0.0, 6.0))    # phi_reward (prefer non-negative)
    # Xi bounds (action preferences - hint, left, right for each profile)
    for k in range(K):
        bounds.append((-2.0, 3.0))  # hint bias: NOW can be positive!
        bounds.append((-1.5, 1.5))  # left bias
        bounds.append((-1.5, 1.5))  # right bias
    # Z bounds (assignment logits)
    for _ in range(2*(K-1)) if K > 1 else []:
        bounds.append((-3.0, 3.0))
    
    print(f"Total parameters: {num_params}")
    print(f"Bounds: {len(bounds)}")
    print(f"Running optimization with {num_runs} runs per evaluation...")
    print()
    
    # Use all cores except 1 (leave one free for other tasks)
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} parallel workers (leaving 1 core free)")
    print()
    
    # Run optimization
    result = differential_evolution(
        objective_function,
        bounds,
        args=(K, A, B, D, policies, num_actions_per_factor, 
              regime_schedule, num_trials, num_runs, seed),
        maxiter=maxiter,
        popsize=10,  # Reduced from 15 for speed (still good coverage)
        workers=num_workers,  # All cores minus 1
        updating='deferred',  # Parallel-friendly update strategy
        seed=seed,
        disp=True,
        polish=False # Skip polishing to save time
    )
    
    print(f"\nOptimization complete!")
    print(f"Best objective value: {result.fun:.4f}")
    print(f"  (Note: objective = -total_rewards + 0.01*(-mean_ll), lower is better)")
    print(f"  This prioritizes: 1) total rewards earned, 2) reasonable LL")
    
    # Extract best parameters
    best_params = result.x
    best_profiles, best_Z = params_to_profiles_and_Z(best_params, K)
    
    return {
        'K': K,
        'profiles': best_profiles,
        'Z': best_Z,
        'params': best_params,
        'neg_ll': result.fun,
        'll': -result.fun
    }


def evaluate_optimized_volatile(opt_result, A, B, D, num_trials=400, num_runs=10, seed=42):
    """Evaluate optimized model on longer runs."""
    
    K = opt_result['K']
    profiles = opt_result['profiles']
    Z = opt_result['Z']
    
    print(f"\nEvaluating optimized K={K} model on variable volatility...")
    
    # Get policies
    C_temp = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
    temp_agent = Agent(A=A, B=B, C=C_temp, D=D,
                     policy_len=2, inference_horizon=1,
                     control_fac_idx=[1], use_utility=True,
                     use_states_info_gain=True,
                     action_selection="stochastic", gamma=16)
    
    policies = temp_agent.policies
    num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
    regime_schedule = create_regime_schedule()
    
    value_fn = make_value_fn('M3',
                            profiles=profiles,
                            Z=Z,
                            policies=policies,
                            num_actions_per_factor=num_actions_per_factor)
    
    total_ll = 0.0
    total_rewards = 0.0
    total_choice_ratio = 0.0
    stable_hint_usage = []
    volatile_hint_usage = []
    stable_rewards = []
    volatile_rewards = []
    
    for run in tqdm(range(num_runs), desc=f"Evaluating K={K}"):
        np.random.seed(seed + run)
        
        env = RegimeDependentBandit(
            regime_schedule=regime_schedule
        )
        
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                            OBSERVATION_HINTS, OBSERVATION_REWARDS,
                            OBSERVATION_CHOICES, ACTION_CHOICES,
                            reward_mod_idx=1)
        
        actions = []
        rewards = []
        regimes = []
        lls = []
        
        obs = env.step('act_start')
        for t in range(num_trials):
            obs_ids = runner.obs_labels_to_ids(obs)
            action, qs, q_pi, efe, gamma, ll = runner.step_with_ll(obs_ids, t)
            
            actions.append(action)
            regimes.append(env.current_regime or 'stable')
            lls.append(ll)
            total_ll += ll
            
            obs = env.step(action)
            
            # Track rewards
            reward = 1 if obs[1] == 'observe_reward' else 0
            rewards.append(reward)
        
        # Compute metrics
        total_rewards += sum(rewards)
        
        choices = [a for a in actions if a in ['act_left', 'act_right']]
        total_choice_ratio += len(choices) / len(actions)
        
        # Hint usage by regime
        stable_hints = sum(1 for a, r in zip(actions, regimes) if a == 'act_hint' and r == 'stable')
        volatile_hints = sum(1 for a, r in zip(actions, regimes) if a == 'act_hint' and r == 'volatile')
        stable_trials = sum(1 for r in regimes if r == 'stable')
        volatile_trials = sum(1 for r in regimes if r == 'volatile')
        
        stable_hint_usage.append(stable_hints / stable_trials if stable_trials > 0 else 0)
        volatile_hint_usage.append(volatile_hints / volatile_trials if volatile_trials > 0 else 0)
        
        # Rewards by regime
        stable_reward_total = sum(r for r, reg in zip(rewards, regimes) if reg == 'stable')
        volatile_reward_total = sum(r for r, reg in zip(rewards, regimes) if reg == 'volatile')
        stable_rewards.append(stable_reward_total / stable_trials if stable_trials > 0 else 0)
        volatile_rewards.append(volatile_reward_total / volatile_trials if volatile_trials > 0 else 0)
    
    mean_ll = total_ll / (num_runs * num_trials)
    mean_rewards = total_rewards / num_runs
    mean_choice_ratio = total_choice_ratio / num_runs
    
    print(f"\nK={K} Final Results:")
    print(f"  Mean LL: {mean_ll:.2f}")
    print(f"  Total Rewards: {mean_rewards:.1f} / {num_trials} trials")
    print(f"  Reward Rate: {mean_rewards/num_trials:.3f}")
    print(f"  Choice Ratio: {mean_choice_ratio:.3f} (fraction of trials with left/right)")
    print(f"  Stable Hint Usage: {np.mean(stable_hint_usage):.3f} ± {np.std(stable_hint_usage):.3f}")
    print(f"  Volatile Hint Usage: {np.mean(volatile_hint_usage):.3f} ± {np.std(volatile_hint_usage):.3f}")
    print(f"  Stable Reward Rate: {np.mean(stable_rewards):.3f} ± {np.std(stable_rewards):.3f}")
    print(f"  Volatile Reward Rate: {np.mean(volatile_rewards):.3f} ± {np.std(volatile_rewards):.3f}")
    print(f"  Optimized profiles:")
    for k, prof in enumerate(profiles):
        xi = prof['xi_logits']
        print(f"    Profile {k}: γ={prof['gamma']:.2f}, φ={prof['phi_logits'][1]:.2f}/{prof['phi_logits'][2]:.2f}, "
              f"ξ_hint={xi[1]:.2f}, ξ_left={xi[2]:.2f}, ξ_right={xi[3]:.2f}")
    print(f"  Z matrix:")
    print(Z)
    
    return {
        'll': mean_ll,
        'total_rewards': mean_rewards,
        'reward_rate': mean_rewards / num_trials,
        'choice_ratio': mean_choice_ratio,
        'stable_hints': np.mean(stable_hint_usage),
        'volatile_hints': np.mean(volatile_hint_usage),
        'stable_reward_rate': np.mean(stable_rewards),
        'volatile_reward_rate': np.mean(volatile_rewards)
    }


def main():
    """Run K-sweep optimization on variable volatility task."""
    
    print("="*70)
    print("K-SWEEP OPTIMIZATION: VARIABLE VOLATILITY TASK")
    print("="*70)
    print("Task: Regime-dependent TRADE-OFFS between exploration and exploitation")
    print()
    print("  STABLE regime (exploitation-favored):")
    print("    - Hint accuracy: 55% (barely useful)")
    print("    - Reward probability: 90% (easy to learn from feedback)")
    print("    - Reversals: Rare (every 60 trials)")
    print("    → Optimal: Ignore hints, exploit learned preferences")
    print()
    print("  VOLATILE regime (exploration-favored):")
    print("    - Hint accuracy: 95% (very reliable)")
    print("    - Reward probability: 65% (noisy feedback)")
    print("    - Reversals: Frequent (every 8 trials)")
    print("    → Optimal: Use hints to track rapid changes")
    print()
    
    # Build generative model
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)
    
    # Optimize for different K
    results = {}
    for K in [1, 2, 3]:
        opt_result = optimize_K_volatile(K, A, B, D, 
                                        num_trials=400, 
                                        num_runs=3,  # Quick optimization
                                        seed=42,
                                        maxiter=3)  # Reduced for quick test
        
        eval_result = evaluate_optimized_volatile(opt_result, A, B, D,
                                                  num_trials=400,
                                                  num_runs=20,  # Full evaluation
                                                  seed=100)
        
        results[K] = {**opt_result, **eval_result}
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON ACROSS K")
    print("="*70)
    print(f"{'K':<5} {'LL':<10} {'Rewards':<12} {'Rate':<10} {'Stable Hints':<15} {'Volatile Hints':<15}")
    print("-"*70)
    for K in [1, 2, 3]:
        r = results[K]
        print(f"{K:<5} {r['ll']:<10.2f} {r['total_rewards']:<12.1f} {r['reward_rate']:<10.3f} "
              f"{r['stable_hints']:<15.3f} {r['volatile_hints']:<15.3f}")
    
    print("\n" + "="*70)
    print("REGIME-SPECIFIC PERFORMANCE")
    print("="*70)
    print(f"{'K':<5} {'Stable Reward Rate':<20} {'Volatile Reward Rate':<20}")
    print("-"*70)
    for K in [1, 2, 3]:
        r = results[K]
        print(f"{K:<5} {r['stable_reward_rate']:<20.3f} {r['volatile_reward_rate']:<20.3f}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
