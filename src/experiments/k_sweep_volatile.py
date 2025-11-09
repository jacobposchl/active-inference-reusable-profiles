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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.experiment_config import *
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL
from pymdp.agent import Agent
from pymdp import utils


class RegimeDependentBandit:
    """Bandit with regime-dependent hint reliability."""
    
    def __init__(self, regime_schedule=None, **kwargs):
        from src.environment import TwoArmedBandit
        kwargs.pop('probability_hint', None)
        self.bandit = TwoArmedBandit(probability_hint=0.7, **kwargs)
        self.regime_schedule = regime_schedule or []
        self.current_regime = None
        self.base_hint_prob = 0.7
        
    def get_current_regime(self):
        """Get current regime based on trial count."""
        current = None
        for regime in self.regime_schedule:
            if self.bandit.trial_count >= regime['start']:
                current = regime
        return current
    
    def step(self, action):
        """Override step to use regime-dependent hint probability."""
        regime = self.get_current_regime()
        if regime:
            self.bandit.probability_hint = regime.get('hint_prob', self.base_hint_prob)
            self.current_regime = regime['type']
            
            # Check for reversals
            if 'reversals' in regime:
                if self.bandit.trial_count in regime['reversals']:
                    self.bandit.context = ('right_better' if self.bandit.context == 'left_better' 
                                          else 'left_better')
        
        return self.bandit.step(action)


def create_regime_schedule():
    """Create regime schedule with varying hint reliability."""
    return [
        {'start': 0, 'type': 'stable', 'hint_prob': 0.55, 'reversals': [60]},
        {'start': 120, 'type': 'volatile', 'hint_prob': 0.95, 
         'reversals': [128, 136, 144, 152, 160, 168, 176, 184]},
        {'start': 200, 'type': 'stable', 'hint_prob': 0.55, 'reversals': [260]},
        {'start': 320, 'type': 'volatile', 'hint_prob': 0.95, 
         'reversals': [328, 336, 344, 352, 360, 368, 376, 384]},
    ]


def params_to_profiles_and_Z(params, K):
    """Convert flat parameter vector to profiles and Z matrix."""
    
    # Unpack parameters
    gamma_params = params[0:K]
    phi_params = params[K:3*K]
    xi_params = params[3*K:4*K]
    
    # Build profiles
    profiles = []
    for k in range(K):
        profile = {
            'gamma': float(gamma_params[k]),
            'phi_logits': [0.0, float(phi_params[2*k]), float(phi_params[2*k+1])],
            'xi_logits': [0.0, float(xi_params[k]), 0.0, 0.0]
        }
        profiles.append(profile)
    
    # Build Z matrix (optimized separately)
    if K == 1:
        Z = np.ones((2, 1))
    else:
        Z_raw = params[4*K:4*K+2*(K-1)]
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
    """Objective: Maximize log-likelihood on variable volatility task."""
    
    try:
        profiles, Z = params_to_profiles_and_Z(params, K)
        
        value_fn = make_value_fn('M3',
                                profiles=profiles,
                                Z=Z,
                                policies=policies,
                                num_actions_per_factor=num_actions_per_factor)
        
        total_ll = 0.0
        
        for run in range(num_runs):
            seed = seed_base + run
            np.random.seed(seed)
            
            env = RegimeDependentBandit(
                regime_schedule=regime_schedule,
                probability_reward=PROBABILITY_REWARD
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
                obs = env.step(action)
        
        mean_ll = total_ll / (num_runs * num_trials)
        return -mean_ll  # Minimize negative LL
        
    except Exception as e:
        print(f"Error in objective: {e}")
        return 1e6


def optimize_K_volatile(K, A, B, D, num_trials=400, num_runs=3, seed=42, maxiter=30):
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
    num_params = 4*K + 2*(K-1) if K > 1 else 4*K
    
    bounds = []
    # Gamma bounds (policy precision)
    for _ in range(K):
        bounds.append((0.2, 8.0))
    # Phi bounds (outcome preferences)
    for _ in range(2*K):
        bounds.append((-12.0, 6.0))
    # Xi bounds (action preferences - hint seeking)
    for _ in range(K):
        bounds.append((-5.0, 5.0))
    # Z bounds (assignment logits)
    for _ in range(2*(K-1)) if K > 1 else []:
        bounds.append((-3.0, 3.0))
    
    print(f"Total parameters: {num_params}")
    print(f"Bounds: {len(bounds)}")
    print(f"Running optimization with {num_runs} runs per evaluation...")
    print()
    
    # Run optimization
    result = differential_evolution(
        objective_function,
        bounds,
        args=(K, A, B, D, policies, num_actions_per_factor, 
              regime_schedule, num_trials, num_runs, seed),
        maxiter=maxiter,
        popsize=15,
        workers=-1,
        updating='deferred',
        seed=seed,
        disp=True
    )
    
    print(f"\nOptimization complete!")
    print(f"Best negative LL: {result.fun:.4f}")
    print(f"Best LL: {-result.fun:.4f}")
    
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
    total_acc = 0.0
    stable_hint_usage = []
    volatile_hint_usage = []
    
    for run in tqdm(range(num_runs), desc=f"Evaluating K={K}"):
        np.random.seed(seed + run)
        
        env = RegimeDependentBandit(
            regime_schedule=regime_schedule,
            probability_reward=PROBABILITY_REWARD
        )
        
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                            OBSERVATION_HINTS, OBSERVATION_REWARDS,
                            OBSERVATION_CHOICES, ACTION_CHOICES,
                            reward_mod_idx=1)
        
        actions = []
        contexts = []
        regimes = []
        lls = []
        
        obs = env.step('act_start')
        for t in range(num_trials):
            obs_ids = runner.obs_labels_to_ids(obs)
            action, qs, q_pi, efe, gamma, ll = runner.step_with_ll(obs_ids, t)
            
            actions.append(action)
            contexts.append(env.bandit.context)
            regimes.append(env.current_regime or 'stable')
            lls.append(ll)
            total_ll += ll
            
            obs = env.step(action)
        
        # Compute accuracy
        correct = sum(1 for a, c in zip(actions, contexts) if
                     (a == 'act_left' and c == 'left_better') or
                     (a == 'act_right' and c == 'right_better'))
        total_acc += correct / len(actions)
        
        # Hint usage by regime
        stable_hints = sum(1 for a, r in zip(actions, regimes) if a == 'act_hint' and r == 'stable')
        volatile_hints = sum(1 for a, r in zip(actions, regimes) if a == 'act_hint' and r == 'volatile')
        stable_trials = sum(1 for r in regimes if r == 'stable')
        volatile_trials = sum(1 for r in regimes if r == 'volatile')
        
        stable_hint_usage.append(stable_hints / stable_trials if stable_trials > 0 else 0)
        volatile_hint_usage.append(volatile_hints / volatile_trials if volatile_trials > 0 else 0)
    
    mean_ll = total_ll / (num_runs * num_trials)
    mean_acc = total_acc / num_runs
    
    print(f"\nK={K} Final Results:")
    print(f"  Mean LL: {mean_ll:.2f}")
    print(f"  Mean Accuracy: {mean_acc:.3f}")
    print(f"  Stable Hint Usage: {np.mean(stable_hint_usage):.3f} ± {np.std(stable_hint_usage):.3f}")
    print(f"  Volatile Hint Usage: {np.mean(volatile_hint_usage):.3f} ± {np.std(volatile_hint_usage):.3f}")
    print(f"  Optimized profiles:")
    for k, prof in enumerate(profiles):
        print(f"    Profile {k}: γ={prof['gamma']:.2f}, φ={prof['phi_logits'][1]:.2f}/{prof['phi_logits'][2]:.2f}, ξ={prof['xi_logits'][1]:.2f}")
    print(f"  Z matrix:")
    print(Z)
    
    return {
        'll': mean_ll,
        'accuracy': mean_acc,
        'stable_hints': np.mean(stable_hint_usage),
        'volatile_hints': np.mean(volatile_hint_usage)
    }


def main():
    """Run K-sweep optimization on variable volatility task."""
    
    print("="*70)
    print("K-SWEEP OPTIMIZATION: VARIABLE VOLATILITY TASK")
    print("="*70)
    print("Task: Regime-dependent hint reliability")
    print("  Stable: 55% hint accuracy")
    print("  Volatile: 95% hint accuracy")
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
                                        num_runs=3, 
                                        seed=42,
                                        maxiter=30)
        
        eval_result = evaluate_optimized_volatile(opt_result, A, B, D,
                                                  num_trials=400,
                                                  num_runs=10,
                                                  seed=100)
        
        results[K] = {**opt_result, **eval_result}
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON ACROSS K")
    print("="*70)
    print(f"{'K':<5} {'LL':<10} {'Accuracy':<12} {'Stable Hints':<15} {'Volatile Hints':<15}")
    print("-"*70)
    for K in [1, 2, 3]:
        r = results[K]
        print(f"{K:<5} {r['ll']:<10.2f} {r['accuracy']:<12.3f} {r['stable_hints']:<15.3f} {r['volatile_hints']:<15.3f}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
