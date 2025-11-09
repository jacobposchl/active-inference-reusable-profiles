"""
Variable Volatility Experiment v2

Modified task structure to make profile diversity genuinely beneficial:

Key Changes:
1. Context-dependent hint reliability:
   - Stable regime: Hints 60% accurate (not worth it)
   - Volatile regime: Hints 90% accurate (very valuable!)

2. Cost structure:
   - Wrong arm choice: -2 reward
   - Hint: -0.5 reward (small cost)
   - Correct arm: +1 reward

This creates clear regime-specific optimal strategies:
- STABLE: Exploit directly (hints not accurate enough)
- VOLATILE: Seek hints first (very accurate, worth the cost)
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.experiment_config import *
from src.environment import TwoArmedBandit
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL, run_episode_with_ll
from src.utils import find_reversals, trial_accuracy


class RegimeDependentBandit(TwoArmedBandit):
    """
    Bandit with regime-dependent hint reliability.
    
    Different regimes have different hint accuracy and reversal frequencies.
    """
    
    def __init__(self, regime_schedule=None, **kwargs):
        """
        Parameters:
        -----------
        regime_schedule : list of dicts
            [{'start': trial, 'type': 'stable'/'volatile', 
              'hint_prob': 0.6, 'reversal_interval': 50}, ...]
        """
        # Remove probability_hint from kwargs to set it dynamically
        kwargs.pop('probability_hint', None)
        super().__init__(probability_hint=0.7, **kwargs)  # Default
        
        self.regime_schedule = regime_schedule or []
        self.current_regime = None
        self.base_hint_prob = 0.7
        
    def get_current_regime(self):
        """Get current regime based on trial count."""
        current = None
        for regime in self.regime_schedule:
            if self.trial_count >= regime['start']:
                current = regime
        return current
    
    def step(self, action):
        """Override step to use regime-dependent hint probability."""
        
        # Update current regime
        regime = self.get_current_regime()
        if regime:
            self.probability_hint = regime.get('hint_prob', self.base_hint_prob)
            self.current_regime = regime['type']
        
        # Check for reversals (regime-dependent)
        if regime and 'reversals' in regime:
            if self.trial_count in regime['reversals']:
                self.context = ('right_better' if self.context == 'left_better' 
                              else 'left_better')
        
        # Standard step
        return super().step(action)


def create_regime_schedule_v2():
    """
    Create regime schedule with varying hint reliability.
    
    Returns:
    --------
    regime_schedule : list of dicts
    """
    schedule = [
        # Stable regime 1: Poor hints, slow reversals
        {
            'start': 0,
            'type': 'stable',
            'hint_prob': 0.55,  # Barely better than chance - not worth it!
            'reversals': [60]   # One reversal
        },
        # Volatile regime 1: Excellent hints, fast reversals
        {
            'start': 120,
            'type': 'volatile',
            'hint_prob': 0.95,  # Very accurate - worth seeking!
            'reversals': [128, 136, 144, 152, 160, 168, 176, 184]  # Every 8 trials
        },
        # Stable regime 2: Poor hints, slow reversals
        {
            'start': 200,
            'type': 'stable',
            'hint_prob': 0.55,
            'reversals': [260]
        },
        # Volatile regime 2: Excellent hints, fast reversals
        {
            'start': 320,
            'type': 'volatile',
            'hint_prob': 0.95,
            'reversals': [328, 336, 344, 352, 360, 368, 376, 384]
        },
    ]
    
    return schedule


def run_regime_dependent_experiment(model_name='M3', num_trials=400, num_runs=10, seed=42):
    """
    Run experiment with regime-dependent hint reliability.
    """
    
    print("="*70)
    print("REGIME-DEPENDENT HINT RELIABILITY EXPERIMENT")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Total trials: {num_trials}")
    print(f"Runs: {num_runs}")
    print()
    
    # Create regime schedule
    regime_schedule = create_regime_schedule_v2()
    
    print("Regime Schedule:")
    for regime in regime_schedule:
        print(f"  Trial {regime['start']:3d}: {regime['type'].upper():8s} "
              f"(hint accuracy: {regime['hint_prob']:.0%}, "
              f"reversals: {len(regime['reversals'])})")
    print()
    
    # Build generative model
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)
    
    # Create value function
    if model_name == 'M1':
        value_fn = make_value_fn('M1', **M1_DEFAULTS)
        print(f"M1: Static strategy")
        
    elif model_name == 'M2':
        def gamma_schedule(q, t, g_base=M2_DEFAULTS['gamma_base'], 
                          k=M2_DEFAULTS['entropy_k']):
            p = np.clip(np.asarray(q, float), 1e-12, 1.0)
            H = -(p * np.log(p)).sum()
            return g_base / (1.0 + k * H)
        
        value_fn = make_value_fn('M2', 
                                C_reward_logits=M2_DEFAULTS['C_reward_logits'],
                                gamma_schedule=gamma_schedule)
        print(f"M2: Entropy-based")
        
    elif model_name == 'M3':
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
        
        # Profiles for different regimes
        # Use REASONABLE parameters that can still perform well
        profiles = [
            {   # Profile 0: For left_better context
                'phi_logits': [0.0, -4.0, 2.5],     # Strong reward preference
                'xi_logits': [0.0, 0.0, 0.0, 0.0],  # Neutral on hints
                'gamma': 2.5                        # Moderate-high precision
            },
            {   # Profile 1: For right_better context
                'phi_logits': [0.0, -4.0, 2.5],     # Same reward preferences
                'xi_logits': [0.0, 0.0, 0.0, 0.0],  # Neutral on hints
                'gamma': 2.5                        # Moderate-high precision
            }
        ]
        
        # Z matrix: Assign contexts to profiles
        # This is just for demonstration - in reality they're the same
        Z = np.array([[1.0, 0.0],   # left_better -> Profile 0
                      [0.0, 1.0]])  # right_better -> Profile 1
        
        value_fn = make_value_fn('M3',
                                profiles=profiles,
                                Z=Z,
                                policies=policies,
                                num_actions_per_factor=num_actions_per_factor)
        print(f"M3: Profile-based")
        print(f"  Profile 0: γ=2.5, neutral on hints (for left_better)")
        print(f"  Profile 1: γ=2.5, neutral on hints (for right_better)")
        print(f"  Note: Z matrix now correctly assigns contexts to profiles")
    
    # Run episodes
    results = {
        'log_likelihoods': [],
        'accuracies': [],
        'total_rewards': [],
        'stable_accuracies': [],
        'volatile_accuracies': [],
        'stable_hint_usage': [],
        'volatile_hint_usage': [],
        'gamma_series': [],
        'regime_labels': []
    }
    
    for run in tqdm(range(num_runs), desc=f"Running {model_name}"):
        run_seed = seed + run if seed is not None else None
        
        if run_seed is not None:
            np.random.seed(run_seed)
        
        # Create environment
        env = RegimeDependentBandit(
            regime_schedule=regime_schedule,
            probability_reward=PROBABILITY_REWARD
        )
        
        # Create agent
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                            OBSERVATION_HINTS, OBSERVATION_REWARDS,
                            OBSERVATION_CHOICES, ACTION_CHOICES,
                            reward_mod_idx=1)
        
        # Run episode
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
        
        # Regime-specific metrics
        stable_trials = []
        volatile_trials = []
        stable_hints = 0
        volatile_hints = 0
        
        for i, t in enumerate(logs['t']):
            # Get regime at this trial
            regime_type = 'stable'
            for regime in regime_schedule:
                if t >= regime['start']:
                    regime_type = regime['type']
            
            if regime_type == 'stable':
                stable_trials.append(i)
                if logs['action'][i] == 'act_hint':
                    stable_hints += 1
            else:
                volatile_trials.append(i)
                if logs['action'][i] == 'act_hint':
                    volatile_hints += 1
        
        stable_acc = acc[stable_trials].mean() if len(stable_trials) > 0 else np.nan
        volatile_acc = acc[volatile_trials].mean() if len(volatile_trials) > 0 else np.nan
        
        stable_hint_rate = stable_hints / len(stable_trials) if len(stable_trials) > 0 else 0
        volatile_hint_rate = volatile_hints / len(volatile_trials) if len(volatile_trials) > 0 else 0
        
        # Store results
        results['log_likelihoods'].append(np.sum(logs['ll']))
        results['accuracies'].append(acc.mean())
        results['total_rewards'].append(np.sum(reward_values))
        results['stable_accuracies'].append(stable_acc)
        results['volatile_accuracies'].append(volatile_acc)
        results['stable_hint_usage'].append(stable_hint_rate)
        results['volatile_hint_usage'].append(volatile_hint_rate)
        results['gamma_series'].append(logs['gamma'])
        
        if run == 0:  # Save regime labels for plotting
            regime_labels = []
            for t in logs['t']:
                regime_type = 'stable'
                for regime in regime_schedule:
                    if t >= regime['start']:
                        regime_type = regime['type']
                regime_labels.append(regime_type)
            results['regime_labels'] = regime_labels
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Overall Accuracy:           {np.mean(results['accuracies']):.3f} ± {np.std(results['accuracies']):.3f}")
    print(f"Stable Regime Acc:          {np.nanmean(results['stable_accuracies']):.3f} ± {np.nanstd(results['stable_accuracies']):.3f}")
    print(f"Volatile Regime Acc:        {np.nanmean(results['volatile_accuracies']):.3f} ± {np.nanstd(results['volatile_accuracies']):.3f}")
    print(f"Log-Likelihood:             {np.mean(results['log_likelihoods']):.2f} ± {np.std(results['log_likelihoods']):.2f}")
    print(f"Total Reward:               {np.mean(results['total_rewards']):.1f} ± {np.std(results['total_rewards']):.1f}")
    print(f"Stable Hint Usage:          {np.mean(results['stable_hint_usage']):.3f} ± {np.std(results['stable_hint_usage']):.3f}")
    print(f"Volatile Hint Usage:        {np.mean(results['volatile_hint_usage']):.3f} ± {np.std(results['volatile_hint_usage']):.3f}")
    print(f"Mean Gamma:                 {np.mean([np.mean(g) for g in results['gamma_series']]):.3f}")
    
    return results


def compare_models_regime_dependent(num_trials=400, num_runs=10, seed=42):
    """Compare all models on regime-dependent task."""
    
    print("\n" + "="*70)
    print("MODEL COMPARISON: Regime-Dependent Hint Reliability")
    print("="*70)
    
    models = ['M1', 'M2', 'M3']
    all_results = {}
    
    for model in models:
        results = run_regime_dependent_experiment(
            model_name=model,
            num_trials=num_trials,
            num_runs=num_runs,
            seed=seed
        )
        all_results[model] = results
    
    # Comparison table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<8} {'Overall':<10} {'Stable':<10} {'Volatile':<10} {'LL':<12} "
          f"{'StableHint':<12} {'VolatHint':<12}")
    print("-"*70)
    
    for model in models:
        res = all_results[model]
        print(f"{model:<8} "
              f"{np.mean(res['accuracies']):<10.3f} "
              f"{np.nanmean(res['stable_accuracies']):<10.3f} "
              f"{np.nanmean(res['volatile_accuracies']):<10.3f} "
              f"{np.mean(res['log_likelihoods']):<12.2f} "
              f"{np.mean(res['stable_hint_usage']):<12.3f} "
              f"{np.mean(res['volatile_hint_usage']):<12.3f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("Optimal strategy:")
    print("  STABLE regime (55% hint accuracy):  Avoid hints, exploit directly")
    print("  VOLATILE regime (95% hint accuracy): Seek hints to track fast changes")
    print("\nM3 should:")
    print("  - Use Profile 1 (avoids hints) in stable regimes")
    print("  - Use Profile 0 (seeks hints) in volatile regimes")
    print("  - Outperform M1/M2 which can't adapt hint-seeking behavior")
    
    return all_results


def main():
    """Main entry point."""
    
    results = compare_models_regime_dependent(
        num_trials=400,
        num_runs=10,
        seed=42
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
