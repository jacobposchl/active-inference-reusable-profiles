"""
Sanity check: Test hand-crafted K=2 profiles vs optimized K=1.

This tests whether regime-specific profiles CAN beat a single general profile,
if we design them intelligently by hand.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.experiment_config import *
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL
from pymdp.agent import Agent
from pymdp import utils
from tqdm import tqdm


class RegimeDependentBandit:
    """Bandit with regime-dependent parameters."""
    
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
        """Override step to use regime-dependent parameters."""
        regime = self.get_current_regime()
        if regime:
            self.bandit.probability_hint = regime.get('hint_prob', 0.7)
            self.bandit.probability_reward = regime.get('reward_prob', 0.8)
            self.current_regime = regime['type']
            
            if 'reversals' in regime:
                if self.bandit.trial_count in regime['reversals']:
                    self.bandit.context = ('right_better' if self.bandit.context == 'left_better' 
                                          else 'left_better')
        
        return self.bandit.step(action)


def create_regime_schedule():
    """Create regime schedule with complementary trade-offs."""
    return [
        {'start': 0, 'type': 'stable', 
         'hint_prob': 0.55, 'reward_prob': 0.9, 
         'reversals': [60]},
        
        {'start': 120, 'type': 'volatile', 
         'hint_prob': 0.95, 'reward_prob': 0.65,
         'reversals': [128, 136, 144, 152, 160, 168, 176, 184]},
        
        {'start': 200, 'type': 'stable', 
         'hint_prob': 0.55, 'reward_prob': 0.9,
         'reversals': [260]},
        
        {'start': 320, 'type': 'volatile', 
         'hint_prob': 0.95, 'reward_prob': 0.65,
         'reversals': [328, 336, 344, 352, 360, 368, 376, 384]},
    ]


def get_regime_at_trial(t, regime_schedule):
    """Determine which regime we're in at trial t."""
    current_regime = None
    for regime in regime_schedule:
        if t >= regime['start']:
            current_regime = regime['type']
    return current_regime or 'stable'


def make_time_dependent_value_fn(profiles_stable, profiles_volatile, 
                                  Z_stable, Z_volatile, 
                                  regime_schedule, policies, num_actions_per_factor):
    """
    Create value function that switches profiles based on TIME (regime).
    
    This is the "oracle" version - agent knows which regime it's in.
    """
    from src.models.value_functions import map_action_prefs_to_policy_prefs
    from pymdp.maths import softmax
    
    # Normalize Z matrices
    Z_stable = np.asarray(Z_stable, float)
    Z_stable /= Z_stable.sum(axis=1, keepdims=True)
    Z_volatile = np.asarray(Z_volatile, float)
    Z_volatile /= Z_volatile.sum(axis=1, keepdims=True)
    
    # Process stable profiles
    PHI_stable = np.stack([np.asarray(p['phi_logits'], float) for p in profiles_stable], axis=0)
    GAM_stable = np.array([float(p['gamma']) for p in profiles_stable])
    XI_stable_list = []
    for p in profiles_stable:
        policy_logits = map_action_prefs_to_policy_prefs(
            p['xi_logits'], policies, num_actions_per_factor)
        XI_stable_list.append(policy_logits)
    XI_stable = np.stack(XI_stable_list, axis=0)
    
    # Process volatile profiles
    PHI_volatile = np.stack([np.asarray(p['phi_logits'], float) for p in profiles_volatile], axis=0)
    GAM_volatile = np.array([float(p['gamma']) for p in profiles_volatile])
    XI_volatile_list = []
    for p in profiles_volatile:
        policy_logits = map_action_prefs_to_policy_prefs(
            p['xi_logits'], policies, num_actions_per_factor)
        XI_volatile_list.append(policy_logits)
    XI_volatile = np.stack(XI_volatile_list, axis=0)
    
    def value_fn(q_context_t, t):
        """Mix profiles based on current regime (determined by time)."""
        regime = get_regime_at_trial(t, regime_schedule)
        
        # Select regime-appropriate parameters
        if regime == 'stable':
            PHI, GAM, XI, Z = PHI_stable, GAM_stable, XI_stable, Z_stable
        else:  # volatile
            PHI, GAM, XI, Z = PHI_volatile, GAM_volatile, XI_volatile, Z_volatile
        
        # Mix profiles based on context beliefs
        w = np.asarray(q_context_t, float) @ Z
        w = w / (w.sum() + 1e-12)
        
        # Compute mixed values
        phi_t = (w[:, None] * PHI).sum(axis=0)
        C_t = softmax(phi_t)
        
        gamma_t = float((w * GAM).sum())
        
        xi_t = (w[:, None] * XI).sum(axis=0)
        E_t = softmax(xi_t)
        
        return C_t, E_t, gamma_t
    
    return value_fn


def evaluate_model(name, profiles_stable, profiles_volatile, Z_stable, Z_volatile,
                   A, B, D, regime_schedule, num_trials=400, num_runs=10, seed=42):
    """Evaluate a model configuration."""
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {name}")
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
    
    # Create value function
    value_fn = make_time_dependent_value_fn(
        profiles_stable, profiles_volatile,
        Z_stable, Z_volatile,
        regime_schedule, policies, num_actions_per_factor
    )
    
    total_ll = 0.0
    total_rewards = 0.0
    stable_hint_usage = []
    volatile_hint_usage = []
    stable_rewards = []
    volatile_rewards = []
    
    for run in tqdm(range(num_runs), desc=name):
        np.random.seed(seed + run)
        
        env = RegimeDependentBandit(regime_schedule=regime_schedule)
        
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
            
            reward = 1 if obs[1] == 'observe_reward' else 0
            rewards.append(reward)
        
        total_rewards += sum(rewards)
        
        # Regime-specific metrics
        stable_hints = sum(1 for a, r in zip(actions, regimes) if a == 'act_hint' and r == 'stable')
        volatile_hints = sum(1 for a, r in zip(actions, regimes) if a == 'act_hint' and r == 'volatile')
        stable_trials = sum(1 for r in regimes if r == 'stable')
        volatile_trials = sum(1 for r in regimes if r == 'volatile')
        
        stable_hint_usage.append(stable_hints / stable_trials if stable_trials > 0 else 0)
        volatile_hint_usage.append(volatile_hints / volatile_trials if volatile_trials > 0 else 0)
        
        stable_reward_total = sum(r for r, reg in zip(rewards, regimes) if reg == 'stable')
        volatile_reward_total = sum(r for r, reg in zip(rewards, regimes) if reg == 'volatile')
        stable_rewards.append(stable_reward_total / stable_trials if stable_trials > 0 else 0)
        volatile_rewards.append(volatile_reward_total / volatile_trials if volatile_trials > 0 else 0)
    
    mean_ll = total_ll / (num_runs * num_trials)
    mean_rewards = total_rewards / num_runs
    
    print(f"\nResults:")
    print(f"  Mean LL: {mean_ll:.2f}")
    print(f"  Total Rewards: {mean_rewards:.1f} / {num_trials} trials")
    print(f"  Reward Rate: {mean_rewards/num_trials:.3f}")
    print(f"  Stable Hint Usage: {np.mean(stable_hint_usage):.3f} ± {np.std(stable_hint_usage):.3f}")
    print(f"  Volatile Hint Usage: {np.mean(volatile_hint_usage):.3f} ± {np.std(volatile_hint_usage):.3f}")
    print(f"  Stable Reward Rate: {np.mean(stable_rewards):.3f} ± {np.std(stable_rewards):.3f}")
    print(f"  Volatile Reward Rate: {np.mean(volatile_rewards):.3f} ± {np.std(volatile_rewards):.3f}")
    
    return {
        'll': mean_ll,
        'total_rewards': mean_rewards,
        'reward_rate': mean_rewards / num_trials,
        'stable_hints': np.mean(stable_hint_usage),
        'volatile_hints': np.mean(volatile_hint_usage),
        'stable_reward_rate': np.mean(stable_rewards),
        'volatile_reward_rate': np.mean(volatile_rewards)
    }


def main():
    """Run sanity check comparing K=1 optimized vs K=2 hand-crafted."""
    
    print("="*70)
    print("SANITY CHECK: Hand-crafted K=2 vs Optimized K=1")
    print("="*70)
    print()
    
    # Build generative model
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)
    
    regime_schedule = create_regime_schedule()
    
    # =========================================================================
    # MODEL 1: Optimized K=1 (from your results)
    # =========================================================================
    # Profile 0: γ=6.34, φ=-4.39/1.17, ξ_hint=0.88, ξ_left=-1.22, ξ_right=-1.12
    
    k1_profile = {
        'gamma': 6.34,
        'phi_logits': [0.0, -4.39, 1.17],
        'xi_logits': [0.0, 0.88, -1.22, -1.12]
    }
    
    # Use same profile for both regimes
    k1_profiles_stable = [k1_profile]
    k1_profiles_volatile = [k1_profile]
    k1_Z_stable = [[1.0], [1.0]]
    k1_Z_volatile = [[1.0], [1.0]]
    
    results_k1 = evaluate_model(
        "K=1 Optimized (Same profile for both regimes)",
        k1_profiles_stable, k1_profiles_volatile,
        k1_Z_stable, k1_Z_volatile,
        A, B, D, regime_schedule,
        num_trials=400, num_runs=20, seed=100
    )
    
    # =========================================================================
    # MODEL 2: Hand-crafted K=2 with REGIME-SPECIFIC profiles
    # =========================================================================
    
    # STABLE PROFILE: Exploit directly, ignore hints
    # - High reward preference (easy to get rewards with 0.9 prob)
    # - NEGATIVE hint bias (hints are useless at 0.55)
    # - High precision (be decisive)
    stable_profile = {
        'gamma': 7.0,  # High precision
        'phi_logits': [0.0, -5.0, 2.5],  # Strong reward preference
        'xi_logits': [0.0, -2.0, 0.0, 0.0]  # NEGATIVE hint bias, neutral on arms
    }
    
    # VOLATILE PROFILE: Use hints STRATEGICALLY (not exclusively!)
    # - Strong reward preference (still need to get rewards!)
    # - MODERATE hint bias (hints guide choices, but don't replace them)
    # - Medium precision (balance exploration and exploitation)
    volatile_profile = {
        'gamma': 5.0,  # Medium precision
        'phi_logits': [0.0, -4.0, 2.0],  # Strong reward preference
        'xi_logits': [0.0, 1.0, 0.0, 0.0]  # MODERATE hint bias (~40-50% hint usage)
    }
    
    # For K=2, we can use both profiles in each regime,
    # but weight them differently
    k2_profiles_stable = [stable_profile, volatile_profile]
    k2_profiles_volatile = [stable_profile, volatile_profile]
    
    # Z matrices: which profile to use in each context
    # In STABLE regime: mostly use stable_profile (index 0)
    k2_Z_stable = [
        [0.9, 0.1],  # left_better context: 90% stable, 10% volatile
        [0.9, 0.1]   # right_better context: 90% stable, 10% volatile
    ]
    
    # In VOLATILE regime: mostly use volatile_profile (index 1)
    k2_Z_volatile = [
        [0.1, 0.9],  # left_better context: 10% stable, 90% volatile
        [0.1, 0.9]   # right_better context: 10% stable, 90% volatile
    ]
    
    results_k2_handcrafted = evaluate_model(
        "K=2 Hand-crafted (Regime-specific profiles)",
        k2_profiles_stable, k2_profiles_volatile,
        k2_Z_stable, k2_Z_volatile,
        A, B, D, regime_schedule,
        num_trials=400, num_runs=20, seed=100
    )
    
    # =========================================================================
    # MODEL 3: K=2 with EXTREME specialization (sanity check)
    # =========================================================================
    
    # STABLE: Never use hints
    extreme_stable = {
        'gamma': 8.0,
        'phi_logits': [0.0, -6.0, 3.0],
        'xi_logits': [0.0, -3.0, 0.0, 0.0]  # Maximum negative hint bias
    }
    
    # VOLATILE: Always use hints
    extreme_volatile = {
        'gamma': 3.0,
        'phi_logits': [0.0, -2.0, 1.0],
        'xi_logits': [0.0, 3.0, 0.0, 0.0]  # Maximum positive hint bias
    }
    
    k2_profiles_stable_extreme = [extreme_stable, extreme_volatile]
    k2_profiles_volatile_extreme = [extreme_stable, extreme_volatile]
    
    results_k2_extreme = evaluate_model(
        "K=2 Extreme (Maximally differentiated profiles)",
        k2_profiles_stable_extreme, k2_profiles_volatile_extreme,
        k2_Z_stable, k2_Z_volatile,
        A, B, D, regime_schedule,
        num_trials=400, num_runs=20, seed=100
    )
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"{'Model':<50} {'Rewards':<12} {'Stable Hints':<15} {'Volatile Hints':<15}")
    print("-"*70)
    
    models = [
        ("K=1 Optimized", results_k1),
        ("K=2 Hand-crafted", results_k2_handcrafted),
        ("K=2 Extreme", results_k2_extreme)
    ]
    
    for name, r in models:
        print(f"{name:<50} {r['total_rewards']:<12.1f} {r['stable_hints']:<15.3f} {r['volatile_hints']:<15.3f}")
    
    print("\n" + "="*70)
    print("REGIME-SPECIFIC REWARD RATES")
    print("="*70)
    print(f"{'Model':<50} {'Stable':<15} {'Volatile':<15}")
    print("-"*70)
    
    for name, r in models:
        print(f"{name:<50} {r['stable_reward_rate']:<15.3f} {r['volatile_reward_rate']:<15.3f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if results_k2_handcrafted['total_rewards'] > results_k1['total_rewards']:
        improvement = results_k2_handcrafted['total_rewards'] - results_k1['total_rewards']
        print(f"✅ Hand-crafted K=2 BEATS K=1 by {improvement:.1f} rewards!")
        print("   → Regime-specific profiles DO help when properly designed")
        print("   → Optimizer should be able to find this (or better)")
    else:
        deficit = results_k1['total_rewards'] - results_k2_handcrafted['total_rewards']
        print(f"❌ K=1 still wins by {deficit:.1f} rewards")
        print("   → Either:")
        print("      - Hand-crafted profiles aren't good enough")
        print("      - Task doesn't benefit from specialization")
        print("      - Need better profile design")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
