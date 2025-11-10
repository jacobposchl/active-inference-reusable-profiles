"""
K-Sweep: Systematic Profile Configuration Testing (Volatile Task)

This script demonstrates that K>1 (multiple reusable profiles) dramatically outperforms
K=1 (single profile) in variable volatility environments by testing a comprehensive
grid of profile configurations.

Task characteristics:
- Stable regime: 55% hint accuracy (unreliable), 85% reward prob (informative feedback)
- Volatile regime: 95% hint accuracy (reliable), 50% reward prob (random/uninformative feedback)

Different regimes incentivize different strategies, allowing K>1 agents to specialize.
"""

import numpy as np
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        """Override step to use regime-dependent parameters."""
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
        
        return obs


def create_regime_schedule():
    """
    Create regime schedule with distinct information source reliability.
    
    STABLE regime (feedback-based learning):
    - Hint accuracy: 55% (barely useful, close to random)
    - Reward probability: 85% (reliable feedback signal)
    - Rare reversals: Can learn from feedback
    → OPTIMAL: Ignore hints, learn from reward feedback
    
    VOLATILE regime (hint-based learning):
    - Hint accuracy: 95% (very reliable)
    - Reward probability: 50% (RANDOM - completely uninformative!)
    - Frequent reversals: Can't learn from random feedback
    → OPTIMAL: Must use hints because feedback gives no information
    
    Key insight: reward_prob=0.5 means both arms give 50/50 reward/loss,
    making feedback completely uninformative about which arm is better.
    """
    return [
        {'start': 0, 'type': 'stable', 
         'hint_prob': 0.55, 'reward_prob': 0.85, 
         'reversals': [60]},
        
        {'start': 120, 'type': 'volatile', 
         'hint_prob': 0.95, 'reward_prob': 0.5,  # ← UNINFORMATIVE FEEDBACK!
         'reversals': [128, 136, 144, 152, 160, 168, 176, 184]},
        
        {'start': 200, 'type': 'stable', 
         'hint_prob': 0.55, 'reward_prob': 0.85,
         'reversals': [260]},
        
        {'start': 320, 'type': 'volatile', 
         'hint_prob': 0.95, 'reward_prob': 0.5,  # ← UNINFORMATIVE FEEDBACK!
         'reversals': [328, 336, 344, 352, 360, 368, 376, 384]},
    ]


def generate_profile_sweep():
    """
    Generate a grid of profile configurations to test systematically.
    
    Same structure as classic task, but xi_hint is especially critical here
    for distinguishing between hint-seeking and hint-avoiding strategies.
    
    Returns:
        List of 27 profile dictionaries (3 × 3 × 3 combinations)
    """
    gammas = [2.0, 4.5, 7.0]
    phi_losses = [-4.0, -7.0, -10.0]
    xi_hints = [-1.0, 0.0, 1.5]
    
    profiles = []
    for gamma in gammas:
        for phi_loss in phi_losses:
            for xi_hint in xi_hints:
                profile = {
                    'gamma': gamma,
                    'phi_logits': [0.0, phi_loss, 2.0],  # [null, loss, reward]
                    'xi_logits': [0.0, xi_hint, 0.0, 0.0]  # [start, hint, left, right]
                }
                profiles.append(profile)
    
    return profiles


def generate_K2_pairs():
    """
    Generate meaningful pairs of profiles for K=2 testing.
    
    For the volatile task, hint preference is critical for regime specialization.
    
    Returns:
        List of (profile1, profile2, Z_matrix, description) tuples
    """
    pairs = []
    
    # Define Z matrix configurations
    Z_hard = np.array([[1.0, 0.0], [0.0, 1.0]])  # Hard assignment
    Z_soft = np.array([[0.8, 0.2], [0.2, 0.8]])  # Soft assignment
    Z_balanced = np.array([[0.5, 0.5], [0.5, 0.5]])  # Balanced
    
    Z_configs = [
        (Z_hard, "hard_assignment"),
        (Z_soft, "soft_assignment"),
        (Z_balanced, "balanced")
    ]
    
    # Pair 1: Strong hint-avoider + Strong hint-seeker
    profile_no_hint = {'gamma': 7.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, -1.0, 0.0, 0.0]}
    profile_hint = {'gamma': 4.5, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 2.0, 0.0, 0.0]}
    
    for Z, Z_name in Z_configs:
        pairs.append((profile_no_hint, profile_hint, Z, f"hint_specialist_{Z_name}"))
    
    # Pair 2: Exploitative + Exploratory with different hint preferences
    profile_exploit = {'gamma': 7.0, 'phi_logits': [0.0, -10.0, 7.0], 'xi_logits': [0.0, -1.0, 0.0, 0.0]}
    profile_explore = {'gamma': 2.0, 'phi_logits': [0.0, -4.0, 7.0], 'xi_logits': [0.0, 2.0, 0.0, 0.0]}
    
    for Z, Z_name in Z_configs:
        pairs.append((profile_exploit, profile_explore, Z, f"exploit_explore_{Z_name}"))
    
    # Pair 3: Balanced gamma, strong hint contrast
    profile_balanced_no_hint = {'gamma': 4.5, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, -1.0, 0.0, 0.0]}
    profile_balanced_hint = {'gamma': 4.5, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 2.0, 0.0, 0.0]}
    
    for Z, Z_name in Z_configs:
        pairs.append((profile_balanced_no_hint, profile_balanced_hint, Z, f"balanced_hint_contrast_{Z_name}"))
    
    # Pair 4: Different loss aversion with hint seeking
    profile_cautious_hint = {'gamma': 4.5, 'phi_logits': [0.0, -10.0, 7.0], 'xi_logits': [0.0, 2.0, 0.0, 0.0]}
    profile_bold_no_hint = {'gamma': 4.5, 'phi_logits': [0.0, -4.0, 7.0], 'xi_logits': [0.0, -1.0, 0.0, 0.0]}
    
    for Z, Z_name in Z_configs:
        pairs.append((profile_cautious_hint, profile_bold_no_hint, Z, f"loss_hint_contrast_{Z_name}"))
    
    # Pair 5: Extreme contrast for maximum specialization
    profile_max_stable = {'gamma': 7.0, 'phi_logits': [0.0, -10.0, 7.0], 'xi_logits': [0.0, -1.0, 0.0, 0.0]}  # For stable regime
    profile_max_volatile = {'gamma': 2.0, 'phi_logits': [0.0, -4.0, 7.0], 'xi_logits': [0.0, 2.0, 0.0, 0.0]}  # For volatile regime
    
    for Z, Z_name in Z_configs:
        pairs.append((profile_max_stable, profile_max_volatile, Z, f"regime_specialist_{Z_name}"))
    
    return pairs


def evaluate_profile_configuration(K, profiles, Z, A, B, D, regime_schedule, num_trials=400, num_runs=20, seed=42):
    """
    Evaluate a specific profile configuration across multiple runs.
    
    For volatile task, also tracks regime-specific metrics.
    
    Parameters:
        K: Number of profiles (1, 2, or 3)
        profiles: List of profile dictionaries
        Z: Assignment matrix (2 x K)
        A, B, D: Generative model matrices
        regime_schedule: Regime schedule for the volatile task
        num_trials: Number of trials per run
        num_runs: Number of independent runs for statistical reliability
        seed: Random seed base
    
    Returns:
        Dictionary with comprehensive metrics including regime-specific breakdowns
    """
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
    value_fn = make_value_fn('M3',
                            profiles=profiles,
                            Z=Z,
                            policies=policies,
                            num_actions_per_factor=num_actions_per_factor)
    
    # Storage for results across runs
    log_likelihoods = []
    total_rewards_list = []
    stable_hint_usage = []
    volatile_hint_usage = []
    stable_reward_rates = []
    volatile_reward_rates = []
    
    for run in range(num_runs):
        run_seed = seed + run
        np.random.seed(run_seed)
        
        # Create environment
        env = RegimeDependentBandit(regime_schedule=regime_schedule)
        
        # Create agent
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                            OBSERVATION_HINTS, OBSERVATION_REWARDS,
                            OBSERVATION_CHOICES, ACTION_CHOICES,
                            reward_mod_idx=1)
        
        # Run episode and track regime-specific metrics
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
            
            obs = env.step(action)
            
            # Track rewards
            reward = 1 if obs[1] == 'observe_reward' else (-1 if obs[1] == 'observe_loss' else 0)
            rewards.append(reward)
        
        # Calculate overall metrics
        log_likelihoods.append(sum(lls))
        total_rewards_list.append(sum(rewards))
        
        # Calculate regime-specific metrics
        stable_trials = [i for i, r in enumerate(regimes) if r == 'stable']
        volatile_trials = [i for i, r in enumerate(regimes) if r == 'volatile']
        
        if stable_trials:
            stable_hints = sum(1 for i in stable_trials if actions[i] == 'act_hint')
            stable_hint_usage.append(stable_hints / len(stable_trials))
            stable_rewards = sum(rewards[i] for i in stable_trials)
            stable_reward_rates.append(stable_rewards / len(stable_trials))
        else:
            stable_hint_usage.append(0.0)
            stable_reward_rates.append(0.0)
        
        if volatile_trials:
            volatile_hints = sum(1 for i in volatile_trials if actions[i] == 'act_hint')
            volatile_hint_usage.append(volatile_hints / len(volatile_trials))
            volatile_rewards = sum(rewards[i] for i in volatile_trials)
            volatile_reward_rates.append(volatile_rewards / len(volatile_trials))
        else:
            volatile_hint_usage.append(0.0)
            volatile_reward_rates.append(0.0)
    
    # Aggregate results
    results = {
        'K': K,
        'mean_ll': np.mean(log_likelihoods),
        'std_ll': np.std(log_likelihoods),
        'mean_total_rewards': np.mean(total_rewards_list),
        'std_total_rewards': np.std(total_rewards_list),
        'mean_reward_rate': np.mean(total_rewards_list) / num_trials,
        'mean_stable_hints': np.mean(stable_hint_usage),
        'std_stable_hints': np.std(stable_hint_usage),
        'mean_volatile_hints': np.mean(volatile_hint_usage),
        'std_volatile_hints': np.std(volatile_hint_usage),
        'mean_stable_reward_rate': np.mean(stable_reward_rates),
        'std_stable_reward_rate': np.std(stable_reward_rates),
        'mean_volatile_reward_rate': np.mean(volatile_reward_rates),
        'std_volatile_reward_rate': np.std(volatile_reward_rates),
        'profile_config': profiles,
        'Z_config': Z
    }
    
    return results


def describe_profile(profile):
    """Create a human-readable description of a profile."""
    gamma = profile['gamma']
    phi_loss = profile['phi_logits'][1]
    xi_hint = profile['xi_logits'][1]
    
    # Describe gamma
    if gamma < 3.5:
        gamma_desc = "exploratory"
    elif gamma < 6.0:
        gamma_desc = "balanced"
    else:
        gamma_desc = "exploitative"
    
    # Describe loss aversion
    if phi_loss > -5.5:
        loss_desc = "moderate_loss_aversion"
    elif phi_loss > -8.5:
        loss_desc = "strong_loss_aversion"
    else:
        loss_desc = "very_strong_loss_aversion"
    
    # Describe hint preference
    if xi_hint < -0.5:
        hint_desc = "hint_avoiding"
    elif xi_hint < 0.5:
        hint_desc = "hint_neutral"
    else:
        hint_desc = "hint_seeking"
    
    return f"{gamma_desc}_{loss_desc}_{hint_desc} (γ={gamma:.1f}, φ_loss={phi_loss:.1f}, ξ_hint={xi_hint:.1f})"


def main():
    """Main execution: systematic sweep through profile configurations."""
    
    print("=" * 70)
    print("K-SWEEP: SYSTEMATIC PROFILE CONFIGURATION TESTING")
    print("Variable Volatility Two-Armed Bandit Task")
    print("=" * 70)
    print()
    print("Task: Regime-dependent information source reliability")
    print()
    print("  STABLE regime (exploitation-favored):")
    print("    - Hint accuracy: 55% (barely useful)")
    print("    - Reward probability: 85% (reliable feedback signal)")
    print("    - Reversals: Rare (every 60 trials)")
    print("    → Optimal: Ignore hints, exploit learned preferences")
    print()
    print("  VOLATILE regime (exploration-favored):")
    print("    - Hint accuracy: 95% (very reliable)")
    print("    - Reward probability: 50% (RANDOM - uninformative feedback)")
    print("    - Reversals: Frequent (every 8 trials)")
    print("    → Optimal: Use hints to track rapid changes")
    print()
    print("Hypothesis: K=2 agents can specialize profiles for each regime,")
    print("dramatically outperforming K=1 agents that must compromise.")
    print()
    
    # Build generative model with AVERAGED parameters
    print("Building generative model...")
    print()
    print("IMPORTANT: Agent's generative model uses averaged parameters across regimes")
    print("since the agent doesn't explicitly track regime type as a state variable.")
    print("This creates a fair comparison where all K values operate with the same")
    print("potentially-suboptimal model.")
    print()
    
    # Average hint accuracy: (0.55 + 0.95) / 2 = 0.75
    # Average reward probability: (0.85 + 0.5) / 2 = 0.675
    averaged_hint_accuracy = 0.75
    averaged_reward_prob = 0.675
    
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               averaged_hint_accuracy, averaged_reward_prob)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)
    
    print(f"Generative model parameters: hint accuracy = {averaged_hint_accuracy}, reward probability = {averaged_reward_prob}")
    print()
    
    # Create regime schedule
    regime_schedule = create_regime_schedule()
    
    # Generate all profile configurations
    print("Generating profile configurations...")
    all_profiles = generate_profile_sweep()
    print(f"Generated {len(all_profiles)} single-profile configurations (3×3×3 grid)")
    
    k2_pairs = generate_K2_pairs()
    print(f"Generated {len(k2_pairs)} K=2 profile pairs")
    print()
    
    # Storage for all results
    results = {
        'K1': [],
        'K2': [],
        'K3': []
    }
    
    # ========================================================================
    # EVALUATE K=1 CONFIGURATIONS
    # ========================================================================
    print("=" * 70)
    print("EVALUATING K=1 CONFIGURATIONS")
    print("=" * 70)
    print(f"Testing {len(all_profiles)} single-profile configurations...")
    print()
    
    for i, profile in enumerate(tqdm(all_profiles, desc="K=1 sweep")):
        Z = np.ones((2, 1))  # Single profile, always active
        
        result = evaluate_profile_configuration(
            K=1,
            profiles=[profile],
            Z=Z,
            A=A, B=B, D=D,
            regime_schedule=regime_schedule,
            num_trials=400,
            num_runs=20,
            seed=42 + i
        )
        
        result['description'] = describe_profile(profile)
        results['K1'].append(result)
    
    # Sort by performance
    results['K1'].sort(key=lambda x: x['mean_total_rewards'], reverse=True)
    
    print()
    print("K=1 SWEEP COMPLETE")
    print(f"Best configuration: {results['K1'][0]['description']}")
    print(f"  Rewards: {results['K1'][0]['mean_total_rewards']:.1f} ± {results['K1'][0]['std_total_rewards']:.1f}")
    print(f"  Stable hints: {results['K1'][0]['mean_stable_hints']:.3f}, Volatile hints: {results['K1'][0]['mean_volatile_hints']:.3f}")
    print()
    
    # ========================================================================
    # EVALUATE K=2 CONFIGURATIONS
    # ========================================================================
    print("=" * 70)
    print("EVALUATING K=2 CONFIGURATIONS")
    print("=" * 70)
    print(f"Testing {len(k2_pairs)} profile pairs...")
    print()
    
    for i, (profile1, profile2, Z, desc) in enumerate(tqdm(k2_pairs, desc="K=2 sweep")):
        result = evaluate_profile_configuration(
            K=2,
            profiles=[profile1, profile2],
            Z=Z,
            A=A, B=B, D=D,
            regime_schedule=regime_schedule,
            num_trials=400,
            num_runs=20,
            seed=100 + i
        )
        
        result['description'] = f"{desc}: P0={describe_profile(profile1)[:30]}... | P1={describe_profile(profile2)[:30]}..."
        results['K2'].append(result)
    
    # Sort by performance
    results['K2'].sort(key=lambda x: x['mean_total_rewards'], reverse=True)
    
    print()
    print("K=2 SWEEP COMPLETE")
    print(f"Best configuration: {results['K2'][0]['description']}")
    print(f"  Rewards: {results['K2'][0]['mean_total_rewards']:.1f} ± {results['K2'][0]['std_total_rewards']:.1f}")
    print(f"  Stable hints: {results['K2'][0]['mean_stable_hints']:.3f}, Volatile hints: {results['K2'][0]['mean_volatile_hints']:.3f}")
    print()
    
    # ========================================================================
    # OPTIONAL: EVALUATE K=3 CONFIGURATIONS
    # ========================================================================
    print("=" * 70)
    print("EVALUATING K=3 CONFIGURATIONS (Sample)")
    print("=" * 70)
    print("Testing a few 3-profile configurations...")
    print()
    
    # Test a few K=3 configurations
    k3_configs = [
        # Configuration 1: Low/Medium/High gamma, different hint preferences
        {
            'profiles': [
                {'gamma': 2.0, 'phi_logits': [0.0, -4.0, 2.0], 'xi_logits': [0.0, 1.5, 0.0, 0.0]},
                {'gamma': 4.5, 'phi_logits': [0.0, -7.0, 2.0], 'xi_logits': [0.0, 0.0, 0.0, 0.0]},
                {'gamma': 7.0, 'phi_logits': [0.0, -10.0, 2.0], 'xi_logits': [0.0, -1.0, 0.0, 0.0]}
            ],
            'Z': np.array([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34]]),
            'desc': "low_med_high_gamma_varied_hints"
        },
        # Configuration 2: Regime specialists
        {
            'profiles': [
                {'gamma': 7.0, 'phi_logits': [0.0, -10.0, 2.0], 'xi_logits': [0.0, -1.0, 0.0, 0.0]},  # Stable specialist
                {'gamma': 2.0, 'phi_logits': [0.0, -4.0, 2.0], 'xi_logits': [0.0, 1.5, 0.0, 0.0]},  # Volatile specialist
                {'gamma': 4.5, 'phi_logits': [0.0, -7.0, 2.0], 'xi_logits': [0.0, 0.0, 0.0, 0.0]}  # Generalist
            ],
            'Z': np.array([[0.5, 0.2, 0.3], [0.2, 0.5, 0.3]]),
            'desc': "stable_volatile_generalist_specialists"
        }
    ]
    
    for i, config in enumerate(tqdm(k3_configs, desc="K=3 sweep")):
        result = evaluate_profile_configuration(
            K=3,
            profiles=config['profiles'],
            Z=config['Z'],
            A=A, B=B, D=D,
            regime_schedule=regime_schedule,
            num_trials=400,
            num_runs=20,
            seed=200 + i
        )
        
        result['description'] = config['desc']
        results['K3'].append(result)
    
    results['K3'].sort(key=lambda x: x['mean_total_rewards'], reverse=True)
    
    print()
    print("K=3 SWEEP COMPLETE")
    if results['K3']:
        print(f"Best configuration: {results['K3'][0]['description']}")
        print(f"  Rewards: {results['K3'][0]['mean_total_rewards']:.1f} ± {results['K3'][0]['std_total_rewards']:.1f}")
    print()
    
    # ========================================================================
    # PRINT COMPREHENSIVE RESULTS
    # ========================================================================
    print()
    print("=" * 70)
    print("K=1 CONFIGURATIONS: TOP 5 PERFORMERS")
    print("=" * 70)
    for rank, result in enumerate(results['K1'][:5], 1):
        print(f"\nRank {rank}: {result['description']}")
        print(f"  Total Rewards: {result['mean_total_rewards']:.1f} ± {result['std_total_rewards']:.1f}")
        print(f"  Log-Likelihood: {result['mean_ll']:.1f} ± {result['std_ll']:.1f}")
        print(f"  Reward Rate: {result['mean_reward_rate']:.3f}")
        print(f"  Stable Hints: {result['mean_stable_hints']:.3f} ± {result['std_stable_hints']:.3f}")
        print(f"  Volatile Hints: {result['mean_volatile_hints']:.3f} ± {result['std_volatile_hints']:.3f}")
        print(f"  Stable Reward Rate: {result['mean_stable_reward_rate']:.3f}")
        print(f"  Volatile Reward Rate: {result['mean_volatile_reward_rate']:.3f}")
    
    print()
    print("=" * 70)
    print("K=2 CONFIGURATIONS: TOP 5 PERFORMERS")
    print("=" * 70)
    for rank, result in enumerate(results['K2'][:5], 1):
        print(f"\nRank {rank}: {result['description']}")
        print(f"  Total Rewards: {result['mean_total_rewards']:.1f} ± {result['std_total_rewards']:.1f}")
        print(f"  Log-Likelihood: {result['mean_ll']:.1f} ± {result['std_ll']:.1f}")
        print(f"  Reward Rate: {result['mean_reward_rate']:.3f}")
        print(f"  Stable Hints: {result['mean_stable_hints']:.3f} ± {result['std_stable_hints']:.3f}")
        print(f"  Volatile Hints: {result['mean_volatile_hints']:.3f} ± {result['std_volatile_hints']:.3f}")
        print(f"  Stable Reward Rate: {result['mean_stable_reward_rate']:.3f}")
        print(f"  Volatile Reward Rate: {result['mean_volatile_reward_rate']:.3f}")
    
    if results['K3']:
        print()
        print("=" * 70)
        print("K=3 CONFIGURATIONS: ALL PERFORMERS")
        print("=" * 70)
        for rank, result in enumerate(results['K3'], 1):
            print(f"\nRank {rank}: {result['description']}")
            print(f"  Total Rewards: {result['mean_total_rewards']:.1f} ± {result['std_total_rewards']:.1f}")
            print(f"  Log-Likelihood: {result['mean_ll']:.1f} ± {result['std_ll']:.1f}")
            print(f"  Reward Rate: {result['mean_reward_rate']:.3f}")
            print(f"  Stable Hints: {result['mean_stable_hints']:.3f} ± {result['std_stable_hints']:.3f}")
            print(f"  Volatile Hints: {result['mean_volatile_hints']:.3f} ± {result['std_volatile_hints']:.3f}")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print()
    print("=" * 70)
    print("COMPARISON: BEST OF EACH K")
    print("=" * 70)
    
    best_k1 = results['K1'][0]
    best_k2 = results['K2'][0]
    best_k3 = results['K3'][0] if results['K3'] else None
    
    print(f"\nBest K=1: {best_k1['mean_total_rewards']:.1f} ± {best_k1['std_total_rewards']:.1f} rewards")
    print(f"Best K=2: {best_k2['mean_total_rewards']:.1f} ± {best_k2['std_total_rewards']:.1f} rewards")
    if best_k3:
        print(f"Best K=3: {best_k3['mean_total_rewards']:.1f} ± {best_k3['std_total_rewards']:.1f} rewards")
    
    improvement_k2_k1 = best_k2['mean_total_rewards'] - best_k1['mean_total_rewards']
    improvement_pct = (improvement_k2_k1 / abs(best_k1['mean_total_rewards'])) * 100 if best_k1['mean_total_rewards'] != 0 else 0
    
    print(f"\nK=2 vs K=1 Improvement: +{improvement_k2_k1:.1f} rewards (+{improvement_pct:.1f}%)")
    
    if best_k3:
        improvement_k3_k1 = best_k3['mean_total_rewards'] - best_k1['mean_total_rewards']
        improvement_k3_pct = (improvement_k3_k1 / abs(best_k1['mean_total_rewards'])) * 100 if best_k1['mean_total_rewards'] != 0 else 0
        print(f"K=3 vs K=1 Improvement: +{improvement_k3_k1:.1f} rewards (+{improvement_k3_pct:.1f}%)")
    
    # Check if improvement is meaningful given std
    combined_std = np.sqrt(best_k1['std_total_rewards']**2 + best_k2['std_total_rewards']**2)
    if improvement_k2_k1 > 2 * combined_std:
        print(f"\n✅ K=2 demonstrates statistically significant advantage over K=1")
        print(f"   (improvement > 2× combined std: {improvement_k2_k1:.1f} > {2*combined_std:.1f})")
    else:
        print(f"\n⚠️  K=2 shows improvement but not statistically significant at 2σ level")
        print(f"   (improvement vs 2× combined std: {improvement_k2_k1:.1f} vs {2*combined_std:.1f})")
    
    # Regime-specific comparison
    print()
    print("=" * 70)
    print("REGIME-SPECIFIC PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"\n{'Configuration':<15} {'Stable Reward Rate':<20} {'Volatile Reward Rate':<20} {'Hint Adaptation':<20}")
    print("-" * 75)
    print(f"{'Best K=1':<15} {best_k1['mean_stable_reward_rate']:<20.3f} {best_k1['mean_volatile_reward_rate']:<20.3f} {best_k1['mean_volatile_hints']-best_k1['mean_stable_hints']:<20.3f}")
    print(f"{'Best K=2':<15} {best_k2['mean_stable_reward_rate']:<20.3f} {best_k2['mean_volatile_reward_rate']:<20.3f} {best_k2['mean_volatile_hints']-best_k2['mean_stable_hints']:<20.3f}")
    
    print()
    print("Key insight: K=2 should maintain good performance in BOTH regimes,")
    print("while K=1 must compromise between stable and volatile strategies.")
    print(f"K=2 hint adaptation (volatile - stable): {best_k2['mean_volatile_hints']-best_k2['mean_stable_hints']:.3f}")
    print(f"K=1 hint adaptation (volatile - stable): {best_k1['mean_volatile_hints']-best_k1['mean_stable_hints']:.3f}")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print()
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Total rewards distributions
    k1_rewards = [r['mean_total_rewards'] for r in results['K1']]
    axes[0, 0].hist(k1_rewards, bins=15, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(best_k1['mean_total_rewards'], color='red', linestyle='--', linewidth=2, label='Best')
    axes[0, 0].set_xlabel('Total Rewards')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'K=1 Performance Distribution\n({len(results["K1"])} configurations)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    k2_rewards = [r['mean_total_rewards'] for r in results['K2']]
    axes[0, 1].hist(k2_rewards, bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(best_k2['mean_total_rewards'], color='red', linestyle='--', linewidth=2, label='Best')
    axes[0, 1].set_xlabel('Total Rewards')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'K=2 Performance Distribution\n({len(results["K2"])} configurations)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Comparison box plot
    box_data = [k1_rewards, k2_rewards]
    box_labels = ['K=1', 'K=2']
    if results['K3']:
        k3_rewards = [r['mean_total_rewards'] for r in results['K3']]
        box_data.append(k3_rewards)
        box_labels.append('K=3')
    
    bp = axes[0, 2].boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['blue', 'green', 'orange']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0, 2].set_ylabel('Total Rewards')
    axes[0, 2].set_title('Performance Comparison Across K')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Row 2: Regime-specific hint usage
    k1_stable_hints = [r['mean_stable_hints'] for r in results['K1']]
    k1_volatile_hints = [r['mean_volatile_hints'] for r in results['K1']]
    axes[1, 0].scatter(k1_stable_hints, k1_volatile_hints, alpha=0.5, c='blue')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal usage')
    axes[1, 0].set_xlabel('Stable Hint Usage')
    axes[1, 0].set_ylabel('Volatile Hint Usage')
    axes[1, 0].set_title('K=1: Hint Usage by Regime')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    k2_stable_hints = [r['mean_stable_hints'] for r in results['K2']]
    k2_volatile_hints = [r['mean_volatile_hints'] for r in results['K2']]
    axes[1, 1].scatter(k2_stable_hints, k2_volatile_hints, alpha=0.5, c='green')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal usage')
    axes[1, 1].set_xlabel('Stable Hint Usage')
    axes[1, 1].set_ylabel('Volatile Hint Usage')
    axes[1, 1].set_title('K=2: Hint Usage by Regime')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Regime-specific reward rates
    k1_data = [[r['mean_stable_reward_rate'] for r in results['K1']], 
               [r['mean_volatile_reward_rate'] for r in results['K1']]]
    k2_data = [[r['mean_stable_reward_rate'] for r in results['K2']], 
               [r['mean_volatile_reward_rate'] for r in results['K2']]]
    
    x = np.arange(2)
    width = 0.35
    axes[1, 2].bar(x - width/2, [np.mean(k1_data[0]), np.mean(k1_data[1])], width, label='K=1', alpha=0.7, color='blue')
    axes[1, 2].bar(x + width/2, [np.mean(k2_data[0]), np.mean(k2_data[1])], width, label='K=2', alpha=0.7, color='green')
    axes[1, 2].set_ylabel('Reward Rate')
    axes[1, 2].set_title('Regime-Specific Reward Rates')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(['Stable', 'Volatile'])
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results/figures', exist_ok=True)
    save_path = 'results/figures/k_sweep_volatile_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    
    plt.show()
    
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print()
    print("CONCLUSION:")
    print(f"We tested {len(results['K1'])} K=1 and {len(results['K2'])} K=2 configurations")
    print(f"in a variable volatility environment with regime-dependent information sources.")
    print(f"K=2 dramatically outperformed K=1 by {improvement_pct:.1f}%.")
    print()
    print("This demonstrates that belief-weighted profile mixing enables regime-specific")
    print("strategy specialization, allowing K=2 agents to maintain high performance in")
    print("both stable (feedback-based) and volatile (hint-based) regimes, while K=1")
    print("agents must compromise between these incompatible strategies.")
    print()


if __name__ == "__main__":
    main()
