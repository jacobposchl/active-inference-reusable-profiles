"""
K-Sweep: Systematic Profile Configuration Testing (Classic Task)

This script demonstrates that K>1 (multiple reusable profiles) consistently outperforms
K=1 (single profile) by testing a comprehensive grid of profile configurations.

Key insight: We're not optimizing to find the "best" configuration, but rather
demonstrating that having access to multiple distinct behavioral profiles provides
a robust advantage across many reasonable parameter settings.
"""
import numpy as np
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.experiment_config import *
from src.environment import TwoArmedBandit
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL, run_episode_with_ll
from src.utils import find_reversals, trial_accuracy


def generate_profile_sweep():
    """
    Generate K=1 profile configurations with NEUTRAL preferences.
    
    These profiles cannot adapt to context - they have averaged/neutral preferences
    across both contexts (left_better vs right_better). This is the baseline to beat.
    
    Varying dimensions:
    - gamma (policy precision): 2.0 (exploratory), 4.5 (balanced), 7.0 (exploitative)
    - phi_loss (loss aversion): -4.0 (moderate), -7.0 (strong), -10.0 (very strong)
    
    Fixed parameters:
    - phi_reward: 7.0 (strong reward preference)
    - xi_hint: 0.0 (neutral - don't bias toward/away from hints)
    - xi_left, xi_right: 0.0 (NEUTRAL - no arm preference)
    
    Key point: K=1 profiles are context-agnostic. They must work equally well
    regardless of which arm is better. This is like M1 in the model comparison.
    
    Returns:
        List of 9 profile dictionaries (3 × 3 combinations)
    """
    gammas = [2.0, 4.5, 7.0]
    phi_losses = [-4.0, -7.0, -10.0]
    
    profiles = []
    for gamma in gammas:
        for phi_loss in phi_losses:
            profile = {
                'gamma': gamma,
                'phi_logits': [0.0, phi_loss, 7.0],  # [null, loss, reward]
                'xi_logits': [0.0, 0.0, 0.0, 0.0]  # [start, hint, left, right] - ALL NEUTRAL
            }
            profiles.append(profile)
    
    return profiles


def generate_K2_pairs():
    """
    Generate K=2 profile pairs with CONTEXT-SPECIFIC SPECIALIZATION.
    
    Core principle: Profiles should encode preferences aligned with different contexts.
    The Z matrix allows the model to weight profiles based on beliefs about which
    context is active (left_better vs right_better).
    
    Primary differentiation: Action preferences (xi_left, xi_right) that favor
    the appropriate arm for each context.
    
    Secondary variation: Policy precision (gamma) to test whether combining
    arm specialization with exploration/exploitation helps.
    
    Hypothesis: K=2 should beat K=1 because:
    1. When beliefs say "left is better", left-specialist profile is weighted more
    2. That profile already prefers the left arm (faster decisions, better coherence)
    3. During uncertainty (around reversals), profiles mix smoothly
    4. Better inference coherence (higher log-likelihood) than neutral K=1
    
    Returns:
        List of (profile1, profile2, Z_matrix, description) tuples
    """
    pairs = []
    
    # Define Z matrix configurations
    Z_hard = np.array([[1.0, 0.0], [0.0, 1.0]])  # Hard assignment: strict specialization
    Z_soft = np.array([[0.8, 0.2], [0.2, 0.8]])  # Soft assignment: mostly specialized
    Z_balanced = np.array([[0.5, 0.5], [0.5, 0.5]])  # Balanced: always mix equally (control)
    
    Z_configs = [
        (Z_hard, "hard_assignment"),
        (Z_soft, "soft_assignment"),
        (Z_balanced, "balanced")
    ]
    
    # ========================================================================
    # ARM SPECIALISTS: Core test of context-specific preferences
    # ========================================================================
    
    # Pair 1: Pure arm specialists (matched gamma, matched loss aversion)
    # This is the CLEANEST test: only arm preference differs
    profile_left_specialist = {
        'gamma': 4.5,
        'phi_logits': [0.0, -7.0, 7.0],
        'xi_logits': [0.0, 0.0, 3.0, -3.0]  # [start, hint, LEFT+, right-]
    }
    profile_right_specialist = {
        'gamma': 4.5,
        'phi_logits': [0.0, -7.0, 7.0],
        'xi_logits': [0.0, 0.0, -3.0, 3.0]  # [start, hint, left-, RIGHT+]
    }
    for Z, Z_name in Z_configs:
        pairs.append((profile_left_specialist, profile_right_specialist, Z, 
                     f"arm_specialists_balanced_{Z_name}"))
    
    # Pair 2: Arm specialists with different precision
    # Tests whether arm specialization + precision variation helps
    profile_left_exploit = {
        'gamma': 7.0,  # High precision when confident left is better
        'phi_logits': [0.0, -7.0, 7.0],
        'xi_logits': [0.0, 0.0, 2.0, -2.0]  # Moderate left bias
    }
    profile_right_explore = {
        'gamma': 2.0,  # Low precision when confident right is better
        'phi_logits': [0.0, -7.0, 7.0],
        'xi_logits': [0.0, 0.0, -2.0, 2.0]  # Moderate right bias
    }
    for Z, Z_name in Z_configs:
        pairs.append((profile_left_exploit, profile_right_explore, Z,
                     f"arm_specialists_precision_varied_{Z_name}"))
    
    # Pair 3: Arm specialists with different loss aversion
    # Tests whether arm specialization + risk preferences helps
    profile_left_cautious = {
        'gamma': 4.5,
        'phi_logits': [0.0, -10.0, 7.0],  # Very loss-averse
        'xi_logits': [0.0, 0.0, 3.0, -3.0]  # Strong left bias
    }
    profile_right_bold = {
        'gamma': 4.5,
        'phi_logits': [0.0, -4.0, 7.0],  # Moderate loss aversion
        'xi_logits': [0.0, 0.0, -3.0, 3.0]  # Strong right bias
    }
    for Z, Z_name in Z_configs:
        pairs.append((profile_left_cautious, profile_right_bold, Z,
                     f"arm_specialists_loss_varied_{Z_name}"))
    
    # Pair 4: Strong arm specialists (maximum differentiation)
    # Tests whether stronger arm biases improve performance
    profile_left_strong = {
        'gamma': 5.0,
        'phi_logits': [0.0, -7.0, 7.0],
        'xi_logits': [0.0, 0.0, 5.0, -5.0]  # Very strong left bias
    }
    profile_right_strong = {
        'gamma': 5.0,
        'phi_logits': [0.0, -7.0, 7.0],
        'xi_logits': [0.0, 0.0, -5.0, 5.0]  # Very strong right bias
    }
    for Z, Z_name in Z_configs:
        pairs.append((profile_left_strong, profile_right_strong, Z,
                     f"arm_specialists_strong_{Z_name}"))
    
    # Pair 5: Combined variation (arm specialization + precision + loss aversion)
    # Tests whether multiple dimensions of differentiation help
    profile_left_combo = {
        'gamma': 7.0,  # Exploitative
        'phi_logits': [0.0, -10.0, 7.0],  # Very cautious
        'xi_logits': [0.0, 0.0, 3.0, -3.0]  # Left specialist
    }
    profile_right_combo = {
        'gamma': 2.0,  # Exploratory
        'phi_logits': [0.0, -4.0, 7.0],  # Bold
        'xi_logits': [0.0, 0.0, -3.0, 3.0]  # Right specialist
    }
    for Z, Z_name in Z_configs:
        pairs.append((profile_left_combo, profile_right_combo, Z,
                     f"arm_specialists_combined_{Z_name}"))
    
    return pairs


def evaluate_profile_configuration(K, profiles, Z, A, B, D, num_trials=400, num_runs=20, seed=42):
    """
    Evaluate a specific profile configuration across multiple runs.
    
    Parameters:
        K: Number of profiles (1, 2, or 3)
        profiles: List of profile dictionaries
        Z: Assignment matrix (2 x K)
        A, B, D: Generative model matrices
        num_trials: Number of trials per run
        num_runs: Number of independent runs for statistical reliability
        seed: Random seed base
    
    Returns:
        Dictionary with comprehensive metrics including mean and std values
    """
    from pymdp.agent import Agent
    from pymdp import utils
    
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
    hint_usages = []
    
    for run in range(num_runs):
        run_seed = seed + run
        np.random.seed(run_seed)
        
        # Create environment
        env = TwoArmedBandit(
            probability_hint=PROBABILITY_HINT,
            probability_reward=PROBABILITY_REWARD,
            reversal_schedule=DEFAULT_REVERSAL_SCHEDULE
        )
        
        # Create agent
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                            OBSERVATION_HINTS, OBSERVATION_REWARDS,
                            OBSERVATION_CHOICES, ACTION_CHOICES,
                            reward_mod_idx=1)
        
        # Run episode
        logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)
        
        # Calculate metrics
        ll_sum = np.sum(logs['ll'])
        log_likelihoods.append(ll_sum)
        
        # Calculate total rewards (rewards = +1, losses = -1, null = 0)
        total_rewards = 0
        for reward_label in logs['reward_label']:
            if reward_label == 'observe_reward':
                total_rewards += 1
            elif reward_label == 'observe_loss':
                total_rewards -= 1
        
        total_rewards_list.append(total_rewards)
        
        # Calculate hint usage (proportion of trials where hint was chosen)
        hint_count = sum(1 for action in logs['action'] if action == 'act_hint')
        hint_usages.append(hint_count / num_trials)
    
    # Aggregate results
    results = {
        'K': K,
        'mean_ll': np.mean(log_likelihoods),
        'std_ll': np.std(log_likelihoods),
        'mean_total_rewards': np.mean(total_rewards_list),
        'std_total_rewards': np.std(total_rewards_list),
        'mean_reward_rate': np.mean(total_rewards_list) / num_trials,
        'mean_hint_usage': np.mean(hint_usages),
        'std_hint_usage': np.std(hint_usages),
        'profile_config': profiles,
        'Z_config': Z
    }
    
    return results


def describe_profile(profile):
    """Create a human-readable description of a profile."""
    gamma = profile['gamma']
    phi_loss = profile['phi_logits'][1]
    phi_reward = profile['phi_logits'][2]
    xi_hint = profile['xi_logits'][1]
    xi_left = profile['xi_logits'][2]
    xi_right = profile['xi_logits'][3]
    
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
    
    # Describe arm specialization (most important!)
    if xi_left > 1.0 and xi_right < -1.0:
        arm_desc = "LEFT_specialist"
    elif xi_right > 1.0 and xi_left < -1.0:
        arm_desc = "RIGHT_specialist"
    elif abs(xi_left) < 0.5 and abs(xi_right) < 0.5:
        arm_desc = "neutral"
    else:
        arm_desc = f"custom(L={xi_left:.1f},R={xi_right:.1f})"
    
    return f"{gamma_desc}_{loss_desc}_{arm_desc} (γ={gamma:.1f}, φ_loss={phi_loss:.1f}, ξ_L={xi_left:.1f}, ξ_R={xi_right:.1f})"


def main():
    """Main execution: systematic sweep through profile configurations."""
    
    print("=" * 70)
    print("K-SWEEP: SYSTEMATIC PROFILE CONFIGURATION TESTING")
    print("Classic Two-Armed Bandit Task")
    print("=" * 70)
    print()
    print("HYPOTHESIS: K>1 (multiple profiles) outperforms K=1 (single profile)")
    print("because belief-weighted profile mixing enables context-specific")
    print("adaptation that a single neutral profile cannot achieve.")
    print()
    print("K=1: Profiles with NEUTRAL preferences (context-agnostic)")
    print("     Like M1 model - must work equally for both contexts")
    print()
    print("K=2: ARM-SPECIALIZED profile pairs (context-specific)")
    print("     Left-specialist + Right-specialist")
    print("     Z matrix routes based on beliefs about which arm is better")
    print()
    print("K=3: Tests whether redundancy (more profiles than contexts)")
    print("     gracefully handled or provides additional benefit")
    print()
    
    # Build generative model
    print("Building generative model...")
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)
    
    print(f"Task parameters: hint accuracy = {PROBABILITY_HINT}, reward probability = {PROBABILITY_REWARD}")
    print()
    
    # Generate all profile configurations
    print("Generating profile configurations...")
    all_profiles = generate_profile_sweep()
    print(f"Generated {len(all_profiles)} K=1 configurations (neutral, context-agnostic)")
    
    k2_pairs = generate_K2_pairs()
    print(f"Generated {len(k2_pairs)} K=2 configurations (arm-specialized pairs)")
    print()
    print("NOTE: K=2 profiles have arm-specific preferences (xi_left, xi_right)")
    print("      while K=1 profiles are neutral (xi_left=0, xi_right=0)")
    print("      This tests whether context-specialization beats neutrality.")
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
    print()
    
    # ========================================================================
    # OPTIONAL: EVALUATE K=3 CONFIGURATIONS
    # ========================================================================
    print("=" * 70)
    print("EVALUATING K=3 CONFIGURATIONS (Sample)")
    print("=" * 70)
    print("Testing a few 3-profile configurations...")
    print()
    
    # Test a few K=3 configurations: low/medium/high gamma
    k3_configs = [
        # Configuration 1: Low/Medium/High gamma, all hint-neutral
        {
            'profiles': [
                {'gamma': 2.0, 'phi_logits': [0.0, -7.0, 2.0], 'xi_logits': [0.0, 0.0, 0.0, 0.0]},
                {'gamma': 4.5, 'phi_logits': [0.0, -7.0, 2.0], 'xi_logits': [0.0, 0.0, 0.0, 0.0]},
                {'gamma': 7.0, 'phi_logits': [0.0, -7.0, 2.0], 'xi_logits': [0.0, 0.0, 0.0, 0.0]}
            ],
            'Z': np.array([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34]]),
            'desc': "low_med_high_gamma_balanced"
        },
        # Configuration 2: Different strategies
        {
            'profiles': [
                {'gamma': 2.0, 'phi_logits': [0.0, -4.0, 2.0], 'xi_logits': [0.0, 1.5, 0.0, 0.0]},  # Exploratory hint-seeker
                {'gamma': 4.5, 'phi_logits': [0.0, -7.0, 2.0], 'xi_logits': [0.0, 0.0, 0.0, 0.0]},  # Balanced neutral
                {'gamma': 7.0, 'phi_logits': [0.0, -10.0, 2.0], 'xi_logits': [0.0, -1.0, 0.0, 0.0]}  # Exploitative hint-avoider
            ],
            'Z': np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]),
            'desc': "complementary_strategies_soft"
        }
    ]
    
    for i, config in enumerate(tqdm(k3_configs, desc="K=3 sweep")):
        result = evaluate_profile_configuration(
            K=3,
            profiles=config['profiles'],
            Z=config['Z'],
            A=A, B=B, D=D,
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
        print(f"  Hint Usage: {result['mean_hint_usage']:.3f} ± {result['std_hint_usage']:.3f}")
    
    print()
    print("=" * 70)
    print("K=2 CONFIGURATIONS: TOP 5 PERFORMERS")
    print("=" * 70)
    for rank, result in enumerate(results['K2'][:5], 1):
        print(f"\nRank {rank}: {result['description']}")
        print(f"  Total Rewards: {result['mean_total_rewards']:.1f} ± {result['std_total_rewards']:.1f}")
        print(f"  Log-Likelihood: {result['mean_ll']:.1f} ± {result['std_ll']:.1f}")
        print(f"  Reward Rate: {result['mean_reward_rate']:.3f}")
        print(f"  Hint Usage: {result['mean_hint_usage']:.3f} ± {result['std_hint_usage']:.3f}")
    
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
            print(f"  Hint Usage: {result['mean_hint_usage']:.3f} ± {result['std_hint_usage']:.3f}")
    
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
    improvement_pct = (improvement_k2_k1 / abs(best_k1['mean_total_rewards'])) * 100
    
    print(f"\nK=2 vs K=1 Improvement: +{improvement_k2_k1:.1f} rewards (+{improvement_pct:.1f}%)")
    
    if best_k3:
        improvement_k3_k1 = best_k3['mean_total_rewards'] - best_k1['mean_total_rewards']
        improvement_k3_pct = (improvement_k3_k1 / abs(best_k1['mean_total_rewards'])) * 100
        print(f"K=3 vs K=1 Improvement: +{improvement_k3_k1:.1f} rewards (+{improvement_k3_pct:.1f}%)")
    
    # Check if improvement is meaningful given std
    combined_std = np.sqrt(best_k1['std_total_rewards']**2 + best_k2['std_total_rewards']**2)
    if improvement_k2_k1 > 2 * combined_std:
        print(f"\n✅ K=2 demonstrates statistically significant advantage over K=1")
        print(f"   (improvement > 2× combined std: {improvement_k2_k1:.1f} > {2*combined_std:.1f})")
    else:
        print(f"\n⚠️  K=2 shows improvement but not statistically significant at 2σ level")
        print(f"   (improvement vs 2× combined std: {improvement_k2_k1:.1f} vs {2*combined_std:.1f})")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print()
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # K=1 distribution
    k1_rewards = [r['mean_total_rewards'] for r in results['K1']]
    axes[0].hist(k1_rewards, bins=15, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(best_k1['mean_total_rewards'], color='red', linestyle='--', linewidth=2, label='Best')
    axes[0].set_xlabel('Total Rewards')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'K=1 Performance Distribution\n({len(results["K1"])} configurations)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # K=2 distribution
    k2_rewards = [r['mean_total_rewards'] for r in results['K2']]
    axes[1].hist(k2_rewards, bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(best_k2['mean_total_rewards'], color='red', linestyle='--', linewidth=2, label='Best')
    axes[1].set_xlabel('Total Rewards')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'K=2 Performance Distribution\n({len(results["K2"])} configurations)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Comparison box plot
    box_data = [k1_rewards, k2_rewards]
    box_labels = ['K=1', 'K=2']
    if results['K3']:
        k3_rewards = [r['mean_total_rewards'] for r in results['K3']]
        box_data.append(k3_rewards)
        box_labels.append('K=3')
    
    bp = axes[2].boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['blue', 'green', 'orange']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[2].set_ylabel('Total Rewards')
    axes[2].set_title('Performance Comparison Across K')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results/figures', exist_ok=True)
    save_path = 'results/figures/k_sweep_classic_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    
    plt.show()
    
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print()
    print("CONCLUSION:")
    print(f"We tested {len(results['K1'])} K=1 (neutral) and {len(results['K2'])} K=2 (specialized) configurations.")
    
    if improvement_k2_k1 > 0:
        print(f"✅ K=2 outperformed K=1 by {improvement_pct:.1f}% on average.")
        print()
        print("This demonstrates that context-specific profile specialization")
        print("(arm-specific action preferences) provides better inference coherence")
        print("than a single neutral profile, validating the core claim that")
        print("differentiation across contexts matters.")
    else:
        print(f"⚠️  K=1 outperformed K=2 by {abs(improvement_pct):.1f}%.")
        print()
        print("This suggests that the task does not require context-specific")
        print("specialization, or that profile mixing overhead outweighs benefits.")
        print("Consider: Are arm biases too strong? Is Z assignment appropriate?")
    print()


if __name__ == "__main__":
    main()
