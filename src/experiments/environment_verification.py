"""
Analytical Strategy Evaluation for Multi-Armed Bandit Task

This script calculates the theoretical expected value per trial for different
behavioral strategies across different regime configurations. Use this to 
verify that your environment actually incentivizes different strategies in
different regimes BEFORE running expensive optimizations.

Key insight: If all regimes favor the same strategy, then K=1 will be optimal
no matter what. If different regimes favor different strategies, then K>1 
can provide benefits by specializing profiles.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path


@dataclass
class RegimeConfig:
    """Configuration for a single regime."""
    name: str
    hint_accuracy: float  # p(hint correct | state)
    reward_prob: float    # p(reward | correct choice)
    penalty: float        # negative reward for wrong choice
    trials_between_reversals: int  # how long until context flips
    
    def __str__(self):
        return (f"{self.name}: hint_acc={self.hint_accuracy:.2f}, "
                f"reward_prob={self.reward_prob:.2f}, "
                f"penalty={self.penalty:.1f}, "
                f"reversal_freq={self.trials_between_reversals}")


def calculate_active_learning_accuracy_curve(trials_between_reversals: int, 
                                             reward_prob: float) -> np.ndarray:
    """
    Model how agent accuracy improves over trials through active inference.
    
    Active inference agents start uncertain after a reversal, then gradually
    learn which arm is better through Bayesian belief updating. The learning
    rate depends on the feedback signal strength (reward_prob).
    
    Parameters:
    -----------
    trials_between_reversals : int
        Number of trials before the next context reversal
    reward_prob : float
        Probability of observing reward when choosing the correct arm.
        Higher values = stronger learning signal.
    
    Returns:
    --------
    accuracy_curve : np.ndarray
        Expected accuracy on each trial after a reversal (length = trials_between_reversals)
        
    Model assumptions:
    ------------------
    - Agent starts at 50% accuracy (random guessing) immediately after reversal
    - Accuracy improves exponentially toward asymptote
    - Learning rate is proportional to reward_prob (better feedback = faster learning)
    - Asymptotic accuracy is limited by reward_prob (noisy feedback = lower ceiling)
    """
    
    # Start at chance performance
    initial_accuracy = 0.5
    
    # Asymptotic accuracy: even with perfect learning, noisy feedback limits performance
    # With perfect feedback (reward_prob=1.0), agent can reach ~95% accuracy
    # With noisy feedback (reward_prob=0.6), agent plateaus around ~75% accuracy
    asymptotic_accuracy = 0.5 + (reward_prob * 0.45)
    
    # Learning rate: how quickly agent approaches asymptote
    # Higher reward_prob = faster learning (stronger signal)
    # Typical: reach 90% of asymptote in 4-8 trials
    learning_rate = 0.3 + (reward_prob * 0.4)
    
    # Generate accuracy curve using exponential approach to asymptote
    trials = np.arange(trials_between_reversals)
    accuracy_curve = asymptotic_accuracy - (asymptotic_accuracy - initial_accuracy) * np.exp(-learning_rate * trials)
    
    return accuracy_curve


def evaluate_pure_exploitation(regime: RegimeConfig) -> Dict:
    """
    Calculate expected value for "pure exploitation" strategy.
    
    Strategy: Never take hints. Just make direct choices and learn from feedback.
    This relies on active inference's Bayesian belief updating.
    
    Returns:
    --------
    results : dict
        - 'ev_per_trial': float, expected value averaged over reversal cycle
        - 'accuracy_curve': np.ndarray, accuracy on each trial
        - 'ev_curve': np.ndarray, expected value on each trial
        - 'total_rewards': float, total rewards over full cycle
    """
    
    # Get accuracy trajectory over the reversal cycle
    accuracy_curve = calculate_active_learning_accuracy_curve(
        regime.trials_between_reversals, 
        regime.reward_prob
    )
    
    # Calculate expected value for each trial
    # EV = P(correct) * reward_prob * reward + P(incorrect) * penalty
    # Note: reward on correct = reward_prob (not guaranteed), penalty on incorrect
    ev_curve = (
        accuracy_curve * regime.reward_prob * 1.0 +  # correct choice → chance of reward
        (1 - accuracy_curve) * regime.penalty          # incorrect choice → penalty
    )
    
    # Average over the full reversal cycle
    ev_per_trial = np.mean(ev_curve)
    total_rewards = np.sum(ev_curve)
    
    return {
        'strategy': 'Pure Exploitation',
        'ev_per_trial': ev_per_trial,
        'accuracy_curve': accuracy_curve,
        'ev_curve': ev_curve,
        'total_rewards': total_rewards,
        'trials_per_cycle': regime.trials_between_reversals
    }


def evaluate_pure_hint_following(regime: RegimeConfig) -> Dict:
    """
    Calculate expected value for "pure hint-following" strategy.
    
    Strategy: Alternate between taking hints and acting on them.
    - Trial 1: Take hint (observe hint, no choice, no reward)
    - Trial 2: Act on hint (make choice guided by hint)
    - Repeat...
    
    This strategy doesn't learn from feedback - it just follows hints.
    
    Returns:
    --------
    results : dict
        - 'ev_per_trial': float, expected value averaged across both hint and choice trials
        - 'ev_per_cycle': float, total expected value over full pattern (2 trials)
        - 'hint_usage': float, fraction of trials spent taking hints (0.5)
    """
    
    # Trial 1: Take hint
    # Reward: 0 (no choice made)
    ev_hint_trial = 0.0
    
    # Trial 2: Act on hint
    # The hint tells you the correct arm with probability hint_accuracy
    # If hint is correct, you choose correctly and might get reward
    # If hint is wrong, you choose incorrectly and get penalty
    ev_choice_trial = (
        regime.hint_accuracy * regime.reward_prob * 1.0 +  # hint correct → choose correctly → maybe reward
        (1 - regime.hint_accuracy) * regime.penalty         # hint wrong → choose incorrectly → penalty
    )
    
    # Average over the 2-trial pattern
    ev_per_cycle = ev_hint_trial + ev_choice_trial
    ev_per_trial = ev_per_cycle / 2.0
    
    # Scale up to full reversal cycle
    num_cycles_per_reversal = regime.trials_between_reversals // 2
    total_rewards = ev_per_cycle * num_cycles_per_reversal
    
    return {
        'strategy': 'Pure Hint-Following',
        'ev_per_trial': ev_per_trial,
        'ev_per_cycle': ev_per_cycle,
        'total_rewards': total_rewards,
        'hint_usage': 0.5,
        'trials_per_cycle': regime.trials_between_reversals
    }


def evaluate_mixed_strategy(regime: RegimeConfig, hint_frequency: float) -> Dict:
    """
    Calculate expected value for mixed strategy with specified hint frequency.
    
    Strategy: Take hints with some probability, otherwise exploit learned beliefs.
    This combines hint-based and feedback-based learning.
    
    Parameters:
    -----------
    hint_frequency : float
        Probability of taking a hint on any given trial (0 to 1)
        
    Model:
    ------
    On hint trials: EV = 0 (information only)
    On choice trials after recent hint: accuracy = hint_accuracy
    On choice trials with no recent hint: accuracy = active learning curve
    
    Simplified assumption: hints decay instantly, so we weight between
    hint-guided accuracy and active learning accuracy by hint_frequency.
    """
    
    # Get accuracy from active learning (when not using hints)
    active_accuracy_curve = calculate_active_learning_accuracy_curve(
        regime.trials_between_reversals,
        regime.reward_prob
    )
    active_mean_accuracy = np.mean(active_accuracy_curve)
    
    # Hint-guided accuracy
    hint_accuracy = regime.hint_accuracy
    
    # On hint trials: no reward
    ev_hint_trials = 0.0
    
    # On choice trials: weighted mixture of hint-guided and active learning
    # Simplification: assume hints are used uniformly throughout cycle
    effective_accuracy = (
        hint_frequency * hint_accuracy +           # when following hints
        (1 - hint_frequency) * active_mean_accuracy  # when using learned beliefs
    )
    
    ev_choice_trials = (
        effective_accuracy * regime.reward_prob * 1.0 +
        (1 - effective_accuracy) * regime.penalty
    )
    
    # Weighted average: hint_frequency trials are hints, rest are choices
    ev_per_trial = hint_frequency * ev_hint_trials + (1 - hint_frequency) * ev_choice_trials
    total_rewards = ev_per_trial * regime.trials_between_reversals
    
    return {
        'strategy': f'Mixed (hints {hint_frequency:.0%})',
        'ev_per_trial': ev_per_trial,
        'total_rewards': total_rewards,
        'hint_usage': hint_frequency,
        'effective_accuracy': effective_accuracy,
        'trials_per_cycle': regime.trials_between_reversals
    }


def analyze_regime(regime: RegimeConfig, verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate all strategies for a single regime and identify the best one.
    
    Returns:
    --------
    results_df : pd.DataFrame
        Comparison of all strategies with their expected values
    """
    
    results = []
    
    # Evaluate pure strategies
    exploit_result = evaluate_pure_exploitation(regime)
    results.append(exploit_result)
    
    hint_result = evaluate_pure_hint_following(regime)
    results.append(hint_result)
    
    # Evaluate mixed strategies at different hint frequencies
    for hint_freq in [0.25, 0.33, 0.5, 0.67, 0.75]:
        mixed_result = evaluate_mixed_strategy(regime, hint_freq)
        results.append(mixed_result)
    
    # Create comparison dataframe
    df = pd.DataFrame([{
        'Strategy': r['strategy'],
        'EV per Trial': r['ev_per_trial'],
        'Total Rewards': r['total_rewards'],
        'Hint Usage': r.get('hint_usage', 0.0)
    } for r in results])
    
    df = df.sort_values('EV per Trial', ascending=False)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"REGIME ANALYSIS: {regime.name}")
        print(f"{'='*70}")
        print(regime)
        print(f"\n{df.to_string(index=False)}")
        print(f"\n{'='*70}")
        best = df.iloc[0]
        print(f"BEST STRATEGY: {best['Strategy']}")
        print(f"  Expected Value: {best['EV per Trial']:.4f} per trial")
        print(f"  Total Rewards: {best['Total Rewards']:.2f} over {regime.trials_between_reversals} trials")
        print(f"  Hint Usage: {best['Hint Usage']:.1%}")
        print(f"{'='*70}")
    
    return df


def compare_regimes(regime1: RegimeConfig, regime2: RegimeConfig):
    """
    Compare two regimes to see if they favor different strategies.
    
    This is the key analysis: if both regimes favor the same strategy,
    then K=1 is optimal. If they favor different strategies, K>1 can help.
    """
    
    print("\n" + "="*80)
    print("MULTI-REGIME COMPARISON: Do different regimes favor different strategies?")
    print("="*80)
    
    # Analyze each regime
    df1 = analyze_regime(regime1, verbose=True)
    df2 = analyze_regime(regime2, verbose=True)
    
    # Identify best strategy in each regime
    best1 = df1.iloc[0]
    best2 = df2.iloc[0]
    
    # Critical comparison
    print("\n" + "="*80)
    print("CROSS-REGIME STRATEGY ANALYSIS")
    print("="*80)
    print(f"\n{regime1.name}:")
    print(f"  Best Strategy: {best1['Strategy']}")
    print(f"  EV per Trial: {best1['EV per Trial']:.4f}")
    print(f"  Hint Usage: {best1['Hint Usage']:.1%}")
    
    print(f"\n{regime2.name}:")
    print(f"  Best Strategy: {best2['Strategy']}")
    print(f"  EV per Trial: {best2['EV per Trial']:.4f}")
    print(f"  Hint Usage: {best2['Hint Usage']:.1%}")
    
    # Check if strategies differ
    print("\n" + "-"*80)
    if best1['Strategy'] == best2['Strategy']:
        print("⚠️  WARNING: BOTH REGIMES FAVOR THE SAME STRATEGY!")
        print(f"    Both favor: {best1['Strategy']}")
        print("\n    This means K=1 will be optimal - the agent doesn't need")
        print("    different behavioral profiles for different regimes.")
        print("\n    RECOMMENDATION: Adjust regime parameters to create distinct")
        print("    optimal strategies in each regime.")
    else:
        print("✅ SUCCESS: REGIMES FAVOR DIFFERENT STRATEGIES!")
        print(f"    {regime1.name}: {best1['Strategy']}")
        print(f"    {regime2.name}: {best2['Strategy']}")
        print("\n    This creates opportunity for K>1 agents to specialize profiles")
        print("    for different regimes and outperform K=1 agents.")
        
        # Calculate theoretical benefit of K=2
        # K=1: must use same strategy in both regimes (best is to use better of the two)
        regime1_duration = 120  # trials in regime 1
        regime2_duration = 80   # trials in regime 2
        
        # K=1 performance: use single strategy everywhere
        # Best K=1 would use whichever strategy has better weighted average
        k1_strategy1_total = (best1['EV per Trial'] * regime1_duration + 
                             df2[df2['Strategy'] == best1['Strategy']]['EV per Trial'].values[0] * regime2_duration)
        k1_strategy2_total = (df1[df1['Strategy'] == best2['Strategy']]['EV per Trial'].values[0] * regime1_duration +
                             best2['EV per Trial'] * regime2_duration)
        k1_best_total = max(k1_strategy1_total, k1_strategy2_total)
        
        # K=2 performance: use optimal strategy in each regime
        k2_total = best1['EV per Trial'] * regime1_duration + best2['EV per Trial'] * regime2_duration
        
        benefit = k2_total - k1_best_total
        
        print("\n" + "-"*80)
        print("THEORETICAL K=2 BENEFIT:")
        print(f"  K=1 best total: {k1_best_total:.2f} rewards")
        print(f"  K=2 optimal total: {k2_total:.2f} rewards")
        print(f"  Theoretical benefit: +{benefit:.2f} rewards ({100*benefit/(regime1_duration + regime2_duration):.2%} per trial)")
        
        if benefit > 0:
            print(f"\n  ✅ K=2 has theoretical advantage of {benefit:.1f} rewards")
            print("     If optimization finds K=1 better, there's likely an issue with:")
            print("     - Optimization not finding global optimum")
            print("     - Model architecture not allowing regime-specific adaptation")
            print("     - Belief uncertainty not aligning with regime changes")
        else:
            print(f"\n  ⚠️  K=2 theoretical advantage is minimal ({benefit:.1f} rewards)")
            print("     The environment may need stronger differentiation between regimes")
    
    print("="*80)
    
    return df1, df2


def visualize_regime_comparison(regime1: RegimeConfig, regime2: RegimeConfig):
    """
    Create visualization comparing strategy performance across regimes.
    """
    
    # Evaluate strategies
    exploit1 = evaluate_pure_exploitation(regime1)
    hint1 = evaluate_pure_hint_following(regime1)
    exploit2 = evaluate_pure_exploitation(regime2)
    hint2 = evaluate_pure_hint_following(regime2)
    
    # Calculate mixed strategies
    hint_freqs = np.linspace(0, 1, 21)
    mixed_evs_1 = [evaluate_mixed_strategy(regime1, hf)['ev_per_trial'] for hf in hint_freqs]
    mixed_evs_2 = [evaluate_mixed_strategy(regime2, hf)['ev_per_trial'] for hf in hint_freqs]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: EV vs Hint Frequency for both regimes
    ax1 = axes[0]
    ax1.plot(hint_freqs * 100, mixed_evs_1, 'b-', linewidth=2, label=regime1.name)
    ax1.plot(hint_freqs * 100, mixed_evs_2, 'r-', linewidth=2, label=regime2.name)
    
    # Mark optimal points
    opt_idx_1 = np.argmax(mixed_evs_1)
    opt_idx_2 = np.argmax(mixed_evs_2)
    ax1.plot(hint_freqs[opt_idx_1] * 100, mixed_evs_1[opt_idx_1], 'bo', markersize=10, label=f'{regime1.name} optimum')
    ax1.plot(hint_freqs[opt_idx_2] * 100, mixed_evs_2[opt_idx_2], 'ro', markersize=10, label=f'{regime2.name} optimum')
    
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Hint Usage (%)', fontsize=12)
    ax1.set_ylabel('Expected Value per Trial', fontsize=12)
    ax1.set_title('Strategy Performance Across Regimes', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning curves for pure exploitation
    ax2 = axes[1]
    
    acc_curve_1 = calculate_active_learning_accuracy_curve(regime1.trials_between_reversals, regime1.reward_prob)
    acc_curve_2 = calculate_active_learning_accuracy_curve(regime2.trials_between_reversals, regime2.reward_prob)
    
    ax2.plot(acc_curve_1, 'b-', linewidth=2, label=f'{regime1.name} (p_reward={regime1.reward_prob:.2f})')
    ax2.plot(acc_curve_2[:len(acc_curve_1)], 'r-', linewidth=2, label=f'{regime2.name} (p_reward={regime2.reward_prob:.2f})')
    ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='Chance')
    
    ax2.set_xlabel('Trial (within reversal cycle)', fontsize=12)
    ax2.set_ylabel('Expected Accuracy', fontsize=12)
    ax2.set_title('Active Learning Trajectories', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to results/figures directory
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'regime_comparison.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    
    return fig


def main():
    """
    Run complete analysis comparing regime configurations.
    """
    
    print("\n" + "="*80)
    print("ANALYTICAL STRATEGY EVALUATION")
    print("="*80)
    print("\nThis script calculates the theoretical expected value for different")
    print("behavioral strategies across regime configurations. Use this to verify")
    print("that your environment truly incentivizes different strategies before")
    print("running expensive parameter optimizations.")
    print("\n" + "="*80)
    
    # ==========================================================================
    # CONFIGURATION: Define your regime parameters here
    # ==========================================================================
    
    # STABLE REGIME: Feedback is reliable, hints are unreliable
    stable = RegimeConfig(
        name="STABLE",
        hint_accuracy=0.55,      # Hints barely better than chance
        reward_prob=0.85,        # Strong feedback signal for learning
        penalty=-1.0,            # Moderate penalty for mistakes
        trials_between_reversals=60  # Long time to learn
    )
    
    # VOLATILE REGIME: Hints are reliable, feedback is absent
    volatile = RegimeConfig(
        name="VOLATILE",
        hint_accuracy=0.95,      # Very accurate hints
        reward_prob=0.0,         # NO FEEDBACK - can't learn from choices!
        penalty=-2.0,            # Higher penalty for guessing wrong
        trials_between_reversals=8   # Frequent reversals
    )
    
    # Alternative volatile config to test
    volatile_alt = RegimeConfig(
        name="VOLATILE_ALT",
        hint_accuracy=0.95,
        reward_prob=0.65,        # Some feedback, but noisy
        penalty=-5.0,            # Very harsh penalty (makes guessing dangerous)
        trials_between_reversals=8
    )
    
    # ==========================================================================
    # RUN ANALYSIS
    # ==========================================================================
    
    # Compare main configuration
    df1, df2 = compare_regimes(stable, volatile)
    
    # Visualize
    fig = visualize_regime_comparison(stable, volatile)
    
    # Optional: test alternative configurations
    print("\n\n" + "="*80)
    print("TESTING ALTERNATIVE CONFIGURATION")
    print("="*80)
    df1_alt, df2_alt = compare_regimes(stable, volatile_alt)
    
    print("\n\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nUse this information to:")
    print("  1. Verify that your regimes favor different strategies")
    print("  2. Estimate the theoretical benefit of K>1 models")
    print("  3. Set expectations for what the optimizer should find")
    print("  4. Debug if empirical results don't match theoretical predictions")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()