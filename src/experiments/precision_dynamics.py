"""
Precision Dynamics Analysis

Analyzes how policy precision (gamma) changes over time,
particularly around reversals and in response to uncertainty.

Focuses on:
- M2: Entropy-based gamma modulation
- M3: Belief-dependent profile mixing effects on gamma
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.experiment_config import *
from src.environment import TwoArmedBandit
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunner, run_episode
from src.utils import find_reversals, trial_accuracy


def compute_entropy(belief):
    """Compute entropy of a probability distribution."""
    p = np.clip(np.asarray(belief, float), 1e-12, 1.0)
    return -(p * np.log(p)).sum()


def analyze_reversal_dynamics(logs, window=15):
    """Analyze gamma and belief dynamics around reversals."""
    
    reversals = find_reversals(logs['context'])
    
    gamma_series = np.array(logs['gamma'])
    belief_series = logs['belief']
    
    # Extract windows around reversals
    reversal_windows = {
        'pre_gamma': [],
        'post_gamma': [],
        'pre_entropy': [],
        'post_entropy': [],
        'pre_belief': [],
        'post_belief': []
    }
    
    for rev_t in reversals:
        # Pre-reversal window
        pre_start = max(0, rev_t - window)
        pre_end = rev_t
        
        # Post-reversal window
        post_start = rev_t
        post_end = min(len(gamma_series), rev_t + window)
        
        if pre_end > pre_start and post_end > post_start:
            # Gamma
            reversal_windows['pre_gamma'].append(gamma_series[pre_start:pre_end])
            reversal_windows['post_gamma'].append(gamma_series[post_start:post_end])
            
            # Entropy
            pre_entropies = [compute_entropy(b) for b in belief_series[pre_start:pre_end]]
            post_entropies = [compute_entropy(b) for b in belief_series[post_start:post_end]]
            reversal_windows['pre_entropy'].append(pre_entropies)
            reversal_windows['post_entropy'].append(post_entropies)
            
            # Belief in left_better
            pre_beliefs = [b[0] for b in belief_series[pre_start:pre_end]]
            post_beliefs = [b[0] for b in belief_series[post_start:post_end]]
            reversal_windows['pre_belief'].append(pre_beliefs)
            reversal_windows['post_belief'].append(post_beliefs)
    
    # Average across reversals
    avg_dynamics = {}
    for key in reversal_windows:
        if reversal_windows[key]:
            # Pad to same length
            max_len = max(len(w) for w in reversal_windows[key])
            padded = []
            for w in reversal_windows[key]:
                if len(w) < max_len:
                    pad_val = w[-1] if len(w) > 0 else 0
                    w_padded = list(w) + [pad_val] * (max_len - len(w))
                else:
                    w_padded = w
                padded.append(w_padded)
            avg_dynamics[key] = np.mean(padded, axis=0)
        else:
            avg_dynamics[key] = np.array([])
    
    return avg_dynamics, reversals


def analyze_gamma_entropy_correlation(logs):
    """Analyze correlation between gamma and belief entropy."""
    
    gamma_series = np.array(logs['gamma'])
    entropy_series = np.array([compute_entropy(b) for b in logs['belief']])
    
    # Remove NaNs
    valid_idx = ~(np.isnan(gamma_series) | np.isnan(entropy_series))
    gamma_clean = gamma_series[valid_idx]
    entropy_clean = entropy_series[valid_idx]
    
    if len(gamma_clean) > 2:
        corr, p_value = pearsonr(gamma_clean, entropy_clean)
    else:
        corr, p_value = np.nan, np.nan
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'gamma': gamma_clean,
        'entropy': entropy_clean
    }


def run_precision_analysis(model_name='M3', num_trials=DEFAULT_TRIALS, seed=42):
    """Run precision dynamics analysis for a specific model."""
    
    print("="*70)
    print(f"PRECISION DYNAMICS ANALYSIS - {model_name}")
    print("="*70)
    
    np.random.seed(seed)
    
    # Build environment and model
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)
    
    env = TwoArmedBandit(
        probability_hint=PROBABILITY_HINT,
        probability_reward=PROBABILITY_REWARD,
        reversal_schedule=DEFAULT_REVERSAL_SCHEDULE
    )
    
    # Create value function
    if model_name == 'M1':
        value_fn = make_value_fn('M1', **M1_DEFAULTS)
        
    elif model_name == 'M2':
        def gamma_schedule(q, t, g_base=M2_DEFAULTS['gamma_base'], 
                          k=M2_DEFAULTS['entropy_k']):
            p = np.clip(np.asarray(q, float), 1e-12, 1.0)
            H = -(p * np.log(p)).sum()
            return g_base / (1.0 + k * H)
        
        value_fn = make_value_fn('M2', 
                                C_reward_logits=M2_DEFAULTS['C_reward_logits'],
                                gamma_schedule=gamma_schedule)
        
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
        
        value_fn = make_value_fn('M3',
                                profiles=M3_DEFAULTS['profiles'],
                                Z=np.array(M3_DEFAULTS['Z']),
                                policies=policies,
                                num_actions_per_factor=num_actions_per_factor)
    
    # Create agent and run
    runner = AgentRunner(A, B, D, value_fn,
                        OBSERVATION_HINTS, OBSERVATION_REWARDS,
                        OBSERVATION_CHOICES, ACTION_CHOICES,
                        reward_mod_idx=1)
    
    print(f"Running {num_trials} trials...")
    logs = run_episode(runner, env, T=num_trials, verbose=False)
    
    # Analyze dynamics
    print("\nAnalyzing reversal dynamics...")
    reversal_dynamics, reversals = analyze_reversal_dynamics(logs, window=15)
    
    print("\nAnalyzing gamma-entropy correlation...")
    correlation_results = analyze_gamma_entropy_correlation(logs)
    
    # Print statistics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Number of reversals: {len(reversals)}")
    print(f"Gamma range: [{np.min(logs['gamma']):.3f}, {np.max(logs['gamma']):.3f}]")
    print(f"Mean gamma: {np.mean(logs['gamma']):.3f}")
    print(f"Gamma std: {np.std(logs['gamma']):.3f}")
    
    if not np.isnan(correlation_results['correlation']):
        print(f"\nGamma-Entropy Correlation: {correlation_results['correlation']:.3f}")
        print(f"P-value: {correlation_results['p_value']:.4f}")
    
    # Generate plots
    plot_precision_dynamics(logs, reversal_dynamics, correlation_results, model_name)
    
    return logs, reversal_dynamics, correlation_results


def plot_precision_dynamics(logs, reversal_dynamics, correlation_results, model_name):
    """Generate precision dynamics plots."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Gamma over time
    ax1 = fig.add_subplot(gs[0, :])
    gamma_series = logs['gamma']
    reversals = find_reversals(logs['context'])
    
    ax1.plot(gamma_series, linewidth=1.5, color='purple', alpha=0.7)
    window = 7
    gamma_smooth = np.convolve(gamma_series, np.ones(window)/window, mode='valid')
    ax1.plot(range(window//2, len(gamma_series) - window//2), gamma_smooth, 
             linewidth=2, color='darkviolet', label='Rolling avg')
    
    for rev in reversals:
        ax1.axvline(rev, color='red', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Policy Precision (γ)')
    ax1.set_title(f'Gamma Dynamics - {model_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Belief entropy over time
    ax2 = fig.add_subplot(gs[1, :])
    entropy_series = [compute_entropy(b) for b in logs['belief']]
    ax2.plot(entropy_series, linewidth=1.5, color='orange', alpha=0.7)
    
    entropy_smooth = np.convolve(entropy_series, np.ones(window)/window, mode='valid')
    ax2.plot(range(window//2, len(entropy_series) - window//2), entropy_smooth,
             linewidth=2, color='darkorange', label='Rolling avg')
    
    for rev in reversals:
        ax2.axvline(rev, color='red', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Belief Entropy')
    ax2.set_title('Belief Uncertainty Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gamma around reversals
    ax3 = fig.add_subplot(gs[2, 0])
    if len(reversal_dynamics['pre_gamma']) > 0:
        pre_gamma = reversal_dynamics['pre_gamma']
        post_gamma = reversal_dynamics['post_gamma']
        
        x_pre = np.arange(-len(pre_gamma), 0)
        x_post = np.arange(0, len(post_gamma))
        
        ax3.plot(x_pre, pre_gamma, 'o-', color='blue', label='Pre-reversal')
        ax3.plot(x_post, post_gamma, 's-', color='red', label='Post-reversal')
        ax3.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Trials from reversal')
        ax3.set_ylabel('γ')
        ax3.set_title('Gamma Around Reversals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Entropy around reversals
    ax4 = fig.add_subplot(gs[2, 1])
    if len(reversal_dynamics['pre_entropy']) > 0:
        pre_entropy = reversal_dynamics['pre_entropy']
        post_entropy = reversal_dynamics['post_entropy']
        
        x_pre = np.arange(-len(pre_entropy), 0)
        x_post = np.arange(0, len(post_entropy))
        
        ax4.plot(x_pre, pre_entropy, 'o-', color='blue', label='Pre-reversal')
        ax4.plot(x_post, post_entropy, 's-', color='red', label='Post-reversal')
        ax4.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Trials from reversal')
        ax4.set_ylabel('Entropy')
        ax4.set_title('Entropy Around Reversals')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Gamma vs Entropy scatter
    ax5 = fig.add_subplot(gs[2, 2])
    if not np.isnan(correlation_results['correlation']):
        gamma = correlation_results['gamma']
        entropy = correlation_results['entropy']
        
        ax5.scatter(entropy, gamma, alpha=0.3, s=20, color='purple')
        
        # Fit line
        z = np.polyfit(entropy, gamma, 1)
        p = np.poly1d(z)
        x_line = np.linspace(entropy.min(), entropy.max(), 100)
        ax5.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8,
                label=f'r = {correlation_results["correlation"]:.3f}')
        
        ax5.set_xlabel('Belief Entropy')
        ax5.set_ylabel('Policy Precision (γ)')
        ax5.set_title('Gamma vs Entropy')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    plt.savefig(f'results/figures/precision_dynamics_{model_name}.png', 
                dpi=FIG_DPI, bbox_inches='tight')
    print(f"\nPlot saved: results/figures/precision_dynamics_{model_name}.png")
    plt.show()


def compare_models_precision():
    """Compare precision dynamics across all three models."""
    
    print("\n" + "="*70)
    print("COMPARING PRECISION DYNAMICS ACROSS MODELS")
    print("="*70)
    
    models = ['M1', 'M2', 'M3']
    all_results = {}
    
    for model in models:
        print(f"\n{'='*70}")
        logs, reversal_dynamics, correlation = run_precision_analysis(
            model_name=model, 
            num_trials=DEFAULT_TRIALS, 
            seed=42
        )
        all_results[model] = {
            'logs': logs,
            'reversal_dynamics': reversal_dynamics,
            'correlation': correlation
        }
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {'M1': 'blue', 'M2': 'green', 'M3': 'red'}
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        logs = all_results[model]['logs']
        gamma_series = logs['gamma']
        reversals = find_reversals(logs['context'])
        
        ax.plot(gamma_series, linewidth=1, alpha=0.5, color=colors[model])
        window = 7
        gamma_smooth = np.convolve(gamma_series, np.ones(window)/window, mode='valid')
        ax.plot(range(window//2, len(gamma_series) - window//2), gamma_smooth,
                linewidth=2, color=colors[model])
        
        for rev in reversals:
            ax.axvline(rev, color='red', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Trial')
        ax.set_ylabel('γ')
        ax.set_title(f'{model}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(4, np.max(gamma_series) * 1.1)])
    
    plt.tight_layout()
    plt.savefig('results/figures/precision_comparison_all_models.png', 
                dpi=FIG_DPI, bbox_inches='tight')
    print(f"\nComparison plot saved: results/figures/precision_comparison_all_models.png")
    plt.show()


def main():
    """Main entry point."""
    
    # Analyze M3 in detail
    run_precision_analysis(model_name='M3', num_trials=DEFAULT_TRIALS, seed=42)
    
    # Compare all models
    # compare_models_precision()  # Uncomment to run full comparison
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
