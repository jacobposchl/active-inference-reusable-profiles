"""
Model Comparison Experiment

Compares the three models (M1, M2, M3) on the two-armed bandit task
with reversals. Analyzes performance, adaptation speed, and behavior.
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
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL, run_episode, run_episode_with_ll
from src.utils import find_reversals, trial_accuracy, bootstrap_ci
from src.utils.plotting import plot_gamma_over_time, plot_entropy_over_time


def create_model(model_name, A, B, D):
    """Create value function for specified model."""
    
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
        # Get policies from temporary agent
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
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return value_fn


def run_single_agent(model_name, A, B, D, num_trials, seed=None, reversal_schedule=None):
    """Run a single agent for one episode."""
    
    if seed is not None:
        np.random.seed(seed)
    
    # Create fresh environment for this run
    env = TwoArmedBandit(
        probability_hint=PROBABILITY_HINT,
        probability_reward=PROBABILITY_REWARD,
        reversal_schedule=reversal_schedule if reversal_schedule is not None else DEFAULT_REVERSAL_SCHEDULE
    )
    
    # Create value function
    value_fn = create_model(model_name, A, B, D)
    
    # Create agent runner with log-likelihood tracking
    runner = AgentRunnerWithLL(A, B, D, value_fn,
                        OBSERVATION_HINTS, OBSERVATION_REWARDS,
                        OBSERVATION_CHOICES, ACTION_CHOICES,
                        reward_mod_idx=1)
    
    # Run episode WITH log-likelihood tracking
    logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)
    
    return logs


def get_num_parameters(model_name):
    """Get number of free parameters for each model."""
    
    if model_name == 'M1':
        # C_reward_logits: 3 values (but one is constrained, so 2 free)
        # gamma: 1 value
        return 3
        
    elif model_name == 'M2':
        # C_reward_logits: 2 free parameters
        # gamma_base: 1 parameter
        # entropy_k: 1 parameter
        return 4
        
    elif model_name == 'M3':
        # Careful count of free parameters for M3:
        # For each profile:
        #  - phi_logits: 3 outcome logits -> 2 free parameters (sum-to-1 via softmax)
        #  - xi_logits: 4 action logits -> 3 free parameters
        #  - gamma: 1 parameter
        # For 2 profiles: (2 + 3 + 1) * 2 = 12
        # Z matrix: 2x2 with each row summing to 1 -> 1 free param per row -> 2
        # Total free parameters = 12 + 2 = 14
        return 14
    
    return 0


def compute_metrics(logs):
    """Compute performance metrics from episode logs."""
    
    # Accuracy
    acc = trial_accuracy(logs['action'], logs['context'])
    
    # Reversals
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
    
    # Log-likelihood
    log_likelihood = np.sum(logs['ll'])
    num_trials = len(logs['ll'])
    
    # Adaptation speed (trials to recover after reversal)
    adaptation_times = []
    for rev_t in reversals:
        # Look at accuracy in window after reversal
        window = min(20, len(acc) - rev_t)
        if window > 5:
            post_rev_acc = acc[rev_t:rev_t+window]
            # Find when accuracy crosses 0.7 threshold
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


def run_comparison(num_runs=20, num_trials=DEFAULT_TRIALS, seed=42, reversal_interval=None):
    """Run comparison experiment across multiple runs."""
    
    print("="*70)
    print("MODEL COMPARISON EXPERIMENT")
    print("="*70)
    print(f"Runs per model: {num_runs}")
    print(f"Trials per run: {num_trials}")
    print(f"Random seed: {seed}")
    if reversal_interval is None:
        print(f"Reversal schedule: default ({len(DEFAULT_REVERSAL_SCHEDULE)} reversals)")
    else:
        print(f"Reversal interval: every {reversal_interval} trials")
    print()
    
    # Build shared components
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)
    
    models = ['M1', 'M2', 'M3']
    results = {model: [] for model in models}
    
    # Build reversal schedule if provided
    if reversal_interval is not None:
        reversal_schedule = [i for i in range(reversal_interval, num_trials, reversal_interval)]
    else:
        reversal_schedule = DEFAULT_REVERSAL_SCHEDULE

    # Run experiments
    for model_name in models:
        print(f"Running {model_name}...")
        
        for run in tqdm(range(num_runs), desc=f"  {model_name}"):
            run_seed = seed + run if seed is not None else None
            logs = run_single_agent(model_name, A, B, D, num_trials, seed=run_seed, reversal_schedule=reversal_schedule)
            metrics = compute_metrics(logs)
            results[model_name].append(metrics)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Print statistics
    for model_name in models:
        model_results = results[model_name]
        k = get_num_parameters(model_name)
        
        mean_acc = np.mean([r['mean_accuracy'] for r in model_results])
        std_acc = np.std([r['mean_accuracy'] for r in model_results])
        
        mean_reward = np.mean([r['total_reward'] for r in model_results])
        std_reward = np.std([r['total_reward'] for r in model_results])
        
        mean_adapt = np.nanmean([r['adaptation_time'] for r in model_results])
        
        mean_gamma = np.mean([r['gamma_mean'] for r in model_results])
        
        # Log-likelihood and model fit metrics
    ll_vals = np.array([r['log_likelihood'] for r in model_results])
    mean_ll = ll_vals.mean()
    std_ll = ll_vals.std()
    # 95% CI via bootstrap helper
    _, ll_ci_lower, ll_ci_upper = bootstrap_ci(ll_vals, n_bootstrap=5000, ci=95)

    n = model_results[0]['num_trials']  # Number of observations
    # Compute per-run AIC/BIC then summarize
    aic_per_run = 2 * k - 2 * ll_vals
    bic_per_run = k * np.log(n) - 2 * ll_vals

    mean_aic = aic_per_run.mean()
    std_aic = aic_per_run.std()
    mean_bic = bic_per_run.mean()
    std_bic = bic_per_run.std()

    print(f"\n{model_name}:")
    print(f"  Parameters:     {k}")
    print(f"  Accuracy:       {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"  Total Reward:   {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"  Adaptation:     {mean_adapt:.1f} trials")
    print(f"  Mean γ:         {mean_gamma:.3f}")
    print(f"  Log-Likelihood: {mean_ll:.2f} ± {std_ll:.2f}  (95% CI: [{ll_ci_lower:.2f}, {ll_ci_upper:.2f}])")
    print(f"  AIC:            {mean_aic:.2f} ± {std_aic:.2f}")
    print(f"  BIC:            {mean_bic:.2f} ± {std_bic:.2f}")
    
    # Model comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON (Lower is better for AIC/BIC)")
    print("="*70)
    
    aic_values = {}
    bic_values = {}
    ll_values = {}
    
    for model_name in models:
        model_results = results[model_name]
        k = get_num_parameters(model_name)
        n = model_results[0]['num_trials']
        
        mean_ll = np.mean([r['log_likelihood'] for r in model_results])
        aic_values[model_name] = 2 * k - 2 * mean_ll
        bic_values[model_name] = k * np.log(n) - 2 * mean_ll
        ll_values[model_name] = mean_ll
    
    best_aic = min(aic_values, key=aic_values.get)
    best_bic = min(bic_values, key=bic_values.get)
    best_ll = max(ll_values, key=ll_values.get)
    
    print(f"\nBest Log-Likelihood: {best_ll} ({ll_values[best_ll]:.2f})")
    print(f"Best AIC:            {best_aic} ({aic_values[best_aic]:.2f})")
    print(f"Best BIC:            {best_bic} ({bic_values[best_bic]:.2f})")
    
    # Compute AIC/BIC differences
    print(f"\nΔAIC from best:")
    for model in models:
        delta = aic_values[model] - aic_values[best_aic]
        print(f"  {model}: {delta:+.2f}")
    
    print(f"\nΔBIC from best:")
    for model in models:
        delta = bic_values[model] - bic_values[best_bic]
        print(f"  {model}: {delta:+.2f}")
    
    # Plot comparison
    plot_model_comparison(results, num_trials, reversal_interval=reversal_interval)
    
    return results


def plot_model_comparison(results, num_trials, reversal_interval=None):
    """Generate comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = ['M1', 'M2', 'M3']
    colors = {'M1': 'blue', 'M2': 'green', 'M3': 'red'}
    
    # Plot 1: Accuracy over time (averaged)
    ax = axes[0, 0]
    for model_name in models:
        model_results = results[model_name]
        
        # Average accuracy across runs
        all_accs = np.array([r['accuracy'] for r in model_results])
        mean_acc = all_accs.mean(axis=0)
        std_acc = all_accs.std(axis=0)
        
        # Rolling average
        window = ROLLING_WINDOW
        mean_acc_smooth = np.convolve(mean_acc, np.ones(window)/window, mode='valid')
        
        ax.plot(mean_acc_smooth, label=model_name, color=colors[model_name], linewidth=2)
        
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Accuracy (rolling avg)')
    ax.set_title('Performance Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy distribution
    ax = axes[0, 1]
    acc_data = [[r['mean_accuracy'] for r in results[m]] for m in models]
    bp = ax.boxplot(acc_data, tick_labels=models, patch_artist=True)
    for patch, model in zip(bp['boxes'], models):
        patch.set_facecolor(colors[model])
        patch.set_alpha(0.6)
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Accuracy Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Total rewards
    ax = axes[1, 0]
    reward_data = [[r['total_reward'] for r in results[m]] for m in models]
    bp = ax.boxplot(reward_data, tick_labels=models, patch_artist=True)
    for patch, model in zip(bp['boxes'], models):
        patch.set_facecolor(colors[model])
        patch.set_alpha(0.6)
    ax.set_ylabel('Total Reward')
    ax.set_title('Cumulative Rewards')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Gamma dynamics (one example run)
    ax = axes[1, 1]
    for model_name in models:
        gamma_series = results[model_name][0]['logs']['gamma']
        window = 5
        gamma_smooth = np.convolve(gamma_series, np.ones(window)/window, mode='valid')
        ax.plot(gamma_smooth, label=model_name, color=colors[model_name], linewidth=2)
    
    ax.set_xlabel('Trial')
    ax.set_ylabel('Policy Precision (γ)')
    ax.set_title('Precision Dynamics (Example Run)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if reversal_interval is None:
        outpath = 'results/figures/model_comparison.png'
    else:
        outpath = f'results/figures/model_comparison_interval_{reversal_interval}.png'
    plt.savefig(outpath, dpi=FIG_DPI, bbox_inches='tight')
    print(f"\nPlot saved: {outpath}")
    plt.show()


def main():
    """Main entry point."""
    
    # Run comparison across several reversal-interval settings
    reversal_intervals = [40, 80, 160, 320]
    all_results = {}
    for interval in reversal_intervals:
        print("\n" + "#"*70)
        print(f"Running experiment with reversal interval = {interval}")
        print("#"*70 + "\n")
        results = run_comparison(num_runs=20, num_trials=DEFAULT_TRIALS, seed=42, reversal_interval=interval)
        all_results[interval] = results
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
