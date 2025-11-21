"""
Model Comparison Experiment

Compares the three models (M1, M2, M3) on the two-armed bandit task
with reversals. Analyzes performance, adaptation speed, and behavior.
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import csv
import concurrent.futures

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.experiment_config import *
def _print_info(*args, **kwargs):
    print("[model_comparison]", *args, **kwargs)

def _print_debug(*args, **kwargs):
    print("[model_comparison DEBUG]", *args, **kwargs)
from src.environment import TwoArmedBandit
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL, run_episode_with_ll
from src.utils.model_utils import create_model, get_num_parameters, compute_metrics
from src.utils.simulate import simulate_baseline_run, run_single_agent
from src.utils.ll_eval import compute_sequence_ll_for_model, evaluate_ll_with_valuefn, _worker_init, compute_sequence_ll_for_model_worker, _eval_m1_gamma, _eval_m2_params, _eval_m3_params
from src.experiments.recovery import run_model_recovery, run_model_recovery_with_fitting
from src.utils import find_reversals, trial_accuracy, bootstrap_ci
from src.utils.plotting import plot_gamma_over_time, plot_entropy_over_time, plot_model_comparison



def run_comparison(num_runs=20, num_trials=DEFAULT_TRIALS, seed=42, reversal_interval=None):
    """Run comparison experiment across multiple runs."""
    
    _print_info("MODEL COMPARISON EXPERIMENT")
    _print_info(f"Runs per model: {num_runs}")
    _print_info(f"Trials per run: {num_trials}")
    _print_info(f"Random seed: {seed}")
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

    # Reference generator types to evaluate against
    # include generators derived from M1 and M2 plus simple baselines
    reference_types = ['expert', 'noisy', 'perturbed', 'M1', 'M2', 'egreedy', 'softmax']

    for ref_type in reference_types:
        _print_info(f"Generating reference rollouts (type: {ref_type})...")
        reference_runs = []
        rng = np.random.RandomState(seed)
        for run in tqdm(range(num_runs), desc=f"  reference:{ref_type}"):
            run_seed = seed + run if seed is not None else None
            # Build environment and generator depending on ref_type
            if ref_type == 'perturbed':
                # decrease hint reliability and inject extra random reversals
                extra_reversals = []
                rrng = np.random.RandomState(run_seed)
                for _ in range(max(1, num_trials // 200)):
                    extra_reversals.append(int(rrng.randint(0, num_trials)))

                merged_reversal_schedule = sorted(set(list(reversal_schedule) + extra_reversals))
                env = TwoArmedBandit(
                    probability_hint=max(0.1, PROBABILITY_HINT - 0.3),
                    probability_reward=PROBABILITY_REWARD,
                    reversal_schedule=merged_reversal_schedule
                )
                # use standard M3 value function
                value_fn = create_model('M3', A, B, D)
                runner = AgentRunnerWithLL(A, B, D, value_fn,
                                           OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                           OBSERVATION_CHOICES, ACTION_CHOICES,
                                           reward_mod_idx=1)

                # moderate stochasticity for perturbed
                try:
                    runner.agent.action_selection = 'stochastic'
                    runner.agent.gamma = 4.0
                except Exception as e:
                    logging.warning("Could not configure runner.agent for perturbed generator: %s", e)

                ref_logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)
                reference_runs.append(ref_logs)

            elif ref_type in ('expert', 'noisy'):
                # Use base environment, but configure M3 agent's selection behavior
                env = TwoArmedBandit(
                    probability_hint=PROBABILITY_HINT,
                    probability_reward=PROBABILITY_REWARD,
                    reversal_schedule=reversal_schedule
                )
                value_fn = create_model('M3', A, B, D)
                runner = AgentRunnerWithLL(A, B, D, value_fn,
                                           OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                           OBSERVATION_CHOICES, ACTION_CHOICES,
                                           reward_mod_idx=1)

                if ref_type == 'expert':
                    # make near-deterministic expert (argmax-like behaviour)
                    try:
                        runner.agent.action_selection = 'deterministic'
                    except Exception as e:
                        logging.warning("Could not set agent.action_selection to deterministic: %s", e)
                    try:
                        runner.agent.gamma = 100.0
                    except Exception as e:
                        logging.warning("Could not set agent.gamma to 100.0: %s", e)
                else:
                    try:
                        runner.agent.action_selection = 'stochastic'
                    except Exception as e:
                        logging.warning("Could not set agent.action_selection to stochastic: %s", e)
                    try:
                        runner.agent.gamma = 2.0
                    except Exception as e:
                        logging.warning("Could not set agent.gamma to 2.0: %s", e)

                ref_logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)
                reference_runs.append(ref_logs)

            elif ref_type in ('M1', 'M2'):
                # Generate reference trajectories from other candidate models
                env = TwoArmedBandit(
                    probability_hint=PROBABILITY_HINT,
                    probability_reward=PROBABILITY_REWARD,
                    reversal_schedule=reversal_schedule
                )
                value_fn = create_model(ref_type, A, B, D)
                runner = AgentRunnerWithLL(A, B, D, value_fn,
                                           OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                           OBSERVATION_CHOICES, ACTION_CHOICES,
                                           reward_mod_idx=1)
                # Use deterministic-ish selection for these generators to expose
                # model-specific behaviour (but keep some stochasticity by default)
                try:
                    runner.agent.action_selection = 'stochastic'
                    runner.agent.gamma = 8.0
                except Exception as e:
                    logging.warning("Could not configure runner.agent for M1/M2 generator: %s", e)

                ref_logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)
                reference_runs.append(ref_logs)

            elif ref_type in ('egreedy', 'softmax'):
                env = TwoArmedBandit(
                    probability_hint=PROBABILITY_HINT,
                    probability_reward=PROBABILITY_REWARD,
                    reversal_schedule=reversal_schedule
                )
                # Simple baseline controller simulated directly
                ref_logs = simulate_baseline_run(env, policy_type=ref_type, T=num_trials, seed=run_seed,
                                                 epsilon=0.1, temp=1.0, alpha=0.1)
                reference_runs.append(ref_logs)

            else:
                # fallback: base M3 rollout
                env = TwoArmedBandit(
                    probability_hint=PROBABILITY_HINT,
                    probability_reward=PROBABILITY_REWARD,
                    reversal_schedule=reversal_schedule
                )
                value_fn = create_model('M3', A, B, D)
                runner = AgentRunnerWithLL(A, B, D, value_fn,
                                           OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                           OBSERVATION_CHOICES, ACTION_CHOICES,
                                           reward_mod_idx=1)
                ref_logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)
                reference_runs.append(ref_logs)

        # Evaluate each candidate model on the same reference rollouts (teacher-forcing)
        _print_info(f"Evaluating models against reference type '{ref_type}' using up to {os.cpu_count() or 1} workers...")
        tasks = []
        for model_name in models:
            for run_idx, ref_logs in enumerate(reference_runs):
                tasks.append((model_name, ref_logs, ref_type))

        # Use ProcessPoolExecutor to parallelize evaluations across models/runs
        max_workers = int(os.environ.get('MODEL_COMP_MAX_WORKERS', min(len(tasks), os.cpu_count() or 1)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init, initargs=(A, B, D)) as exe:
            # submit tasks without re-sending A/B/D (workers have them via init)
            future_map = {exe.submit(compute_sequence_ll_for_model_worker, model_name, ref_logs): (model_name, ref_logs, ref_type) for (model_name, ref_logs, ref_type) in tasks}
            for fut in tqdm(concurrent.futures.as_completed(future_map), total=len(future_map), desc=f"  eval:{ref_type}"):
                model_name, ref_logs, ref_type = future_map[fut]
                ll_seq = fut.result()
                # compute metrics in main process (avoid sending compute_metrics to workers)
                new_logs = dict(ref_logs)
                new_logs['ll'] = ll_seq
                metrics = compute_metrics(new_logs)
                metrics['reference_type'] = ref_type
                metrics['model_name'] = model_name
                _print_debug(f"Collected metrics for {metrics.get('model_name')} run (ref={metrics.get('reference_type')}): mean_accuracy={metrics.get('mean_accuracy', np.nan):.3f} ll={metrics.get('log_likelihood', np.nan):.2f}")
                results[metrics['model_name']].append(metrics)
    
    _print_info("RESULTS SUMMARY")
    
    # Print statistics grouped by reference type
    for model_name in models:
        model_results = results[model_name]
        k = get_num_parameters(model_name)

        # Determine which reference types are present
        ref_types_present = sorted(set(r.get('reference_type', 'default') for r in model_results))
        for ref_type in ref_types_present:
            subset = [r for r in model_results if r.get('reference_type', 'default') == ref_type]

            mean_acc = np.mean([r['mean_accuracy'] for r in subset])
            std_acc = np.std([r['mean_accuracy'] for r in subset])

            mean_reward = np.mean([r['total_reward'] for r in subset])
            std_reward = np.std([r['total_reward'] for r in subset])

            mean_adapt = np.nanmean([r['adaptation_time'] for r in subset])

            mean_gamma = np.mean([r['gamma_mean'] for r in subset])

            # Log-likelihood and model fit metrics
            ll_vals = np.array([r['log_likelihood'] for r in subset])
            mean_ll = ll_vals.mean()
            std_ll = ll_vals.std()
            # 95% CI via bootstrap helper
            _, ll_ci_lower, ll_ci_upper = bootstrap_ci(ll_vals, n_bootstrap=5000, ci=95)

            n = subset[0]['num_trials']  # Number of observations
            # Compute per-run AIC/BIC then summarize
            aic_per_run = 2 * k - 2 * ll_vals
            bic_per_run = k * np.log(n) - 2 * ll_vals

            mean_aic = aic_per_run.mean()
            std_aic = aic_per_run.std()
            mean_bic = bic_per_run.mean()
            std_bic = bic_per_run.std()

            print(f"\n{model_name} (reference: {ref_type}):")
            print(f"  Parameters:     {k}")
            print(f"  Accuracy:       {mean_acc:.3f} ± {std_acc:.3f}")
            print(f"  Total Reward:   {mean_reward:.1f} ± {std_reward:.1f}")
            print(f"  Adaptation:     {mean_adapt:.1f} trials")
            print(f"  Mean γ:         {mean_gamma:.3f}")
            print(f"  Log-Likelihood: {mean_ll:.2f} ± {std_ll:.2f}  (95% CI: [{ll_ci_lower:.2f}, {ll_ci_upper:.2f}])")
            print(f"  AIC:            {mean_aic:.2f} ± {std_aic:.2f}")
            print(f"  BIC:            {mean_bic:.2f} ± {std_bic:.2f}")
    
    # Paired ΔNLL (within-reference) summaries
    print("\n" + "="*70)
    print("PAIRED ΔNLL (within each reference type)")
    print("="*70)
    for ref_type in sorted({r.get('reference_type', 'default') for model in models for r in results[model]}):
        print(f"\nReference type: {ref_type}")
        # collect per-model ll arrays aligned by run index
        per_model_ll = {}
        for model_name in models:
            vals = [r['log_likelihood'] for r in results.get(model_name, []) if r.get('reference_type') == ref_type]
            per_model_ll[model_name] = np.array(vals)

        # ensure same length across models (if not, skip paired stats)
        lengths = [len(v) for v in per_model_ll.values()]
        if len(set(lengths)) != 1 or lengths[0] == 0:
            print(f"  Skipping paired ΔNLL (unequal or empty runs): lengths={lengths}")
            continue

        n_runs = lengths[0]
        # compute pairwise differences and bootstrap CI
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                m_i = models[i]
                m_j = models[j]
                delta = per_model_ll[m_i] - per_model_ll[m_j]
                mean_d, ci_lo, ci_hi = bootstrap_ci(delta, n_bootstrap=5000, ci=95)
                print(f"  ΔNLL {m_i} - {m_j}: {mean_d:.3f} (95% CI [{ci_lo:.3f}, {ci_hi:.3f}])")

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

    # Save per-run metrics to CSV for this interval
    csv_dir = os.path.join('results', 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    if reversal_interval is None:
        csv_path = os.path.join(csv_dir, f'model_comparison_default.csv')
    else:
        csv_path = os.path.join(csv_dir, f'model_comparison_interval_{reversal_interval}.csv')

    # Prepare header and rows
    header = ['interval', 'reference_type', 'model', 'run_idx', 'parameters', 'mean_accuracy', 'total_reward', 'adaptation_time', 'mean_gamma', 'log_likelihood', 'aic', 'bic']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for model_name in models:
            model_results = results[model_name]
            k = get_num_parameters(model_name)
            n = model_results[0]['num_trials']
            for run_idx, r in enumerate(model_results):
                ll_i = r['log_likelihood']
                aic_i = 2 * k - 2 * ll_i
                bic_i = k * np.log(n) - 2 * ll_i
                writer.writerow([reversal_interval if reversal_interval is not None else 'default',
                                 r.get('reference_type', 'default'),
                                 model_name, run_idx, k,
                                 r['mean_accuracy'], r['total_reward'], r['adaptation_time'], r['gamma_mean'],
                                 ll_i, aic_i, bic_i])

    print(f"Per-run CSV saved: {csv_path}")
    
    return results

def main():
    """Main entry point."""
    
    # Run comparison across several reversal-interval settings
    reversal_intervals = [40, 80, 160]
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
