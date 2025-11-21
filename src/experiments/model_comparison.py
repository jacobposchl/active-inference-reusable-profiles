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
import csv

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


def simulate_baseline_run(env, policy_type, T=200, seed=None, epsilon=0.1, temp=1.0, alpha=0.1):
    """Simulate a simple baseline controller (epsilon-greedy or softmax Q-learner).

    Returns logs dict compatible with `run_episode_with_ll` output so it can
    be used as a reference trajectory for teacher-forcing evaluation.
    """
    rng = np.random.RandomState(seed)

    # Initialize simple Q-values for each discrete action label
    Q = {a: 0.0 for a in ACTION_CHOICES}

    logs = {
        't': [],
        'context': [],
        'belief': [],
        'gamma': [],
        'action': [],
        'reward_label': [],
        'choice_label': [],
        'hint_label': [],
        'll': []
    }

    # Initialize neutral belief vector for compatibility
    try:
        zero_belief = np.zeros(len(STATE_CONTEXTS))
    except Exception:
        zero_belief = np.zeros(2)

    for t in range(T):
        # Choose action
        if policy_type == 'egreedy':
            if rng.rand() < epsilon:
                action_label = rng.choice(ACTION_CHOICES)
            else:
                # argmax (break ties randomly)
                maxval = max(Q.values())
                bests = [a for a, v in Q.items() if v == maxval]
                action_label = rng.choice(bests)
        elif policy_type == 'softmax':
            vals = np.array([Q[a] for a in ACTION_CHOICES], dtype=float)
            # numerical stability
            vals = vals - vals.max()
            exps = np.exp(vals / float(max(1e-8, temp)))
            probs = exps / (exps.sum() + 1e-16)
            action_label = rng.choice(ACTION_CHOICES, p=probs)
        else:
            # fallback: random
            action_label = rng.choice(ACTION_CHOICES)

        # Environment step
        obs_labels = env.step(action_label)

        # Convert reward label to numeric for simple Q update
        if obs_labels[1] == 'observe_reward':
            r = 1.0
        elif obs_labels[1] == 'observe_loss':
            r = -1.0
        else:
            r = 0.0

        # Update simple Q estimate for action
        Q[action_label] = Q[action_label] + alpha * (r - Q[action_label])

        # Log
        logs['t'].append(t)
        logs['context'].append(env.context)
        logs['belief'].append(zero_belief.copy())
        logs['gamma'].append(np.nan)
        logs['action'].append(action_label)
        logs['hint_label'].append(obs_labels[0])
        logs['reward_label'].append(obs_labels[1])
        logs['choice_label'].append(obs_labels[2])
        logs['ll'].append(0.0)

    return logs


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


def compute_sequence_ll_for_model(model_name, A, B, D, ref_logs):
    """Compute per-trial log-probabilities of the actions in `ref_logs`
    under `model_name` using teacher-forcing (condition on the same
    observed history).
    Returns a list of per-trial log-likelihoods.
    """
    # Build model
    value_fn = create_model(model_name, A, B, D)

    # Runner for evaluation (does not interact with env)
    runner = AgentRunnerWithLL(A, B, D, value_fn,
                               OBSERVATION_HINTS, OBSERVATION_REWARDS,
                               OBSERVATION_CHOICES, ACTION_CHOICES,
                               reward_mod_idx=1)

    # Start from initial observation
    initial_obs_labels = ['null', 'null', 'observe_start']
    obs_ids = runner.obs_labels_to_ids(initial_obs_labels)

    T = len(ref_logs['action'])
    ll_seq = []
    for t in range(T):
        action_label = ref_logs['action'][t]
        ll_t = runner.action_logprob(obs_ids, action_label, t)
        ll_seq.append(ll_t)

        # Advance observation to the one recorded after this action in the
        # reference run (teacher-forcing uses the observed next observation).
        if 'hint_label' in ref_logs and ref_logs['hint_label']:
            next_obs = [ref_logs['hint_label'][t], ref_logs['reward_label'][t], ref_logs['choice_label'][t]]
        else:
            next_obs = ['null', ref_logs['reward_label'][t], ref_logs['choice_label'][t]]

        obs_ids = runner.obs_labels_to_ids(next_obs)

    return ll_seq


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

    # Reference generator types to evaluate against
    # include generators derived from M1 and M2 plus simple baselines
    reference_types = ['expert', 'noisy', 'perturbed', 'M1', 'M2', 'egreedy', 'softmax']

    for ref_type in reference_types:
        print(f"\nGenerating reference rollouts (type: {ref_type})...")
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
                except Exception:
                    pass

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
                    except Exception:
                        pass
                    try:
                        runner.agent.gamma = 100.0
                    except Exception:
                        pass
                else:
                    try:
                        runner.agent.action_selection = 'stochastic'
                    except Exception:
                        pass
                    try:
                        runner.agent.gamma = 2.0
                    except Exception:
                        pass

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
                except Exception:
                    pass

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
        for model_name in models:
            print(f"Evaluating {model_name} against reference type '{ref_type}'...")
            for run_idx, ref_logs in enumerate(tqdm(reference_runs, desc=f"  {model_name}:{ref_type}")):
                ll_seq = compute_sequence_ll_for_model(model_name, A, B, D, ref_logs)
                # create a copy of the reference logs but with ll replaced by the
                # teacher-forced log-probabilities produced by this model
                new_logs = dict(ref_logs)
                new_logs['ll'] = ll_seq
                metrics = compute_metrics(new_logs)
                # include reference type info in metrics for later aggregation
                metrics['reference_type'] = ref_type
                results[model_name].append(metrics)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
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


def plot_model_comparison(results, num_trials, reversal_interval=None):
    """Generate comparison plots."""
    models = ['M1', 'M2', 'M3']
    colors = {'M1': 'blue', 'M2': 'green', 'M3': 'red'}

    # Collect reference types present across results
    ref_types = set()
    for m in models:
        for r in results.get(m, []):
            ref_types.add(r.get('reference_type', 'default'))
    ref_types = sorted(ref_types)

    # Create per-reference-type figures
    for ref_type in ref_types:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Accuracy over time (averaged within this reference type)
        ax = axes[0, 0]
        for model_name in models:
            model_results = [r for r in results.get(model_name, []) if r.get('reference_type') == ref_type]
            if not model_results:
                continue

            all_accs = np.array([r['accuracy'] for r in model_results])
            mean_acc = all_accs.mean(axis=0)
            std_acc = all_accs.std(axis=0)

            # Rolling average
            window = ROLLING_WINDOW
            if len(mean_acc) >= window:
                mean_acc_smooth = np.convolve(mean_acc, np.ones(window)/window, mode='valid')
                ax.plot(mean_acc_smooth, label=model_name, color=colors[model_name], linewidth=2)
            else:
                ax.plot(mean_acc, label=model_name, color=colors[model_name], linewidth=2)

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Accuracy (rolling avg)')
        ax.set_title(f'Performance Over Time (reference: {ref_type})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Accuracy distribution (per model)
        ax = axes[0, 1]
        acc_data = []
        labels = []
        for model_name in models:
            vals = [r['mean_accuracy'] for r in results.get(model_name, []) if r.get('reference_type') == ref_type]
            acc_data.append(vals if vals else [np.nan])
            labels.append(model_name)

        bp = ax.boxplot(acc_data, labels=labels, patch_artist=True)
        for patch, model in zip(bp['boxes'], models):
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.6)
        ax.set_ylabel('Mean Accuracy')
        ax.set_title(f'Accuracy Distribution (reference: {ref_type})')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Total rewards
        ax = axes[1, 0]
        reward_data = []
        for model_name in models:
            vals = [r['total_reward'] for r in results.get(model_name, []) if r.get('reference_type') == ref_type]
            reward_data.append(vals if vals else [np.nan])

        bp = ax.boxplot(reward_data, labels=labels, patch_artist=True)
        for patch, model in zip(bp['boxes'], models):
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.6)
        ax.set_ylabel('Total Reward')
        ax.set_title(f'Cumulative Rewards (reference: {ref_type})')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 4: Gamma dynamics (example run per model)
        ax = axes[1, 1]
        for model_name in models:
            model_results = [r for r in results.get(model_name, []) if r.get('reference_type') == ref_type]
            if not model_results:
                continue
            gamma_series = model_results[0]['logs']['gamma']
            window = 5
            if len(gamma_series) >= window:
                gamma_smooth = np.convolve(gamma_series, np.ones(window)/window, mode='valid')
            else:
                gamma_smooth = np.array(gamma_series)
            ax.plot(gamma_smooth, label=model_name, color=colors[model_name], linewidth=2)

        ax.set_xlabel('Trial')
        ax.set_ylabel('Policy Precision (γ)')
        ax.set_title(f'Precision Dynamics (reference: {ref_type})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        # Build outpath per reference type
        if reversal_interval is None:
            outpath = f'results/figures/{ref_type}/model_comparison_{ref_type}.png'
        else:
            outpath = f'results/figures/{ref_type}/model_comparison_{ref_type}_interval_{reversal_interval}.png'
        out_dir = os.path.dirname(outpath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(outpath, dpi=FIG_DPI, bbox_inches='tight')
        print(f"\nPlot saved: {outpath}")
        plt.close(fig)

    # Additionally, save an aggregated summary figure across reference types
    # Boxplot of log-likelihoods per model, grouped by reference type
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = []
    labels = []
    data = []
    offset = 0
    gap = 1
    for i, model_name in enumerate(models):
        model_vals = results.get(model_name, [])
        # For each reference type, collect ll values
        for ref_type in ref_types:
            vals = [r['log_likelihood'] for r in model_vals if r.get('reference_type') == ref_type]
            if not vals:
                vals = [np.nan]
            data.append(vals)
            positions.append(i * (len(ref_types) + gap) + offset)
            labels.append(f"{model_name}\n{ref_type}")
            offset += 1
        offset = 0

    bp = ax.boxplot(data, positions=positions, patch_artist=True)
    # color boxes by model (cycle)
    for idx, patch in enumerate(bp['boxes']):
        model_idx = idx // len(ref_types)
        model_name = models[model_idx]
        patch.set_facecolor(colors[model_name])
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Run Log-Likelihood')
    ax.set_title('Log-Likelihood by Model and Reference Type')
    plt.tight_layout()
    if reversal_interval is None:
        outpath = 'results/figures/model_comparison_by_ref.png'
    else:
        outpath = f'results/figures/model_comparison_by_ref_interval_{reversal_interval}.png'
    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(outpath, dpi=FIG_DPI, bbox_inches='tight')
    print(f"\nAggregated plot saved: {outpath}")
    plt.show()


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
