"""
Mechanistic analysis around reversals.

Runs short experiments (90 trials, reversals at 30 and 60) and visualizes
how M1/M2/M3 behave around reversal points.

Produces figures:
- accuracy_around_reversal.png: accuracy aligned to reversal (pooled across runs and reversals) for all models
- gamma_around_reversal.png: policy precision (gamma) aligned to reversals for all models

Usage:
    python src\experiments\mechanistic_analysis.py

"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path (so imports from src.* work)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.experiment_config import *
from src.models import build_A, build_B, build_D
from src.experiments.model_comparison import run_single_agent
from src.utils import trial_accuracy, bootstrap_ci
from src.utils.helpers import find_reversals
from config.experiment_config import M3_DEFAULTS


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def align_windows(data_list, reversal_indices, pre=10, post=20, n_trials=None):
    """Align a list of 1D series (each length n_trials) to reversal indices.

    Returns array shape (n_samples, window_len) where n_samples = len(data_list) * len(reversal_indices)
    """
    windows = []
    window_len = pre + post
    for series in data_list:
        series = np.asarray(series)
        for r in reversal_indices:
            start = r - pre
            end = r + post
            if start < 0 or end > len(series):
                # skip reversals too close to edges
                continue
            windows.append(series[start:end])
    if len(windows) == 0:
        return np.zeros((0, window_len))
    return np.vstack(windows)


def compute_mean_and_ci(pooled, n_bootstrap=2000):
    """Return mean, lower CI, upper CI arrays for pooled shape (n_samples, time).
    Use bootstrap_ci per timepoint.
    """
    if pooled.size == 0:
        return np.array([]), np.array([]), np.array([])
    means = pooled.mean(axis=0)
    lowers = np.zeros_like(means)
    uppers = np.zeros_like(means)
    for t in range(pooled.shape[1]):
        vals = pooled[:, t]
        # Use bootstrap helper for mean CI
        m, lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap, ci=95)
        means[t] = m
        lowers[t] = lo
        uppers[t] = hi
    return means, lowers, uppers


def plot_accuracy_around_reversals(results_by_model, pre=10, post=20, outpath=None):
    x = np.arange(-pre, post)
    plt.figure(figsize=(8, 5))
    colors = {'M1': 'blue', 'M2': 'green', 'M3': 'red'}
    for model_name, logs_list in results_by_model.items():
        # build accuracy series per run
        acc_series = [trial_accuracy(logs['action'], logs['context']) for logs in logs_list]
        # reversal indices: detect from logs[0] (should be same across runs)
        revs = find_reversals(logs_list[0]['context'])
        pooled = align_windows(acc_series, revs, pre=pre, post=post)
        mean, lo, hi = compute_mean_and_ci(pooled)
        if mean.size == 0:
            continue
        plt.plot(x, mean, label=f"{model_name}", color=colors[model_name])
        plt.fill_between(x, lo, hi, color=colors[model_name], alpha=0.2)
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Trials from reversal')
    plt.ylabel('Accuracy')
    plt.title('Accuracy around reversals (pooled across runs and reversals)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    ensure_dir(os.path.dirname(outpath))
    plt.savefig(outpath, dpi=FIG_DPI, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


def plot_m3_profile_weights_by_direction(results_m3, pre=10, post=20, outpath=None):
    """
    Plot M3 profile weights aligned to reversals separately for reversals that switch
    to `left_better` vs `right_better` (i.e., direction-specific alignment).

    This avoids averaging windows from both reversal directions which can mask
    direction-specific effects in profile weights.
    """
    Z = np.array(M3_DEFAULTS['Z'])
    x = np.arange(-pre, post)
    # For each run, compute profile weights time series: w_t = belief_t @ Z
    weight_series = []
    contexts = []
    for logs in results_m3:
        beliefs = np.vstack(logs['belief'])  # shape (T, n_contexts)
        w = beliefs @ Z  # shape (T, n_profiles)
        weight_series.append(w)
        contexts.append(np.asarray(logs['context']))

    # gather reversals per run with direction (post-reversal context)
    n_profiles = Z.shape[1]
    pooled_to_left = [np.zeros((0, pre+post)) for _ in range(n_profiles)]
    pooled_to_right = [np.zeros((0, pre+post)) for _ in range(n_profiles)]

    for w, ctx in zip(weight_series, contexts):
        revs = find_reversals(ctx)
        for r in revs:
            start = r - pre
            end = r + post
            if start < 0 or end > w.shape[0]:
                continue
            post_ctx = ctx[r]
            for p in range(n_profiles):
                window = w[start:end, p]
                if post_ctx == 'left_better':
                    if pooled_to_left[p].size == 0:
                        pooled_to_left[p] = window[np.newaxis, :]
                    else:
                        pooled_to_left[p] = np.vstack((pooled_to_left[p], window))
                elif post_ctx == 'right_better':
                    if pooled_to_right[p].size == 0:
                        pooled_to_right[p] = window[np.newaxis, :]
                    else:
                        pooled_to_right[p] = np.vstack((pooled_to_right[p], window))

    # plotting helper
    def _plot_pooled(pooled_list, title_suffix, fname_suffix):
        plt.figure(figsize=(8, 5))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for p in range(n_profiles):
            pooled = pooled_list[p]
            mean, lo, hi = compute_mean_and_ci(pooled)
            if mean.size == 0:
                continue
            plt.plot(x, mean, label=f'Profile {p}', color=colors[p])
            plt.fill_between(x, lo, hi, color=colors[p], alpha=0.2)
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('Trials from reversal')
        plt.ylabel('Profile weight')
        plt.title(f'M3 profile weights around reversals ({title_suffix})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if outpath is not None:
            outdir = os.path.dirname(outpath)
            ensure_dir(outdir)
            base = os.path.splitext(os.path.basename(outpath))[0]
            save_name = os.path.join(outdir, f"{base}_{fname_suffix}.png")
            plt.savefig(save_name, dpi=FIG_DPI, bbox_inches='tight')
            print(f"Saved: {save_name}")
        plt.close()

    # plot both directions
    _plot_pooled(pooled_to_left, 'switch -> left_better', 'm3_profiles_to_left')
    _plot_pooled(pooled_to_right, 'switch -> right_better', 'm3_profiles_to_right')


def plot_gamma_around_reversals(results_by_model, pre=10, post=20, outpath=None):
    x = np.arange(-pre, post)
    plt.figure(figsize=(8, 5))
    colors = {'M1': 'blue', 'M2': 'green', 'M3': 'red'}
    for model_name, logs_list in results_by_model.items():
        gamma_series = [logs['gamma'] for logs in logs_list]
        revs = find_reversals(logs_list[0]['context'])
        pooled = align_windows(gamma_series, revs, pre=pre, post=post)
        mean, lo, hi = compute_mean_and_ci(pooled)
        if mean.size == 0:
            continue
        plt.plot(x, mean, label=f"{model_name}", color=colors[model_name])
        plt.fill_between(x, lo, hi, color=colors[model_name], alpha=0.2)
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Trials from reversal')
    plt.ylabel('Gamma (policy precision)')
    plt.title('Policy precision (γ) around reversals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    ensure_dir(os.path.dirname(outpath))
    plt.savefig(outpath, dpi=FIG_DPI, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


def plot_m3_profile_weights(results_m3, pre=10, post=20, outpath=None):
    # (Function removed) Averaged/proportional pooling across all reversal directions
    # was found to be unhelpful because it mixes opposite reversal directions
    # and can mask direction-specific recruitment of profiles. Use
    # `plot_m3_profile_weights_by_direction` instead.
    raise RuntimeError("plot_m3_profile_weights has been removed. Use plot_m3_profile_weights_by_direction instead.")


def run_mechanistic_experiment(models=('M1', 'M2', 'M3'), num_runs=20, num_trials=90, seed=42,
                               reversal_schedule=[30, 60], pre=10, post=20):
    """Run short experiments and produce mechanistic figures.
    """
    print('Mechanistic analysis: short runs around reversals')
    print(f'models={models}, runs={num_runs}, trials={num_trials}, reversals={reversal_schedule}')
    
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)

    results_by_model = {m: [] for m in models}

    for model_name in models:
        print(f'Running {model_name}...')
        for run in tqdm(range(num_runs), desc=f'  {model_name}'):
            run_seed = seed + run if seed is not None else None
            logs = run_single_agent(model_name, A, B, D, num_trials, seed=run_seed, reversal_schedule=reversal_schedule)
            results_by_model[model_name].append(logs)

    # Plotting
    outdir = 'results/figures'
    ensure_dir(outdir)
    plot_accuracy_around_reversals(results_by_model, pre=pre, post=post, outpath=os.path.join(outdir, f'mechanistic_accuracy_rev_{reversal_schedule[0]}_{reversal_schedule[1]}.png'))
    plot_gamma_around_reversals(results_by_model, pre=pre, post=post, outpath=os.path.join(outdir, f'mechanistic_gamma_rev_{reversal_schedule[0]}_{reversal_schedule[1]}.png'))
    plot_m3_profile_weights(results_by_model['M3'], pre=pre, post=post, outpath=os.path.join(outdir, f'mechanistic_m3_profiles_rev_{reversal_schedule[0]}_{reversal_schedule[1]}.png'))
    # Also produce direction-specific profile-weight plots (switch -> left vs switch -> right)
    plot_m3_profile_weights_by_direction(results_by_model['M3'], pre=pre, post=post, outpath=os.path.join(outdir, f'mechanistic_m3_profiles_rev_{reversal_schedule[0]}_{reversal_schedule[1]}.png'))

    print('Mechanistic analysis complete. Figures saved to results/figures')
    return results_by_model


if __name__ == '__main__':
    run_mechanistic_experiment(num_runs=20, num_trials=90, seed=42, reversal_schedule=[30, 60], pre=10, post=20)
