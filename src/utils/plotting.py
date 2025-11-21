"""
Visualization functions for experiment results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .helpers import rolling_mean, compute_entropy, find_reversals, trial_accuracy, bootstrap_ci
from config.experiment_config import M3_DEFAULTS, FIG_DPI, ROLLING_WINDOW


def plot_likelihood(matrix, title_str="Likelihood distribution (A)"):
    """
    Plot a 2-D likelihood matrix as a heatmap.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Column-normalized likelihood matrix
    title_str : str
        Plot title
    """
    if not np.isclose(matrix.sum(axis=0), 1.0).all():
        raise ValueError("Distribution not column-normalized!")
    
    fig = plt.figure(figsize=(6, 6))
    ax = sns.heatmap(matrix, cmap='gray', cbar=False, vmin=0.0, vmax=1.0)
    plt.title(title_str)
    plt.show()


def plot_beliefs(belief_dist, title_str=""):
    """
    Plot a categorical distribution.
    
    Parameters:
    -----------
    belief_dist : np.ndarray
        Probability distribution
    title_str : str
        Plot title
    """
    if not np.isclose(belief_dist.sum(), 1.0):
        raise ValueError("Distribution not normalized!")
    
    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    plt.xticks(range(belief_dist.shape[0]))
    plt.title(title_str)
    plt.show()


def plot_gamma_over_time(gamma_series, reversals=None, belief_left_series=None,
                         roll_k=None, title="Policy precision (γ) over trials",
                         ylabel="γ (policy precision)", show=True):
    """
    Plot policy precision over time with reversals and smoothing.
    
    Parameters:
    -----------
    gamma_series : list or array
        Gamma values per trial
    reversals : list or None
        Trial indices of reversals
    belief_left_series : list or None
        Belief in left_better context (unused, kept for compatibility)
    roll_k : int or None
        Rolling mean window size
    title : str
        Plot title
    ylabel : str
        Y-axis label
    show : bool
        Whether to display plot
        
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    T = len(gamma_series)
    t = np.arange(T)
    
    gamma = np.asarray(gamma_series, dtype=float)
    
    # Forward fill missing values
    if np.isnan(gamma).any():
        for i in range(1, T):
            if np.isnan(gamma[i]) and not np.isnan(gamma[i-1]):
                gamma[i] = gamma[i-1]
        
        first_valid = np.flatnonzero(~np.isnan(gamma))
        if first_valid.size:
            gamma[:first_valid[0]] = gamma[first_valid[0]]
        else:
            gamma[:] = 0.0
    
    gamma_smooth = rolling_mean(gamma, roll_k)
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, gamma, lw=1, alpha=0.35, label="γ (raw)")
    
    if roll_k and roll_k > 1:
        ax.plot(t, gamma_smooth, lw=2, label=f"γ (rolling mean, k={roll_k})")
    
    ax.set_xlabel("trial")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if reversals:
        for r in reversals:
            ax.axvline(r, linestyle="--", linewidth=1, alpha=0.7)
        ax.plot([], [], linestyle="--", color='gray', label="reversal")
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig, ax


def plot_entropy_over_time(belief_series, reversals=None, roll_k=None,
                           title="Belief entropy over trials",
                           ylabel="Entropy H(q(s))", show=True):
    """
    Plot belief entropy over time.
    
    Parameters:
    -----------
    belief_series : list of arrays
        Belief distributions per trial
    reversals : list or None
        Trial indices of reversals
    roll_k : int or None
        Rolling mean window size
    title : str
        Plot title
    ylabel : str
        Y-axis label
    show : bool
        Whether to display plot
        
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    T = len(belief_series)
    
    entropies = np.array([compute_entropy(q) for q in belief_series])
    
    if roll_k and roll_k > 1:
        kernel = np.ones(roll_k) / roll_k
        ent_smooth = np.convolve(entropies, kernel, mode="same")
    else:
        ent_smooth = entropies
    
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(entropies, color="gray", lw=1, alpha=0.5, label="raw")
    
    if roll_k and roll_k > 1:
        ax.plot(ent_smooth, color="red", lw=2, label=f"rolling mean (k={roll_k})")
    
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if reversals:
        for r in reversals:
            ax.axvline(r, linestyle="--", color="black", alpha=0.6)
        ax.plot([], [], "--", color="black", label="reversal")
    
    ax.legend(loc="upper right")
    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig, ax


def plot_switch_dynamics(choices, true_contexts, window=10,
                         title="Switch Dynamics", show=True):
    """
    Plot mean accuracy around reversal points.
    
    Parameters:
    -----------
    choices : list of str
        Action labels per trial
    true_contexts : list of str
        True context per trial
    window : int
        Number of trials before/after reversals
    title : str
        Plot title
    show : bool
        Whether to display plot
        
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    choices = np.array(choices)
    contexts = np.array(true_contexts)
    T = len(choices)
    
    # Find reversals
    reversals = np.where(contexts[1:] != contexts[:-1])[0] + 1
    
    if len(reversals) == 0:
        print("No reversals detected.")
        return None, None
    
    # Compute correctness
    correct = np.zeros(T, dtype=float)
    for t in range(T):
        c = contexts[t]
        a = choices[t]
        
        if c == 'left_better' and 'left' in a:
            correct[t] = 1.0
        elif c == 'right_better' and 'right' in a:
            correct[t] = 1.0
    
    # Extract accuracy around reversals
    rel_acc = []
    for r in reversals:
        start = max(0, r - window)
        end = min(T, r + window)
        
        seg = correct[start:end]
        
        pad_left = window - (r - start)
        pad_right = window - (end - r)
        seg_padded = np.pad(seg, (pad_left, pad_right), constant_values=np.nan)
        
        rel_acc.append(seg_padded)
    
    rel_acc = np.vstack(rel_acc)
    mean_acc = np.nanmean(rel_acc, axis=0)
    
    x = np.arange(-window, window)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, mean_acc, lw=2, color='dodgerblue', marker='o', 
            markersize=4, label='Mean accuracy')
    ax.axvline(0, linestyle='--', color='k', alpha=0.7, label='Reversal')
    ax.set_xlabel("Trials relative to reversal")
    ax.set_ylabel("Mean accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig, ax


def plot_model_comparison(results, num_trials, reversal_interval=None, out_dir_base='results/figures'):
    """Generate comparison plots (moved from experiments.model_comparison).

    Parameters:
    - results: dict mapping model -> list of per-run metrics (each must include 'logs', 'mean_accuracy', etc.)
    - num_trials: int
    - reversal_interval: optional
    - out_dir_base: base directory for saving figures
    """
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
            window = 7
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
            outpath = os.path.join(out_dir_base, f'{ref_type}', f'model_comparison_{ref_type}.png')
        else:
            outpath = os.path.join(out_dir_base, f'{ref_type}', f'model_comparison_{ref_type}_interval_{reversal_interval}.png')
        out_dir = os.path.dirname(outpath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(outpath, dpi=100, bbox_inches='tight')
        print(f"\nPlot saved: {outpath}")
        plt.close(fig)

    # Additionally, save an aggregated summary figure across reference types
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
        outpath = os.path.join(out_dir_base, 'model_comparison_by_ref.png')
    else:
        outpath = os.path.join(out_dir_base, f'model_comparison_by_ref_interval_{reversal_interval}.png')
    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(outpath, dpi=100, bbox_inches='tight')
    print(f"\nAggregated plot saved: {outpath}")
    plt.close(fig)


# Mechanistic analysis plotting helpers (moved from src.experiments.mechanistic_analysis)
def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def align_windows(data_list, reversal_indices, pre=10, post=20, n_trials=None):
    windows = []
    window_len = pre + post
    for series in data_list:
        series = np.asarray(series)
        for r in reversal_indices:
            start = r - pre
            end = r + post
            if start < 0 or end > len(series):
                continue
            windows.append(series[start:end])
    if len(windows) == 0:
        return np.zeros((0, window_len))
    return np.vstack(windows)


def compute_mean_and_ci(pooled, bootstrap_func, n_bootstrap=2000):
    if pooled.size == 0:
        return np.array([]), np.array([]), np.array([])
    means = pooled.mean(axis=0)
    lowers = np.zeros_like(means)
    uppers = np.zeros_like(means)
    for t in range(pooled.shape[1]):
        vals = pooled[:, t]
        m, lo, hi = bootstrap_func(vals, n_bootstrap=n_bootstrap, ci=95)
        means[t] = m
        lowers[t] = lo
        uppers[t] = hi
    return means, lowers, uppers


def plot_accuracy_around_reversals(results_by_model, pre=10, post=20, outpath=None, bootstrap_func=None):
    x = np.arange(-pre, post)
    plt.figure(figsize=(8, 5))
    colors = {'M1': 'blue', 'M2': 'green', 'M3': 'red'}
    for model_name, logs_list in results_by_model.items():
        acc_series = [trial_accuracy(logs['action'], logs['context']) for logs in logs_list]
        revs = find_reversals(logs_list[0]['context'])
        pooled = align_windows(acc_series, revs, pre=pre, post=post)
        mean, lo, hi = compute_mean_and_ci(pooled, bootstrap_func)
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
    if outpath is not None:
        ensure_dir(os.path.dirname(outpath))
        plt.savefig(outpath, dpi=100, bbox_inches='tight')
        print(f"Saved: {outpath}")
    plt.close()


def plot_m3_profile_weights_by_direction(results_m3, pre=10, post=20, outpath=None, bootstrap_func=None):
    Z = np.array(M3_DEFAULTS['Z'])
    x = np.arange(-pre, post)
    weight_series = []
    contexts = []
    for logs in results_m3:
        beliefs = np.vstack(logs['belief'])
        w = beliefs @ Z
        weight_series.append(w)
        contexts.append(np.asarray(logs['context']))

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

    def _plot_pooled(pooled_list, title_suffix, fname_suffix):
        plt.figure(figsize=(8, 5))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for p in range(n_profiles):
            pooled = pooled_list[p]
            mean, lo, hi = compute_mean_and_ci(pooled, bootstrap_func)
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
            plt.savefig(save_name, dpi=100, bbox_inches='tight')
            print(f"Saved: {save_name}")
        plt.close()

    _plot_pooled(pooled_to_left, 'switch -> left_better', 'm3_profiles_to_left')
    _plot_pooled(pooled_to_right, 'switch -> right_better', 'm3_profiles_to_right')


def plot_gamma_around_reversals(results_by_model, pre=10, post=20, outpath=None, bootstrap_func=None):
    x = np.arange(-pre, post)
    plt.figure(figsize=(8, 5))
    colors = {'M1': 'blue', 'M2': 'green', 'M3': 'red'}
    for model_name, logs_list in results_by_model.items():
        gamma_series = [logs['gamma'] for logs in logs_list]
        revs = find_reversals(logs_list[0]['context'])
        pooled = align_windows(gamma_series, revs, pre=pre, post=post)
        mean, lo, hi = compute_mean_and_ci(pooled, bootstrap_func)
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
    if outpath is not None:
        ensure_dir(os.path.dirname(outpath))
        plt.savefig(outpath, dpi=100, bbox_inches='tight')
        print(f"Saved: {outpath}")
    plt.close()
