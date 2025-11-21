"""
Visualization functions for experiment results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .helpers import rolling_mean, compute_entropy


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
