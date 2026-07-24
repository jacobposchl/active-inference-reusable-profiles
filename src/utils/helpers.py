"""
Helper utility functions.
"""
import numpy as np


def rolling_mean(x, k):
    """
    Compute rolling mean with centered window.
    
    Parameters:
    -----------
    x : array-like
        Input data
    k : int or None
        Window size. If None or <= 1, returns original array
        
    Returns:
    --------
    smoothed : np.ndarray
        Smoothed data with same length as input
    """
    if k is None or k <= 1:
        return np.asarray(x, dtype=float)
    
    x = np.asarray(x, dtype=float)
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    cumsum = np.cumsum(xpad)
    out = (cumsum[k:] - cumsum[:-k]) / float(k)
    
    # Handle even k
    if len(out) < len(x):
        out = np.r_[out, np.repeat(out[-1], len(x) - len(out))]
    
    return out[:len(x)]


def compute_entropy(probs):
    """
    Compute Shannon entropy of a probability distribution.
    
    Parameters:
    -----------
    probs : array-like
        Probability distribution
        
    Returns:
    --------
    H : float
        Entropy in nats
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return -np.sum(p * np.log(p))


def find_reversals(context_series):
    """
    Find trial indices where context changes.
    
    Parameters:
    -----------
    context_series : list or array
        Sequence of context labels
        
    Returns:
    --------
    reversals : list
        Trial indices where context switches
    """
    ctx = np.asarray(context_series)
    return list(np.where(ctx[1:] != ctx[:-1])[0] + 1)


def trial_accuracy(choice_labels, better_arm_series):
    """
    Compute trial-by-trial accuracy: did the action select the currently better arm?

    Parameters:
    -----------
    choice_labels : list
        Action labels per trial (e.g. 'act_left', 'act_right', 'act_hint').
    better_arm_series : list
        Which arm is better on each trial. Accepts either the environment's
        'left'/'right' values (as logged in 'current_better_arm') or the
        'left_better'/'right_better' forms; matched by substring so both work.
        Note: pass the better-arm series, NOT the volatility context
        ('volatile'/'stable'), which carries no arm information.

    Returns:
    --------
    accuracy : np.ndarray
        Binary accuracy per trial (1 = chose better arm, 0 = did not).
    """
    acc = []
    for a, c in zip(choice_labels, better_arm_series):
        c = str(c)
        if 'left' in c:
            acc.append(1 if 'left' in str(a) else 0)
        elif 'right' in c:
            acc.append(1 if 'right' in str(a) else 0)
        else:
            acc.append(0)
    return np.array(acc, dtype=float)


def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    """
    Compute bootstrap confidence interval.
    
    Parameters:
    -----------
    data : array-like
        Data to bootstrap
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence interval level (e.g., 95 for 95% CI)
        
    Returns:
    --------
    mean : float
        Mean of data
    ci_lower : float
        Lower confidence bound
    ci_upper : float
        Upper confidence bound
    """
    data = np.array(data)
    means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    
    mean = np.mean(data)
    alpha = (100 - ci) / 2
    ci_lower = np.percentile(means, alpha)
    ci_upper = np.percentile(means, 100 - alpha)
    
    return mean, ci_lower, ci_upper


def print_trial_details(t, env, qs, gamma_t, action_label, obs_labels, show=True):
    """
    Print detailed information about what happened in a trial.
    
    Parameters:
    -----------
    t : int
        Trial number
    env : TwoArmedBandit
        Environment instance
    qs : list of arrays
        Posterior beliefs over hidden states
    gamma_t : float
        Policy precision this trial
    action_label : str
        Action taken
    obs_labels : list of str
        Observations received
    show : bool
        Whether to actually print
    """
    if not show:
        return
    
    q_context = qs[0]
    q_choice = qs[1]
    H_context = compute_entropy(q_context)
    
    better_arm = getattr(env, 'current_better_arm', None)
    correct = "✓" if ((better_arm == 'left' and 'left' in action_label) or
                      (better_arm == 'right' and 'right' in action_label)) else "✗"
    
    print(f"t={t:3d} | True:{env.context:12s} | Action:{action_label:12s} {correct} | "
          f"Reward:{obs_labels[1]:14s} | "
          f"q(left)={q_context[0]:.3f} | H={H_context:.3f} | γ={gamma_t:.3f}")
