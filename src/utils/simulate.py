import numpy as np
from src.environment import TwoArmedBandit
from src.models import build_A, build_B, build_D
from config.experiment_config import *


def simulate_baseline_run(env, policy_type, T=400, seed=None, epsilon=0.1, temp=1.0, alpha=0.1):
    """
    Simulate a simple baseline controller (epsilon-greedy or softmax Q-learner).

    Returns logs dict compatible with `run_episode_with_ll` output so it can
    be used as a reference trajectory for teacher-forcing evaluation.
    """
    rng = np.random.RandomState(seed)

    # Initialize simple Q-values for each discrete action label
    Q = {a: 0.0 for a in ACTION_CHOICES}

    logs = {
        't': [],
        'context': [],
        'current_better_arm': [],  # Track which arm is actually better
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
                maxval = max(Q.values())
                bests = [a for a, v in Q.items() if v == maxval]
                action_label = rng.choice(bests)
        elif policy_type == 'softmax':
            vals = np.array([Q[a] for a in ACTION_CHOICES], dtype=float)
            vals = vals - vals.max()
            exps = np.exp(vals / float(max(1e-8, temp)))
            probs = exps / (exps.sum() + 1e-16)
            action_label = rng.choice(ACTION_CHOICES, p=probs)
        else:
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
        logs['current_better_arm'].append(getattr(env, 'current_better_arm', None))
        logs['belief'].append(zero_belief.copy())
        logs['gamma'].append(np.nan)
        logs['action'].append(action_label)
        logs['hint_label'].append(obs_labels[0])
        logs['reward_label'].append(obs_labels[1])
        logs['choice_label'].append(obs_labels[2])
        logs['ll'].append(0.0)

    return logs
