import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.experiment_config import *
from src.environment import TwoArmedBandit
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL, run_episode_with_ll
from src.utils import trial_accuracy
from pymdp.agent import Agent
from pymdp import utils

"""
hard/soft Z mixing barplot
--------------------------

This helper script runs a small ablation exploring how different Z mixing
matrices (hard vs soft mixing of two K=2 profiles) affect behavior for
the top-left cell of the heatmap (prob_reward=0.65, prob_hint=0.65).

It computes three summary statistics per-profile-mix:
- total reward (sum of +1/-1 for reward/loss labels across trials)
- overall accuracy (per-trial 0/1 accuracy including non-choice trials)
- choice-only accuracy (accuracy computed only on trials when the agent
  made a left/right choice)

The results are plotted as grouped bars with error bars (mean ± std)
across `num_runs` independent simulation runs.
"""

# === SETTINGS (from heatmap C, top-left cell) ===
prob_reward = 0.65
prob_hint = 0.65
num_trials = 800
num_runs = 10
reversal_schedule = [i for i in range(40, num_trials, 40)]

# === Profiles from set C ===
# k1_profile: single-profile K=1 agent
k1_profile = {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 3.0, 0.0, 0.0]}
# k2_profiles: two symmetric K=2 profiles (left/right bias)
k2_profiles = [
    {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 3.0, 6.0, -6.0]},
    {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 3.0, -6.0, 6.0]}
]

# Different Z mixing matrices to compare. 'Hard' corresponds to
# deterministic assignment of contexts to profiles; 'Medium' and 'Soft'
# progressively soften the assignment (mixing probabilities).
Z_matrices = {
    'K=1': np.ones((2, 1)),
    'Hard': np.array([[1.0, 0.0], [0.0, 1.0]]),
    'Medium': np.array([[0.8, 0.2], [0.2, 0.8]]),
    'Soft': np.array([[0.6, 0.4], [0.4, 0.6]])
}

results = {}


def evaluate(profiles, Z, label):
    """
    Evaluate a given profile set and Z mixing matrix by running
    `num_runs` independent simulated episodes and aggregating
    performance statistics.

    Parameters
    ----------
    profiles : list of dict
        Profile parameter dicts (passed to the value-function builder).
    Z : ndarray
        Mixing matrix mapping latent contexts to profile probabilities.
    label : str
        Short label used for bookkeeping (not used in computation).

    Returns
    -------
    tuple
        (mean_total_reward, std_total_reward,
         mean_accuracy, std_accuracy,
         mean_choice_only_accuracy, std_choice_only_accuracy)

    Notes
    -----
    - This function mirrors the evaluation approach used elsewhere in the
      project: build A/B/D, construct policies via a temporary Agent,
      build a value function, then run AgentRunnerWithLL to collect logs.
    - Random seeds are set per-run for reproducibility.
    """
    # Build model components for the environment and agent
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               prob_hint, prob_reward)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)

    # Create a temporary agent to query the policy set used for planning.
    C_temp = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
    temp_agent = Agent(A=A, B=B, C=C_temp, D=D,
                     policy_len=2, inference_horizon=1,
                     control_fac_idx=[1], use_utility=True,
                     use_states_info_gain=True,
                     action_selection="stochastic", gamma=16)
    policies = temp_agent.policies

    # Build the value function callable used by the runner
    num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
    value_fn = make_value_fn('M3',
                            profiles=profiles,
                            Z=Z,
                            policies=policies,
                            num_actions_per_factor=num_actions_per_factor)

    total_rewards = []
    accuracies = []
    choice_only_accuracies = []

    # Run multiple independent episodes and aggregate
    for run in range(num_runs):
        np.random.seed(42 + run)
        env = TwoArmedBandit(
            probability_hint=prob_hint,
            probability_reward=prob_reward,
            reversal_schedule=reversal_schedule
        )

        runner = AgentRunnerWithLL(A, B, D, value_fn,
                            OBSERVATION_HINTS, OBSERVATION_REWARDS,
                            OBSERVATION_CHOICES, ACTION_CHOICES,
                            reward_mod_idx=1)
        logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)

        # Compute a simple total reward summary: +1 for observe_reward,
        # -1 for observe_loss. This mirrors the reward labels emitted by
        # the environment/runner in this project.
        reward = 0
        for lbl in logs['reward_label']:
            if lbl == 'observe_reward':
                reward += 1
            elif lbl == 'observe_loss':
                reward -= 1
        total_rewards.append(reward)

        # Accuracy: use helper that computes per-trial 0/1 accuracy
        acc_array = trial_accuracy(logs['action'], logs['context'])
        accuracies.append(acc_array.mean())

        # Choice-only accuracy: average accuracy over trials where the
        # agent made a left/right action (ignore non-choice trials)
        is_choice = np.array([a in ('act_left', 'act_right') for a in logs['action']])
        if is_choice.any():
            choice_only_accuracies.append(acc_array[is_choice].mean())
        else:
            # If the agent never made a choice in this run (unlikely),
            # record 0.0 to keep dimensions consistent.
            choice_only_accuracies.append(0.0)

    # Return mean and standard deviation for plotting error bars
    return (np.mean(total_rewards), np.std(total_rewards),
        np.mean(accuracies), np.std(accuracies),
        np.mean(choice_only_accuracies), np.std(choice_only_accuracies))


def main(use_multiprocessing=True, n_workers=None):
    """
    Run evaluations for a small set of Z mixing matrices and plot results.

    On Windows, multiprocessing must be triggered under the
    `if __name__ == '__main__'` guard; this function supports an
    argument `use_multiprocessing` to toggle parallel execution.

    Parameters
    ----------
    use_multiprocessing : bool, optional
        Whether to parallelize evaluation across mixing labels.
    n_workers : int or None, optional
        Number of worker processes to use (defaults to cpu_count()).
    """
    global results

    # Tasks: label, profile list, and Z matrix
    tasks = [
        ('K=1', [k1_profile], Z_matrices['K=1']),
        ('Hard', k2_profiles, Z_matrices['Hard']),
        ('Medium', k2_profiles, Z_matrices['Medium']),
        ('Soft', k2_profiles, Z_matrices['Soft'])
    ]

    if use_multiprocessing:
        import multiprocessing as mp
        # Ensure safe start method on Windows. If the start method has
        # already been set in this interpreter, set_start_method will
        # raise a RuntimeError which we ignore.
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        n_workers = n_workers or mp.cpu_count()
        # Prepare args in the order (profiles, Z, label) expected by evaluate
        args = [(profiles, Z, label) for (label, profiles, Z) in tasks]
        with mp.Pool(processes=n_workers) as pool:
            results_list = pool.starmap(evaluate, args)
        # Map results back to a dict keyed by label
        results = {label: res for (label, _, _), res in zip(tasks, results_list)}
    else:
        results = {}
        for label, profiles, Z in tasks:
            results[label] = evaluate(profiles, Z, label)

    # Prepare data for plotting: grouped bars with error bars
    labels = list(results.keys())
    means = [results[k][0] for k in labels]
    stds = [results[k][1] for k in labels]
    acc_means = [results[k][2] for k in labels]
    acc_stds = [results[k][3] for k in labels]
    choice_acc_means = [results[k][4] for k in labels]
    choice_acc_stds = [results[k][5] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    rects1 = ax1.bar(x - width/2, means, width, yerr=stds, label='Total Reward', capsize=5)
    rects2 = ax1.bar(x + width/2, acc_means, width, yerr=acc_stds, label='Accuracy (all trials)', capsize=5)
    # Optionally plot choice-only accuracy as a smaller overlay
    rects3 = ax1.bar(x + width/2, choice_acc_means, width/4, yerr=choice_acc_stds, label='Accuracy (choice-only)', capsize=5,
                     align='center')

    ax1.set_ylabel('Performance')
    ax1.set_title('Ablation: Z-matrix Mixing (Profile Set C, prob_reward=0.65, prob_hint=0.65)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # On Windows it's important that multiprocessing code is protected by this guard.
    # Use multiprocessing by default; override by calling main(False).
    main(use_multiprocessing=True)
