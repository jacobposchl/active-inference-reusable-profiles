"""
k-comparison heatmaps
----------------------

This script runs a grid experiment that compares two model complexities
(K=1 vs K=2) across a small grid of environment parameters: probability
of reward (prob_reward) and probability of receiving a hint (prob_hint).

For each combination of (prob_reward, prob_hint) and for several
pre-specified agent profile sets, the script:

- builds generative model components (A, B, D) for the environment
- constructs value functions and agent runners for the K=1 and K=2
    model variants
- runs multiple simulation runs and collects log-likelihoods and
    choice accuracies
- computes BIC for each model and produces heatmaps of the difference
    (K=2 minus K=1) for log-likelihood, accuracy, and BIC

Outputs
-------
- PNG figures saved to results/figures/k2_vs_k1_heatmaps_{profile}.png
- Numpy compressed data saved to results/data/k2_vs_k1_grid_results_{profile}.npz

Notes / assumptions
-------------------
- This script is intended for offline simulation and analysis. It uses
    ProcessPoolExecutor to parallelize independent grid cells.
- The code expects the project layout where the top-level package
    modules (config, src) are discoverable by adding the repo root to
    sys.path (done below). If you move files, update the path handling.
"""

import sys
import os

# Ensure the current file's directory is importable. This helps when
# running the script directly (so relative imports for local modules
# resolve correctly). We also add the repo root below for project
# imports.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

# Add project root to path (two levels up from experiments)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.experiment_config import *
from src.environment import TwoArmedBandit

# model-building helpers and runner used by the experiments
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL, run_episode_with_ll
from pymdp.agent import Agent
from pymdp import utils

# Experiment grid
prob_rewards = [0.65, 0.75, 0.85, 0.95]
prob_hints = [0.65, 0.75, 0.85, 0.95]

num_trials = 800
num_runs = 10
reversal_schedule = [i for i in range(40, num_trials, 40)]


# Profile sets
#
# Each profile_set is a small dictionary describing a family of agent
# parameterizations. For each profile set we specify:
# - name: short identifier used in filenames and plot titles
# - k1_profiles: list of parameter dicts for the K=1 model (usually 1 profile)
# - k1_Z: latent-to-profile assignment matrix for K=1
# - k2_profiles: list of parameter dicts for the K=2 model (two profiles shown)
# - k2_Z: latent-to-profile assignment matrix for K=2
#
# The parameter dicts typically contain 'gamma', 'phi_logits', and
# 'xi_logits' which are used by the value function builder. These
# choices reflect the particular agent types we want to compare.
profile_sets = [
    # Set A: Baseline
    {
        'name': 'A',
        'k1_profiles': [
            {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, 0.0, 0.0]}
        ],
        'k1_Z': np.ones((2, 1)),
        'k2_profiles': [
            {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 4.0, 6.0, -6.0]},
            {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 4.0, -6.0, 6.0]}
        ],
        'k2_Z': np.array([[1.0, 0.0], [0.0, 1.0]])
    },
    # Set B: Moderate hint seeking
    {
        'name': 'B',
        'k1_profiles': [
            {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 1.0, 0.0, 0.0]}
        ],
        'k1_Z': np.ones((2, 1)),
        'k2_profiles': [
            {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 1.0, 4.0, -4.0]},
            {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 1.0, -4.0, 4.0]}
        ],
        'k2_Z': np.array([[1.0, 0.0], [0.0, 1.0]])
    },
    # Set C: Strong left/right bias, no hint
    {
        'name': 'C',
        'k1_profiles': [
            {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 3.0, 0.0, 0.0]}
        ],
        'k1_Z': np.ones((2, 1)),
        'k2_profiles': [
            {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 3.0, 6.0, -6.0]},
            {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 3.0, -6.0, 6.0]}
        ],
        'k2_Z': np.array([[1.0, 0.0], [0.0, 1.0]])
    }
]
# Helper: Evaluate config
def evaluate_config(profiles, Z, prob_hint, prob_reward, name):
    """
    Build model components for a single grid cell and evaluate an agent
    configuration by simulation.

    Parameters
    ----------
    profiles : list of dict
        Profile parameter dictionaries passed to the value-function builder.
    Z : ndarray
        Profile assignment matrix (latent -> profile) used by the value
        function builder.
    prob_hint : float
        Probability that the environment emits a hint on a trial.
    prob_reward : float
        Probability that the better arm yields a reward on a trial.
    name : str
        Short label for the model variant (used to set model complexity
        when computing BIC; e.g., 'K=1' or 'K=2').

    Returns
    -------
    mean_ll : float
        Mean per-trial log-likelihood across runs/trials (average of all
        per-step log-likelihoods collected from the runner).
    mean_acc : float
        Mean choice accuracy (fraction of trials where the action matched
        the context's better arm).
    bic : float
        Bayesian Information Criterion for the fitted model: -2 * total_ll
        + k * log(n), where k is the number of free parameters that we
        approximate differently for K=1 vs K=2 models.

    Notes
    -----
    - This function merges results from `num_runs` independent runs and
      treats the full collection of per-step log-likelihoods as the data
      used to compute BIC. The choice of k (3 vs 7) is an approximation
      reflecting relative model complexity used in earlier analyses.
    - Random seeds are fixed per-run for reproducibility.
    """
    # Build observation (A), transition (B), and prior (D) matrices for
    # the current (prob_hint, prob_reward) cell.
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               prob_hint, prob_reward)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)

    # Create a temporary agent to obtain the set of policies for planning.
    # We only need the policies structure (list of action sequences), so
    # we use a placeholder C (preferences) initialized to zeros.
    C_temp = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
    temp_agent = Agent(A=A, B=B, C=C_temp, D=D,
                     policy_len=2, inference_horizon=1,
                     control_fac_idx=[1], use_utility=True,
                     use_states_info_gain=True,
                     action_selection="stochastic", gamma=16)
    policies = temp_agent.policies

    # Build the value function used by the agent runner. make_value_fn is
    # expected to return a callable object that the runner uses to evaluate
    # policies given beliefs and observations.
    num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
    value_fn = make_value_fn('M3',
                            profiles=profiles,
                            Z=Z,
                            policies=policies,
                            num_actions_per_factor=num_actions_per_factor)

    all_ll = []   # collects per-time-step log-likelihood values
    all_acc = []  # collects per-time-step accuracy (0/1)

    # Run several independent simulation runs and aggregate results.
    for run in range(num_runs):
        np.random.seed(42 + run)  # reproducible runs with different seeds
        env = TwoArmedBandit(
            probability_hint=prob_hint,
            probability_reward=prob_reward,
            reversal_schedule=reversal_schedule
        )

        # Agent runner encapsulates how the agent is rolled out and records
        # quantities like per-step log-likelihoods.
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                            OBSERVATION_HINTS, OBSERVATION_REWARDS,
                            OBSERVATION_CHOICES, ACTION_CHOICES,
                            reward_mod_idx=1)
        logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)

        # Collect log-likelihoods emitted by the runner. These are often
        # per-time-step values used to compare model fits.
        all_ll.extend(logs['ll'])

        # Compute per-step accuracy by comparing the recorded context
        # ('left_better' or 'right_better') to the chosen action label.
        for ctx, choice in zip(logs['context'], logs['action']):
            if ctx == 'left_better' and choice == 'act_left':
                all_acc.append(1)
            elif ctx == 'right_better' and choice == 'act_right':
                all_acc.append(1)
            else:
                all_acc.append(0)

    # Aggregate results
    total_ll = np.sum(all_ll)
    mean_ll = np.mean(all_ll)
    mean_acc = np.mean(all_acc)

    # BIC calculation: n is total number of datapoints (trials x runs), k
    # is an approximate parameter count depending on model label.
    n = num_trials * num_runs
    k = 3 if name == 'K=1' else 7
    bic = -2 * total_ll + k * np.log(n)
    return mean_ll, mean_acc, bic

# Main grid search for each profile set

def run_grid_cell(args):
    """
    Wrapper to evaluate a single cell of the (prob_reward, prob_hint)
    grid. This is intentionally structured so it can be submitted to a
    ProcessPoolExecutor worker.

    Parameters
    ----------
    args : tuple
        (i, j, prob_reward, prob_hint, profile_set)

    Returns
    -------
    tuple
        (i, j, ll_k1, acc_k1, bic_k1, ll_k2, acc_k2, bic_k2)
    """
    i, j, prob_reward, prob_hint, profile_set = args
    ll1, acc1, bic1_val = evaluate_config(profile_set['k1_profiles'], profile_set['k1_Z'], prob_hint, prob_reward, 'K=1')
    ll2, acc2, bic2_val = evaluate_config(profile_set['k2_profiles'], profile_set['k2_Z'], prob_hint, prob_reward, 'K=2')
    return (i, j, ll1, acc1, bic1_val, ll2, acc2, bic2_val)


if __name__ == "__main__":
    # Iterate over profile sets and evaluate the full grid for each set.
    for profile_set in profile_sets:
        print(f"\n=== Running profile set {profile_set['name']} ===")
        ll_k1 = np.zeros((4, 4))
        ll_k2 = np.zeros((4, 4))
        acc_k1 = np.zeros((4, 4))
        acc_k2 = np.zeros((4, 4))
        bic_k1 = np.zeros((4, 4))
        bic_k2 = np.zeros((4, 4))
        tasks = []
        for i, prob_reward in enumerate(prob_rewards):
            for j, prob_hint in enumerate(prob_hints):
                tasks.append((i, j, prob_reward, prob_hint, profile_set))
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_grid_cell, t) for t in tasks]
            for future in as_completed(futures):
                i, j, ll1, acc1, bic1_val, ll2, acc2, bic2_val = future.result()
                ll_k1[i, j] = ll1
                ll_k2[i, j] = ll2
                acc_k1[i, j] = acc1
                acc_k2[i, j] = acc2
                bic_k1[i, j] = bic1_val
                bic_k2[i, j] = bic2_val
        ll_delta = ll_k2 - ll_k1
        acc_delta = acc_k2 - acc_k1
        bic_delta = bic_k2 - bic_k1
        # Plotting: create three side-by-side heatmaps for the deltas
        # (K=2 minus K=1) of log-likelihood, accuracy, and BIC. The
        # arrays are small (4x4), so simple imshow usage is sufficient.
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        im0 = axs[0].imshow(ll_delta, cmap='coolwarm', origin='lower')
        axs[0].set_title(f'Δ Log Likelihood (K=2 - K=1) [{profile_set["name"]}]')
        axs[0].set_xticks(range(4))
        axs[0].set_yticks(range(4))
        axs[0].set_xticklabels(prob_hints)
        axs[0].set_yticklabels(prob_rewards)
        plt.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(acc_delta, cmap='coolwarm', origin='lower')
        axs[1].set_title(f'Δ Accuracy (K=2 - K=1) [{profile_set["name"]}]')
        axs[1].set_xticks(range(4))
        axs[1].set_yticks(range(4))
        axs[1].set_xticklabels(prob_hints)
        axs[1].set_yticklabels(prob_rewards)
        plt.colorbar(im1, ax=axs[1])
        im2 = axs[2].imshow(bic_delta, cmap='coolwarm', origin='lower')
        axs[2].set_title(f'Δ BIC (K=2 - K=1) [{profile_set["name"]}]')
        axs[2].set_xticks(range(4))
        axs[2].set_yticks(range(4))
        axs[2].set_xticklabels(prob_hints)
        axs[2].set_yticklabels(prob_rewards)
        plt.colorbar(im2, ax=axs[2])
        for ax in axs:
            ax.set_xlabel('prob_hint')
            ax.set_ylabel('prob_reward')
        plt.tight_layout()
        # Save figure to results/figures (relative to this script)
        figures_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, f'k2_vs_k1_heatmaps_{profile_set["name"]}.png'))
        plt.close(fig)
        # Save raw data to results/data (relative to this script)
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'data')
        os.makedirs(data_dir, exist_ok=True)
        np.savez(os.path.join(data_dir, f'k2_vs_k1_grid_results_{profile_set["name"]}.npz'),
                 prob_rewards=prob_rewards, prob_hints=prob_hints,
                 ll_k1=ll_k1, ll_k2=ll_k2, acc_k1=acc_k1, acc_k2=acc_k2, bic_k1=bic_k1, bic_k2=bic_k2,
                 ll_delta=ll_delta, acc_delta=acc_delta, bic_delta=bic_delta)
