"""
Grid experiment: Compare K=1 and K=2 across prob_reward and prob_hint
Outputs: delta log likelihood, delta accuracy, delta BIC heatmaps
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.experiment_config import *
from src.environment import TwoArmedBandit
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL, run_episode_with_ll
from pymdp.agent import Agent
from pymdp import utils

# Experiment grid
prob_rewards = [0.65, 0.75, 0.85, 0.95]
prob_hints = [0.65, 0.75, 0.85, 0.95]

num_trials = 800
num_runs = 10
reversal_schedule = [i for i in range(40, num_trials, 40)]


# Profile sets (move to top level)
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
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               prob_hint, prob_reward)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)
    
    # Get policies
    C_temp = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
    temp_agent = Agent(A=A, B=B, C=C_temp, D=D,
                     policy_len=2, inference_horizon=1,
                     control_fac_idx=[1], use_utility=True,
                     use_states_info_gain=True,
                     action_selection="stochastic", gamma=16)
    policies = temp_agent.policies
    num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
    value_fn = make_value_fn('M3',
                            profiles=profiles,
                            Z=Z,
                            policies=policies,
                            num_actions_per_factor=num_actions_per_factor)
    all_ll = []
    all_acc = []
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
        # Log likelihood
        all_ll.extend(logs['ll'])
        # Accuracy: fraction of correct arm choices
        for ctx, choice in zip(logs['context'], logs['action']):
            if ctx == 'left_better' and choice == 'act_left':
                all_acc.append(1)
            elif ctx == 'right_better' and choice == 'act_right':
                all_acc.append(1)
            else:
                all_acc.append(0)
    total_ll = np.sum(all_ll)
    mean_ll = np.mean(all_ll)
    mean_acc = np.mean(all_acc)
    n = num_trials * num_runs
    k = 3 if name == 'K=1' else 7
    bic = -2 * total_ll + k * np.log(n)
    return mean_ll, mean_acc, bic

# Main grid search for each profile set

def run_grid_cell(args):
    i, j, prob_reward, prob_hint, profile_set = args
    ll1, acc1, bic1_val = evaluate_config(profile_set['k1_profiles'], profile_set['k1_Z'], prob_hint, prob_reward, 'K=1')
    ll2, acc2, bic2_val = evaluate_config(profile_set['k2_profiles'], profile_set['k2_Z'], prob_hint, prob_reward, 'K=2')
    return (i, j, ll1, acc1, bic1_val, ll2, acc2, bic2_val)


if __name__ == "__main__":
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
        # Plotting
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
