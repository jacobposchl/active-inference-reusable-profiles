import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.experiment_config import *
from src.environment import TwoArmedBandit
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL, run_episode_with_ll
from pymdp.agent import Agent
from pymdp import utils

# === SETTINGS (from heatmap C, top-left cell) ===
prob_reward = 0.65
prob_hint = 0.65
num_trials = 800
num_runs = 10
reversal_schedule = [i for i in range(40, num_trials, 40)]

# === Profiles from set C ===
k1_profile = {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 3.0, 0.0, 0.0]}
k2_profiles = [
    {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 3.0, 6.0, -6.0]},
    {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 3.0, -6.0, 6.0]}
]

Z_matrices = {
    'K=1': np.ones((2, 1)),
    'Hard': np.array([[1.0, 0.0], [0.0, 1.0]]),
    'Medium': np.array([[0.8, 0.2], [0.2, 0.8]]),
    'Soft': np.array([[0.6, 0.4], [0.4, 0.6]])
}

results = {}

def evaluate(profiles, Z, label):
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
    total_rewards = []
    accuracies = []
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
        # Total reward
        reward = 0
        for label in logs['reward_label']:
            if label == 'observe_reward':
                reward += 1
            elif label == 'observe_loss':
                reward -= 1
        total_rewards.append(reward)
        # Accuracy
        correct = 0
        for ctx, choice in zip(logs['context'], logs['action']):
            if ctx == 'left_better' and choice == 'act_left':
                correct += 1
            elif ctx == 'right_better' and choice == 'act_right':
                correct += 1
        accuracies.append(correct / num_trials)
    return np.mean(total_rewards), np.std(total_rewards), np.mean(accuracies), np.std(accuracies)

# Evaluate K=1
results['K=1'] = evaluate([k1_profile], Z_matrices['K=1'], 'K=1')
# Evaluate K=2 with different Z
for label in ['Hard', 'Medium', 'Soft']:
    results[label] = evaluate(k2_profiles, Z_matrices[label], label)

# Plot
labels = list(results.keys())
means = [results[k][0] for k in labels]
stds = [results[k][1] for k in labels]
acc_means = [results[k][2] for k in labels]
acc_stds = [results[k][3] for k in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 5))
rects1 = ax1.bar(x - width/2, means, width, yerr=stds, label='Total Reward', capsize=5)
rects2 = ax1.bar(x + width/2, acc_means, width, yerr=acc_stds, label='Accuracy', capsize=5)

ax1.set_ylabel('Performance')
ax1.set_title('Ablation: Z-matrix Mixing (Profile Set C, prob_reward=0.65, prob_hint=0.65)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(['Total Reward', 'Accuracy'])
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
