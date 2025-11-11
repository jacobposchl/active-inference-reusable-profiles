"""
Test K=2 with hint-seeking profiles vs K=1
Key idea: During uncertainty, profiles mix to prefer hints more than K=1
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.experiment_config import *
from src.environment import TwoArmedBandit
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL, run_episode_with_ll
from pymdp.agent import Agent
from concurrent.futures import ProcessPoolExecutor, as_completed
from pymdp import utils


def evaluate_config(profiles, Z, A, B, D, name, num_trials=800, num_runs=10, seed=42):
    """Evaluate a configuration."""
    # Get policies
    C_temp = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
    temp_agent = Agent(A=A, B=B, C=C_temp, D=D,
                     policy_len=2, inference_horizon=1,
                     control_fac_idx=[1], use_utility=True,
                     use_states_info_gain=True,
                     action_selection="stochastic", gamma=16)
    policies = temp_agent.policies
    num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
    # Create value function
    value_fn = make_value_fn('M3',
                            profiles=profiles,
                            Z=Z,
                            policies=policies,
                            num_actions_per_factor=num_actions_per_factor)
    total_rewards_list = []
    log_likelihoods = []
    reversal_schedule = [i for i in range(40, 800, 40)]
    for run in range(num_runs):
        run_seed = seed + run
        np.random.seed(run_seed)
        env = TwoArmedBandit(
            probability_hint=PROBABILITY_HINT,
            probability_reward=PROBABILITY_REWARD,
            reversal_schedule=reversal_schedule
        )
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                            OBSERVATION_HINTS, OBSERVATION_REWARDS,
                            OBSERVATION_CHOICES, ACTION_CHOICES,
                            reward_mod_idx=1)
        logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)
        # Calculate total rewards
        total_rewards = np.sum(np.array(logs['reward']) == 1)
        total_rewards_list.append(total_rewards)
        # Calculate log likelihood (sum over all trials)
        if 'll' in logs:
            log_likelihoods.append(np.sum(logs['ll']))
        else:
            log_likelihoods.append(float('nan'))

    mean_rewards = np.mean(total_rewards_list)
# Multiprocessing helper
def run_profile_eval(args):
    profiles, Z, name, num_trials, num_runs, seed = args
    # Build environment and model matrices
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)
    mean_reward, mean_ll = evaluate_config(profiles, Z, A, B, D, name, num_trials=num_trials, num_runs=num_runs, seed=seed)
    return (name, mean_reward, mean_ll)
    std_rewards = np.std(total_rewards_list)
    mean_ll = np.nanmean(log_likelihoods)
    std_ll = np.nanstd(log_likelihoods)

    print(f"{name}: {mean_rewards:.1f} ± {std_rewards:.1f} | logL: {mean_ll:.1f} ± {std_ll:.1f}")
    return mean_rewards, std_rewards, mean_ll, std_ll


def main():
    print("=" * 80)
    print("TEST: HINT-SEEKING K=2 vs K=1")
    print("=" * 80)
    print()
    
    # Build generative model
    print("Building generative model...")
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)
    print()
    
    # ACTION_CHOICES = ['act_start', 'act_hint', 'act_left', 'act_right']
    # xi_logits format: [start, hint, left, right]
    
    print("=" * 80)
    print("CONFIGURATION COMPARISON")
    print("=" * 80)
    print()
    
    # ========================================================================
    # K=1 Baseline: No hint bias
    # ========================================================================
    print("K=1 BASELINE:")
    print("  xi_logits: [0, 0, 0, 0] - no action biases")
    print()
    
    k1_profiles = [
        {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, 0.0, 0.0]}
    ]
    k1_Z = np.ones((2, 1))
    
    k1_mean, k1_std, k1_ll, k1_ll_std = evaluate_config(k1_profiles, k1_Z, A, B, D, "K=1 (no bias)", 
                                      num_trials=800, num_runs=10, seed=42)
    print()
    
    # ========================================================================
    # K=2: BALANCED approach - moderate hint + action biases
    # ========================================================================
    print("K=2 BALANCED:")
    print("  Profile 0: xi_logits: [0, +1.0, +2.0, -2.0] - slight hint, left bias")
    print("  Profile 1: xi_logits: [0, +1.0, -2.0, +2.0] - slight hint, right bias")
    print("  When mixed 50/50:")
    print("    hint: +1.0 (modest hint preference during uncertainty)")
    print("    left: 0.0 (cancels)")
    print("    right: 0.0 (cancels)")
    print()
    
    k2_profiles = [
        {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, +4.0, +6.0, -6.0]},
        {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, +4.0, -6.0, +6.0]}
    ]
    k2_Z = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    k2_mean, k2_std, k2_ll, k2_ll_std = evaluate_config(k2_profiles, k2_Z, A, B, D, "K=2 (balanced)", 
                                      num_trials=800, num_runs=10, seed=42)
    print()
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    print()
    
    # Get policies for analysis
    C_temp = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
    temp_agent = Agent(A=A, B=B, C=C_temp, D=D,
                     policy_len=2, inference_horizon=1,
                     control_fac_idx=[1], use_utility=True,
                     use_states_info_gain=True,
                     action_selection="stochastic", gamma=16)
    policies = temp_agent.policies
    num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
    
    # Check what happens at different belief states
    k2_value_fn = make_value_fn('M3', profiles=k2_profiles, Z=k2_Z, 
                                policies=policies, num_actions_per_factor=num_actions_per_factor)
    k1_value_fn = make_value_fn('M3', profiles=k1_profiles, Z=k1_Z,
                                policies=policies, num_actions_per_factor=num_actions_per_factor)
    
    print("Action preferences at different belief states:")
    print()
    
    test_beliefs = [
        ([1.0, 0.0], "Certain left_better"),
        ([0.5, 0.5], "Uncertain (50/50)"),
        ([0.0, 1.0], "Certain right_better"),
    ]
    
    for q_context, desc in test_beliefs:
        print(f"{desc}: q={q_context}")
        
        # K=2 analysis
        C_t, E_t, gamma_t = k2_value_fn(q_context, t=0)
        w = np.array(q_context) @ k2_Z
        w = w / (w.sum() + 1e-12)
        xi_mixed = (w[0] * np.array(k2_profiles[0]['xi_logits']) + 
                    w[1] * np.array(k2_profiles[1]['xi_logits']))
        
        print(f"  K=2 mixed xi: {xi_mixed}")
        print(f"    start={xi_mixed[0]:.1f}, hint={xi_mixed[1]:.1f}, left={xi_mixed[2]:.1f}, right={xi_mixed[3]:.1f}")
        
        # K=1 analysis
        C_t, E_t, gamma_t = k1_value_fn(q_context, t=0)
        print(f"  K=1 xi: {k1_profiles[0]['xi_logits']}")
        print(f"    start={k1_profiles[0]['xi_logits'][0]:.1f}, hint={k1_profiles[0]['xi_logits'][1]:.1f}, left={k1_profiles[0]['xi_logits'][2]:.1f}, right={k1_profiles[0]['xi_logits'][3]:.1f}")
        print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"K=1 (no bias):   {k1_mean:.1f} ± {k1_std:.1f} | logL: {k1_ll:.1f} ± {k1_ll_std:.1f}")
    print(f"K=2 (balanced): {k2_mean:.1f} ± {k2_std:.1f} | logL: {k2_ll:.1f} ± {k2_ll_std:.1f}")

    diff = k2_mean - k1_mean
    pct = (diff / abs(k1_mean)) * 100 if k1_mean != 0 else 0
    diff_ll = k2_ll - k1_ll
    pct_ll = (diff_ll / abs(k1_ll)) * 100 if k1_ll != 0 else 0

    print()
    if diff > 0:
        print(f"✅ K=2 WINS by {diff:.1f} rewards ({pct:+.1f}%)")
    else:
        print(f"❌ K=2 loses by {diff:.1f} rewards ({pct:+.1f}%)")
    print()
    if diff_ll > 0:
        print(f"✅ K=2 WINS logL by {diff_ll:.1f} ({pct_ll:+.1f}%)")
    else:
        print(f"❌ K=2 loses logL by {diff_ll:.1f} ({pct_ll:+.1f}%)")


if __name__ == "__main__":
    print("=" * 80)
    print("BATCH PROFILE TESTING: K=1 vs K=2")
    print("=" * 80)
    print()

    # === Batch profile testing template ===
    # Define multiple K=1 and K=2 profile sets to test
        k1_profile_sets = [
                # (profiles, Z, name)
                # Neutral
                ([{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, 0.0, 0.0]}], np.ones((2, 1)), 'K=1_neutral'),
                # Hint-seeking
                ([{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 1.0, 0.0, 0.0]}], np.ones((2, 1)), 'K=1_hintseek'),
                # Left-biased
                ([{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, 6.0, -6.0]}], np.ones((2, 1)), 'K=1_leftbias'),
                # Right-biased
                ([{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, -6.0, 6.0]}], np.ones((2, 1)), 'K=1_rightbias'),
                # Strong hint-seeking
                ([{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 4.0, 0.0, 0.0]}], np.ones((2, 1)), 'K=1_strong_hintseek'),
        ]
        k2_profile_sets = [
                # (profiles, Z, name)
                # Strong hint-seeking, opposing left/right
                ([{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 4.0, 6.0, -6.0]},
                    {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 4.0, -6.0, 6.0]}],
                 np.array([[1.0, 0.0], [0.0, 1.0]]), 'K=2_strong_hintseek'),
                # Moderate hint-seeking, opposing left/right
                ([{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 2.0, 3.0, -3.0]},
                    {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 2.0, -3.0, 3.0]}],
                 np.array([[1.0, 0.0], [0.0, 1.0]]), 'K=2_moderate_hintseek'),
                # Strong left/right bias, no hint
                ([{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, 6.0, -6.0]},
                    {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, -6.0, 6.0]}],
                 np.array([[1.0, 0.0], [0.0, 1.0]]), 'K=2_left_right_bias'),
                # Both strong hint-seeking
                ([{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 4.0, 0.0, 0.0]},
                    {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 4.0, 0.0, 0.0]}],
                 np.array([[1.0, 0.0], [0.0, 1.0]]), 'K=2_both_hintseek'),
                # Mixed: one hint-seeking, one left-biased
                ([{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 4.0, 0.0, 0.0]},
                    {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, 6.0, -6.0]}],
                 np.array([[1.0, 0.0], [0.0, 1.0]]), 'K=2_hintseek_leftbias'),
        ]
    # Combine all jobs
    jobs = []
    for profiles, Z, name in k1_profile_sets:
        jobs.append((profiles, Z, name, 800, 10, 42))
    for profiles, Z, name in k2_profile_sets:
        jobs.append((profiles, Z, name, 800, 10, 42))
    # Run in parallel
    print(f"Evaluating {len(jobs)} profile sets in parallel...")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_profile_eval, job) for job in jobs]
        results = [future.result() for future in as_completed(futures)]
    # Sort and print, with AIC/BIC
    results.sort(key=lambda x: (x[0], -x[1]))
    print("\n=== BATCH PROFILE TEST RESULTS ===")
    n = 800 * 10  # num_trials * num_runs
    print(f"{'Name':20s} | {'Reward':>7s} | {'LogLik':>9s} | {'AIC':>9s} | {'BIC':>9s}")
    for name, mean_reward, mean_ll in results:
        if name.startswith('K=1'):
            k = 3
        else:
            k = 7
        aic = 2 * k - 2 * mean_ll
        bic = k * np.log(n) - 2 * mean_ll
        print(f"{name:20s} | {mean_reward:7.2f} | {mean_ll:9.2f} | {aic:9.2f} | {bic:9.2f}")
