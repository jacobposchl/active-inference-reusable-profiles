"""
Quick test: Single K=2 configuration vs K=1 baseline

Tests paper-aligned opposing action biases to see if they beat K=1.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.experiment_config import *
from src.environment import TwoArmedBandit
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunnerWithLL, run_episode_with_ll
from pymdp.agent import Agent
from pymdp import utils


def evaluate_config(K, profiles, Z, A, B, D, num_trials=400, num_runs=20, seed=42):
    """Quick evaluation of a configuration."""
    
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
    
    for run in range(num_runs):
        run_seed = seed + run
        np.random.seed(run_seed)
        
        env = TwoArmedBandit(
            probability_hint=PROBABILITY_HINT,
            probability_reward=PROBABILITY_REWARD,
            reversal_schedule=DEFAULT_REVERSAL_SCHEDULE
        )
        
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                            OBSERVATION_HINTS, OBSERVATION_REWARDS,
                            OBSERVATION_CHOICES, ACTION_CHOICES,
                            reward_mod_idx=1)
        
        logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)
        
        # Calculate total rewards
        total_rewards = sum(1 if r == 'observe_reward' else (-1 if r == 'observe_loss' else 0) 
                          for r in logs['reward_label'])
        total_rewards_list.append(total_rewards)
    
    mean_rewards = np.mean(total_rewards_list)
    std_rewards = np.std(total_rewards_list)
    
    return mean_rewards, std_rewards


def main():
    print("=" * 70)
    print("QUICK TEST: K=2 vs K=1 Baseline")
    print("=" * 70)
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
    
    # ========================================================================
    # K=1 BASELINE (we know this works well)
    # ========================================================================
    print("Testing K=1 baseline (γ=4.5, neutral)...")
    
    k1_profile = {
        'gamma': 4.5,
        'phi_logits': [0.0, -10.0, 7.0],  # Very cautious
        'xi_logits': [0.0, 0.0, 0.0, 0.0]  # Neutral
    }
    k1_Z = np.ones((2, 1))
    
    k1_mean, k1_std = evaluate_config(1, [k1_profile], k1_Z, A, B, D, 
                                      num_trials=400, num_runs=20, seed=42)
    
    print(f"K=1 Result: {k1_mean:.1f} ± {k1_std:.1f} rewards")
    print()
    
    # ========================================================================
    # K=2 CONFIGURATIONS TO TEST
    # ========================================================================
    
    test_configs = []
    
    # Config 1: Paper-aligned strong biases (±2.0), moderate γ
    test_configs.append({
        'name': 'Strong biases ±2.0, γ=3.0, hard Z',
        'profiles': [
            {'gamma': 3.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, +2.0, -2.0]},
            {'gamma': 3.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, -2.0, +2.0]}
        ],
        'Z': np.array([[1.0, 0.0], [0.0, 1.0]])
    })
    
    # Config 2: Moderate biases (±1.5), moderate γ
    test_configs.append({
        'name': 'Moderate biases ±1.5, γ=3.0, hard Z',
        'profiles': [
            {'gamma': 3.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, +1.5, -1.5]},
            {'gamma': 3.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, -1.5, +1.5]}
        ],
        'Z': np.array([[1.0, 0.0], [0.0, 1.0]])
    })
    
    # Config 3: Strong biases (±2.0), higher γ
    test_configs.append({
        'name': 'Strong biases ±2.0, γ=5.0, hard Z',
        'profiles': [
            {'gamma': 5.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, +2.0, -2.0]},
            {'gamma': 5.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, -2.0, +2.0]}
        ],
        'Z': np.array([[1.0, 0.0], [0.0, 1.0]])
    })
    
    # Config 4: Very strong biases (±3.0), moderate γ
    test_configs.append({
        'name': 'Very strong biases ±3.0, γ=3.0, hard Z',
        'profiles': [
            {'gamma': 3.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, +3.0, -3.0]},
            {'gamma': 3.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, -3.0, +3.0]}
        ],
        'Z': np.array([[1.0, 0.0], [0.0, 1.0]])
    })
    
    # Config 5: Moderate biases with soft Z
    test_configs.append({
        'name': 'Moderate biases ±1.5, γ=3.0, soft Z',
        'profiles': [
            {'gamma': 3.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, +1.5, -1.5]},
            {'gamma': 3.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, -1.5, +1.5]}
        ],
        'Z': np.array([[0.8, 0.2], [0.2, 0.8]])
    })
    
    # Config 6: Strong biases with matched loss aversion
    test_configs.append({
        'name': 'Strong biases ±2.0, γ=3.0, cautious, hard Z',
        'profiles': [
            {'gamma': 3.0, 'phi_logits': [0.0, -10.0, 7.0], 'xi_logits': [0.0, 0.0, +2.0, -2.0]},
            {'gamma': 3.0, 'phi_logits': [0.0, -10.0, 7.0], 'xi_logits': [0.0, 0.0, -2.0, +2.0]}
        ],
        'Z': np.array([[1.0, 0.0], [0.0, 1.0]])
    })
    
    # Config 7: Weak biases (±1.0), moderate γ
    test_configs.append({
        'name': 'Weak biases ±1.0, γ=3.0, hard Z',
        'profiles': [
            {'gamma': 3.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, +1.0, -1.0]},
            {'gamma': 3.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, -1.0, +1.0]}
        ],
        'Z': np.array([[1.0, 0.0], [0.0, 1.0]])
    })
    
    # Config 8: Strong biases with balanced γ matching K=1
    test_configs.append({
        'name': 'Strong biases ±2.0, γ=4.5, hard Z',
        'profiles': [
            {'gamma': 4.5, 'phi_logits': [0.0, -10.0, 7.0], 'xi_logits': [0.0, 0.0, +2.0, -2.0]},
            {'gamma': 4.5, 'phi_logits': [0.0, -10.0, 7.0], 'xi_logits': [0.0, 0.0, -2.0, +2.0]}
        ],
        'Z': np.array([[1.0, 0.0], [0.0, 1.0]])
    })
    
    # ========================================================================
    # TEST ALL K=2 CONFIGS
    # ========================================================================
    
    print("Testing K=2 configurations...")
    print("-" * 70)
    
    best_config = None
    best_mean = -np.inf
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n[{i}/{len(test_configs)}] {config['name']}")
        
        k2_mean, k2_std = evaluate_config(2, config['profiles'], config['Z'], 
                                         A, B, D, num_trials=400, num_runs=20, 
                                         seed=100 + i)
        
        improvement = k2_mean - k1_mean
        improvement_pct = (improvement / abs(k1_mean)) * 100
        
        print(f"      Result: {k2_mean:.1f} ± {k2_std:.1f} rewards")
        print(f"      vs K=1: {improvement:+.1f} ({improvement_pct:+.1f}%)", end="")
        
        if improvement > 0:
            print(" ✅")
            if k2_mean > best_mean:
                best_mean = k2_mean
                best_config = (config, k2_mean, k2_std)
        else:
            print(" ❌")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nK=1 Baseline: {k1_mean:.1f} ± {k1_std:.1f} rewards")
    
    if best_config:
        config, mean, std = best_config
        improvement = mean - k1_mean
        improvement_pct = (improvement / abs(k1_mean)) * 100
        print(f"\nBest K=2: {mean:.1f} ± {std:.1f} rewards")
        print(f"Config: {config['name']}")
        print(f"\n✅ Improvement: +{improvement:.1f} rewards (+{improvement_pct:.1f}%)")
    else:
        print(f"\n❌ No K=2 configuration beat K=1 baseline")
        print("Consider: biases too weak, too strong, or mechanism doesn't apply to this task")


if __name__ == "__main__":
    main()
