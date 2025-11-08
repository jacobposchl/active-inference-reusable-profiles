"""
Run a single trial demonstration with one model.

This is a simple example showing how to use the modular code.
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.experiment_config import *
from src.environment import TwoArmedBandit
from src.models import build_A, build_B, build_D, make_value_fn, AgentRunner, run_episode
from src.utils import find_reversals, trial_accuracy
from src.utils.plotting import plot_gamma_over_time, plot_entropy_over_time


def main():
    """Run single trial demonstration."""
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Choose model
    model_name = 'M3'  # Change to 'M1' or 'M2' to try other models
    
    print("="*70)
    print(f"RUNNING SINGLE TRIAL DEMONSTRATION - MODEL {model_name}")
    print("="*70)
    
    # Build environment
    env = TwoArmedBandit(
        probability_hint=PROBABILITY_HINT,
        probability_reward=PROBABILITY_REWARD,
        reversal_schedule=DEFAULT_REVERSAL_SCHEDULE
    )
    
    print(f"Environment initialized")
    print(f"Initial context: {env.context}")
    print(f"Reversals scheduled at: {DEFAULT_REVERSAL_SCHEDULE}")
    
    # Build generative model (agent's beliefs)
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)
    
    print(f"\nGenerative model constructed:")
    print(f"  Modalities: {NUM_MODALITIES}")
    print(f"  State factors: {NUM_FACTORS}")
    
    # Create value function based on model
    if model_name == 'M1':
        value_fn = make_value_fn('M1', **M1_DEFAULTS)
        print(f"\nModel M1: Static global")
        print(f"  gamma = {M1_DEFAULTS['gamma']}")
        
    elif model_name == 'M2':
        def gamma_schedule(q, t, g_base=1.6, k=1.0):
            p = np.clip(np.asarray(q, float), 1e-12, 1.0)
            H = -(p * np.log(p)).sum()
            return g_base / (1.0 + k * H)
        
        value_fn = make_value_fn('M2', 
                                C_reward_logits=M2_DEFAULTS['C_reward_logits'],
                                gamma_schedule=gamma_schedule)
        print(f"\nModel M2: Dynamic global precision")
        print(f"  gamma_base = {M2_DEFAULTS['gamma_base']}")
        
    elif model_name == 'M3':
        # Need to get policies first
        from pymdp.agent import Agent
        from pymdp import utils
        
        C_temp = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
        temp_agent = Agent(A=A, B=B, C=C_temp, D=D,
                         policy_len=2, inference_horizon=1,
                         control_fac_idx=[1], use_utility=True,
                         use_states_info_gain=True,
                         action_selection="stochastic", gamma=16)
        
        policies = temp_agent.policies
        num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
        
        import numpy as np
        value_fn = make_value_fn('M3',
                                profiles=M3_DEFAULTS['profiles'],
                                Z=np.array(M3_DEFAULTS['Z']),
                                policies=policies,
                                num_actions_per_factor=num_actions_per_factor)
        print(f"\nModel M3: Profile model")
        print(f"  Profiles: {len(M3_DEFAULTS['profiles'])}")
    
    # Create agent runner
    runner = AgentRunner(A, B, D, value_fn,
                        OBSERVATION_HINTS, OBSERVATION_REWARDS,
                        OBSERVATION_CHOICES, ACTION_CHOICES,
                        reward_mod_idx=1)
    
    print(f"\nAgent initialized with {len(runner.agent.policies)} policies")
    
    # Run episode
    print(f"\nRunning {DEFAULT_TRIALS} trials...")
    logs = run_episode(runner, env, T=DEFAULT_TRIALS, verbose=False)
    
    # Compute metrics
    reversals = find_reversals(logs['context'])
    acc = trial_accuracy(logs['action'], logs['context'])
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Mean accuracy: {acc.mean():.3f}")
    print(f"Reversals detected: {len(reversals)}")
    print(f"Mean gamma: {np.mean(logs['gamma']):.3f}")
    print(f"Gamma range: [{np.min(logs['gamma']):.3f}, {np.max(logs['gamma']):.3f}]")
    
    # Generate plots
    print(f"\nGenerating plots...")
    
    belief_left_series = [b[0] for b in logs['belief']]
    
    plot_gamma_over_time(
        logs['gamma'],
        reversals=reversals,
        belief_left_series=belief_left_series,
        roll_k=ROLLING_WINDOW,
        title=f"Policy precision (γ) over trials — {model_name}"
    )
    
    plot_entropy_over_time(
        logs['belief'],
        reversals=reversals,
        roll_k=5,
        title=f"Belief entropy — {model_name}"
    )
    
    print(f"\n{'='*70}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
