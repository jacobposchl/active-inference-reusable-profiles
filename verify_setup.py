"""
Quick verification script - run this before model recovery experiments.

This script runs the most critical tests to ensure everything is working.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from config.experiment_config import (
    OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
    ACTION_CHOICES
)

print("=" * 70)
print("VERIFYING MODEL RECOVERY SETUP")
print("=" * 70)

# Test 1: A matrix structure
print("\n[1/5] Testing A matrix structure...")
try:
    from src.utils.recovery_helpers import build_abd
    A, B, D = build_abd()
    assert len(A) == 4, f"A should have 4 modalities, got {len(A)}"
    assert A[3].shape[0] == 2, "Context modality should have 2 observations"
    print("  ✓ A matrix has correct structure")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Belief propagation during LL eval
print("\n[2/5] Testing belief propagation (CRITICAL)...")
try:
    from src.utils.recovery_helpers import build_abd, _generate_trial_level_predictions
    from src.utils.model_utils import create_model
    from src.environment import TwoArmedBandit
    from src.models.agent_wrapper import AgentRunnerWithLL, run_episode_with_ll
    
    A, B, D = build_abd()
    value_fn_gen = create_model('M1', A, B, D)
    env = TwoArmedBandit(context='volatile', reversal_schedule=[])
    runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                               OBSERVATION_HINTS, OBSERVATION_REWARDS,
                               OBSERVATION_CHOICES, ACTION_CHOICES)
    ref_logs = run_episode_with_ll(runner, env, T=20)
    
    value_fn_eval = create_model('M1', A, B, D)
    rows = _generate_trial_level_predictions(value_fn_eval, A, B, D, ref_logs)
    
    # Check beliefs stay high
    for t in range(5, 15):
        p_volatile = rows[t]['belief_context'][0]
        if p_volatile < 0.8:
            raise AssertionError(f"At t={t}, P(volatile)={p_volatile:.3f} - beliefs resetting!")
    
    print("  ✓ Beliefs propagate correctly (no reset bug)")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: M3 profile switching
print("\n[3/5] Testing M3 profile switching...")
try:
    A, B, D = build_abd()
    value_fn = create_model('M3', A, B, D)
    env = TwoArmedBandit(context='volatile', reversal_schedule=[20])
    runner = AgentRunnerWithLL(A, B, D, value_fn,
                               OBSERVATION_HINTS, OBSERVATION_REWARDS,
                               OBSERVATION_CHOICES, ACTION_CHOICES)
    logs = run_episode_with_ll(runner, env, T=40)
    
    gamma_volatile = np.mean(logs['gamma'][5:15])
    gamma_stable = np.mean(logs['gamma'][25:35])
    
    if gamma_stable <= gamma_volatile:
        raise AssertionError(f"M3 gamma should differ: volatile={gamma_volatile:.2f}, stable={gamma_stable:.2f}")
    
    print(f"  ✓ M3 gamma changes: volatile={gamma_volatile:.2f}, stable={gamma_stable:.2f}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 4: LL evaluation
print("\n[4/5] Testing LL evaluation...")
try:
    from src.utils.ll_eval import evaluate_ll_with_valuefn
    
    A, B, D = build_abd()
    value_fn_gen = create_model('M1', A, B, D)
    env = TwoArmedBandit(reversal_schedule=[15])
    runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                               OBSERVATION_HINTS, OBSERVATION_REWARDS,
                               OBSERVATION_CHOICES, ACTION_CHOICES)
    ref_logs = run_episode_with_ll(runner, env, T=30)
    
    for model_name in ['M1', 'M2', 'M3']:
        value_fn = create_model(model_name, A, B, D)
        total_ll, ll_seq = evaluate_ll_with_valuefn(value_fn, A, B, D, ref_logs)
        
        if not np.isfinite(total_ll):
            raise AssertionError(f"{model_name}: LL not finite: {total_ll}")
        if total_ll > 0:
            raise AssertionError(f"{model_name}: LL should be negative, got {total_ll}")
    
    print("  ✓ LL evaluation works for all models")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 5: Context observations
print("\n[5/5] Testing context observations...")
try:
    A, B, D = build_abd()
    value_fn = create_model('M1', A, B, D)
    env = TwoArmedBandit(context='volatile', reversal_schedule=[20])
    runner = AgentRunnerWithLL(A, B, D, value_fn,
                               OBSERVATION_HINTS, OBSERVATION_REWARDS,
                               OBSERVATION_CHOICES, ACTION_CHOICES)
    logs = run_episode_with_ll(runner, env, T=40)
    
    for t in range(40):
        expected = f"observe_{logs['context'][t]}"
        actual = logs['context_label'][t]
        if actual != expected:
            raise AssertionError(f"At t={t}, context_label={actual} but expected {expected}")
    
    print("  ✓ Context observations match true context")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL CRITICAL TESTS PASSED - Safe to run model recovery!")
print("=" * 70)

