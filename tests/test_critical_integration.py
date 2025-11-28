"""
Critical integration tests - fast, focused on catching real bugs.

These tests validate the most important integration points that could cause
silent failures in model recovery experiments.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.experiment_config import *
from src.utils.recovery_helpers import build_abd, _generate_trial_level_predictions
from src.utils.model_utils import create_model
from src.environment import TwoArmedBandit
from src.models.agent_wrapper import AgentRunnerWithLL, run_episode_with_ll
from src.utils.ll_eval import evaluate_ll_with_valuefn


class TestCriticalBeliefPropagation:
    """CRITICAL: These tests catch the belief propagation bug."""
    
    def test_beliefs_dont_reset_during_ll_eval(self):
        """During LL evaluation, beliefs should NOT reset to 50/50."""
        A, B, D = build_abd()
        
        # Generate reference data
        value_fn_gen = create_model('M1', A, B, D)
        env = TwoArmedBandit(context='volatile', reversal_schedule=[])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=30)
        
        # Evaluate with M1 - beliefs should stay high for volatile
        value_fn_eval = create_model('M1', A, B, D)
        rows = _generate_trial_level_predictions(value_fn_eval, A, B, D, ref_logs)
        
        # After first few trials, beliefs should be near [1, 0] for volatile
        for t in range(5, 25):
            p_volatile = rows[t]['belief_context'][0]
            assert p_volatile > 0.8, \
                f"CRITICAL BUG: At t={t}, P(volatile)={p_volatile:.3f} - beliefs are resetting!"
    
    def test_beliefs_track_context_switches(self):
        """Beliefs should update when context switches."""
        A, B, D = build_abd()
        
        value_fn = create_model('M1', A, B, D)
        env = TwoArmedBandit(context='volatile', reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=50)
        
        # Check beliefs track context
        # First 15 trials: volatile
        for t in range(5, 15):
            p_volatile = ref_logs['belief'][t][0]
            assert p_volatile > 0.7, f"At t={t} (volatile), P(volatile)={p_volatile:.3f}"
        
        # After reversal (t=25+): stable
        for t in range(25, 45):
            p_stable = ref_logs['belief'][t][1]
            assert p_stable > 0.7, f"At t={t} (stable), P(stable)={p_stable:.3f}"


class TestM3ProfileSwitching:
    """CRITICAL: M3 should actually switch profiles based on context."""
    
    def test_m3_gamma_changes_with_context(self):
        """M3 gamma should be different in volatile vs stable."""
        A, B, D = build_abd()
        
        value_fn = create_model('M3', A, B, D)
        env = TwoArmedBandit(context='volatile', reversal_schedule=[30])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        logs = run_episode_with_ll(runner, env, T=60)
        
        # Average gamma in volatile vs stable
        gamma_volatile = np.mean(logs['gamma'][5:25])
        gamma_stable = np.mean(logs['gamma'][35:55])
        
        assert gamma_stable > gamma_volatile, \
            f"CRITICAL: M3 gamma should differ: volatile={gamma_volatile:.2f}, stable={gamma_stable:.2f}"
        assert gamma_volatile < 3.0, f"Volatile gamma should be low (<3.0), got {gamma_volatile:.2f}"
        assert gamma_stable > 3.0, f"Stable gamma should be high (>3.0), got {gamma_stable:.2f}"


class TestLLEvaluationSanity:
    """CRITICAL: LL evaluation should be numerically stable."""
    
    def test_ll_is_finite_for_all_models(self):
        """All models should produce finite LL values."""
        A, B, D = build_abd()
        
        value_fn_gen = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=40)
        
        for model_name in ['M1', 'M2', 'M3']:
            value_fn = create_model(model_name, A, B, D)
            total_ll, ll_seq = evaluate_ll_with_valuefn(value_fn, A, B, D, ref_logs)
            
            assert np.isfinite(total_ll), f"{model_name}: total LL not finite: {total_ll}"
            assert all(np.isfinite(ll) for ll in ll_seq), f"{model_name}: some LLs not finite"
            assert total_ll < 0, f"{model_name}: LL should be negative, got {total_ll}"
    
    def test_ll_consistency_across_evaluations(self):
        """Same model/data should give same LL (deterministic)."""
        A, B, D = build_abd()
        np.random.seed(42)
        
        value_fn_gen = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=30)
        
        value_fn = create_model('M1', A, B, D)
        
        # Evaluate twice - should get same result
        ll1, _ = evaluate_ll_with_valuefn(value_fn, A, B, D, ref_logs)
        ll2, _ = evaluate_ll_with_valuefn(value_fn, A, B, D, ref_logs)
        
        assert abs(ll1 - ll2) < 1e-6, f"LL should be deterministic: {ll1} vs {ll2}"


class TestDataConsistency:
    """CRITICAL: Generated data should be consistent and valid."""
    
    def test_context_observations_match_true_context(self):
        """Context observations must match true context."""
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
            assert actual == expected, \
                f"CRITICAL: At t={t}, context_label={actual} but true context={logs['context'][t]}"
    
    def test_all_required_fields_present(self):
        """Generated logs must have all fields needed for model recovery."""
        A, B, D = build_abd()
        
        value_fn = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        logs = run_episode_with_ll(runner, env, T=30)
        
        required = ['action', 'context', 'context_label', 'belief', 'gamma',
                   'reward_label', 'choice_label', 'hint_label', 'll']
        
        for field in required:
            assert field in logs, f"CRITICAL: Missing field {field}"
            assert len(logs[field]) == 30, f"Field {field} has wrong length"


class TestTrialLevelPredictions:
    """CRITICAL: Trial-level predictions should be valid."""
    
    def test_action_probs_valid(self):
        """Action probabilities should sum to 1 and be non-negative."""
        A, B, D = build_abd()
        
        value_fn_gen = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[15])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=30)
        
        value_fn_eval = create_model('M1', A, B, D)
        rows = _generate_trial_level_predictions(value_fn_eval, A, B, D, ref_logs)
        
        for t, row in enumerate(rows):
            probs = row['action_probs']
            prob_sum = sum(probs)
            assert abs(prob_sum - 1.0) < 1e-5, \
                f"At t={t}, action probs sum to {prob_sum}, not 1.0"
            assert all(p >= 0 for p in probs), \
                f"At t={t}, some action probs are negative: {probs}"
    
    def test_ll_matches_action_prob(self):
        """LL should equal log of probability assigned to generated action."""
        A, B, D = build_abd()
        
        value_fn_gen = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[15])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=30)
        
        value_fn_eval = create_model('M1', A, B, D)
        rows = _generate_trial_level_predictions(value_fn_eval, A, B, D, ref_logs)
        
        for t, row in enumerate(rows):
            gen_action = row['gen_action']
            try:
                gen_idx = ACTION_CHOICES.index(gen_action)
                expected_ll = np.log(row['action_probs'][gen_idx] + 1e-16)
                actual_ll = row['ll']
                assert abs(actual_ll - expected_ll) < 1e-5, \
                    f"At t={t}, LL mismatch: expected {expected_ll:.4f}, got {actual_ll:.4f}"
            except ValueError:
                pass  # Skip if action not in list


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

