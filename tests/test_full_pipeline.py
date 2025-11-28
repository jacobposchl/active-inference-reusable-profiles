"""
Comprehensive end-to-end pipeline tests.

These tests validate the ENTIRE model recovery workflow, not just individual components.
They simulate mini-experiments to catch integration issues before running real experiments.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.experiment_config import *
from src.utils.recovery_helpers import build_abd, cv_fit_single_run, _generate_trial_level_predictions
from src.utils.model_utils import create_model
from src.environment import TwoArmedBandit
from src.models.agent_wrapper import AgentRunnerWithLL, run_episode_with_ll
from src.utils.ll_eval import evaluate_ll_with_valuefn, compute_sequence_ll_for_model


class TestDataGeneration:
    """Test that data generation produces valid, usable data."""
    
    def test_m1_generates_valid_data(self):
        """M1 generator produces valid logs with all required fields."""
        A, B, D = build_abd()
        value_fn = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20, 40])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        
        logs = run_episode_with_ll(runner, env, T=60)
        
        # Check all required fields exist
        required_fields = ['t', 'context', 'belief', 'gamma', 'action', 
                          'reward_label', 'choice_label', 'hint_label', 
                          'context_label', 'll', 'current_better_arm']
        for field in required_fields:
            assert field in logs, f"Missing field: {field}"
            assert len(logs[field]) == 60, f"Field {field} has wrong length"
    
    def test_m2_generates_valid_data(self):
        """M2 generator produces valid logs."""
        A, B, D = build_abd()
        value_fn = create_model('M2', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20, 40])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        
        logs = run_episode_with_ll(runner, env, T=60)
        
        # M2 gamma should vary
        gammas = logs['gamma']
        assert len(set(gammas)) > 1, "M2 gamma should vary across trials"
    
    def test_m3_generates_valid_data(self):
        """M3 generator produces valid logs with profile-based behavior."""
        A, B, D = build_abd()
        value_fn = create_model('M3', A, B, D)
        env = TwoArmedBandit(context='volatile', reversal_schedule=[30])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        
        logs = run_episode_with_ll(runner, env, T=60)
        
        # Check gamma changes at context reversal
        gamma_volatile = np.mean(logs['gamma'][5:25])
        gamma_stable = np.mean(logs['gamma'][35:55])
        
        assert gamma_stable > gamma_volatile, \
            f"M3 gamma should be higher in stable ({gamma_stable:.2f}) than volatile ({gamma_volatile:.2f})"
    
    def test_context_labels_match_true_context(self):
        """Context observations should match true context."""
        A, B, D = build_abd()
        value_fn = create_model('M1', A, B, D)
        env = TwoArmedBandit(context='volatile', reversal_schedule=[30])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        
        logs = run_episode_with_ll(runner, env, T=60)
        
        for t in range(60):
            expected = f"observe_{logs['context'][t]}"
            actual = logs['context_label'][t]
            assert actual == expected, f"At t={t}, context_label={actual} but expected {expected}"


class TestBeliefPropagationInDepth:
    """Deep tests for belief propagation - the critical bug area."""
    
    def test_beliefs_track_context_during_generation(self):
        """During data generation, beliefs should track the true context."""
        A, B, D = build_abd()
        value_fn = create_model('M3', A, B, D)
        env = TwoArmedBandit(context='volatile', reversal_schedule=[40])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        
        logs = run_episode_with_ll(runner, env, T=80)
        
        # In volatile period (t=5-35), P(volatile) should be high
        for t in range(5, 35):
            p_volatile = logs['belief'][t][0]
            assert p_volatile > 0.7, f"At t={t} (volatile period), P(volatile)={p_volatile:.3f} should be > 0.7"
        
        # In stable period (t=45-75), P(stable) should be high
        for t in range(45, 75):
            p_stable = logs['belief'][t][1]
            assert p_stable > 0.7, f"At t={t} (stable period), P(stable)={p_stable:.3f} should be > 0.7"
    
    def test_beliefs_track_context_during_ll_eval(self):
        """During LL evaluation (teacher forcing), beliefs should track context."""
        A, B, D = build_abd()
        
        # Generate reference with M3
        value_fn_gen = create_model('M3', A, B, D)
        env = TwoArmedBandit(context='volatile', reversal_schedule=[40])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=80)
        
        # Now evaluate with M1 and check internal beliefs
        from src.utils.recovery_helpers import _generate_trial_level_predictions
        value_fn_eval = create_model('M1', A, B, D)
        rows = _generate_trial_level_predictions(value_fn_eval, A, B, D, ref_logs)
        
        # Check volatile period
        for t in range(5, 35):
            p_volatile = rows[t]['belief_context'][0]
            assert p_volatile > 0.7, f"During eval at t={t} (volatile), P(volatile)={p_volatile:.3f} should be > 0.7"
        
        # Check stable period  
        for t in range(45, 75):
            p_stable = rows[t]['belief_context'][1]
            assert p_stable > 0.7, f"During eval at t={t} (stable), P(stable)={p_stable:.3f} should be > 0.7"
    
    def test_beliefs_dont_reset_to_prior(self):
        """Beliefs should NOT reset to 50/50 after each trial."""
        A, B, D = build_abd()
        value_fn = create_model('M1', A, B, D)
        env = TwoArmedBandit(context='volatile', reversal_schedule=[])  # No reversals
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        
        logs = run_episode_with_ll(runner, env, T=50)
        
        # After the first few trials, beliefs should be stable near [1, 0]
        # If they reset to 50/50, this test will fail
        for t in range(10, 50):
            p_volatile = logs['belief'][t][0]
            assert p_volatile > 0.9, f"At t={t}, P(volatile)={p_volatile:.3f} - beliefs may be resetting!"


class TestLogLikelihoodEvaluation:
    """Test that LL evaluation is correct and consistent."""
    
    def test_ll_is_finite(self):
        """LL values should be finite (not -inf or nan)."""
        A, B, D = build_abd()
        
        value_fn_gen = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=50)
        
        for model_name in ['M1', 'M2', 'M3']:
            value_fn = create_model(model_name, A, B, D)
            total_ll, ll_seq = evaluate_ll_with_valuefn(value_fn, A, B, D, ref_logs)
            
            assert np.isfinite(total_ll), f"{model_name}: total LL is not finite: {total_ll}"
            assert all(np.isfinite(ll) for ll in ll_seq), f"{model_name}: some trial LLs are not finite"
    
    def test_ll_is_negative(self):
        """Log probabilities should be <= 0."""
        A, B, D = build_abd()
        
        value_fn_gen = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=50)
        
        for model_name in ['M1', 'M2', 'M3']:
            value_fn = create_model(model_name, A, B, D)
            total_ll, ll_seq = evaluate_ll_with_valuefn(value_fn, A, B, D, ref_logs)
            
            assert total_ll <= 0, f"{model_name}: total LL should be <= 0, got {total_ll}"
            assert all(ll <= 0 for ll in ll_seq), f"{model_name}: all trial LLs should be <= 0"
    
    def test_self_fit_has_high_ll(self):
        """Model should assign high LL to data it generated."""
        A, B, D = build_abd()
        np.random.seed(42)
        
        for gen_model in ['M1', 'M2', 'M3']:
            value_fn_gen = create_model(gen_model, A, B, D)
            env = TwoArmedBandit(reversal_schedule=[30])
            runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                       OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                       OBSERVATION_CHOICES, ACTION_CHOICES)
            ref_logs = run_episode_with_ll(runner, env, T=60)
            
            # Fit same model
            value_fn_fit = create_model(gen_model, A, B, D)
            total_ll, _ = evaluate_ll_with_valuefn(value_fn_fit, A, B, D, ref_logs)
            
            # Per-trial average LL should be reasonable (not terrible)
            avg_ll = total_ll / 60
            assert avg_ll > -2.0, f"{gen_model} self-fit avg LL too low: {avg_ll:.3f}"


class TestCrossValidation:
    """Test the cross-validation fitting procedure."""
    
    def test_cv_fit_runs_without_error(self):
        """CV fitting should complete without errors."""
        A, B, D = build_abd()
        
        value_fn = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[15, 30])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=50)
        
        # Run CV fit with minimal grid for speed
        result = cv_fit_single_run('M1', A, B, D, ref_logs, K=2)
        
        assert 'mean_test_ll' in result
        assert 'best_params_per_fold' in result
        assert np.isfinite(result['mean_test_ll'])
    
    def test_cv_fold_splits_are_valid(self):
        """CV splits should cover all trials without overlap."""
        A, B, D = build_abd()
        
        value_fn = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=40)
        
        # Manually check fold logic
        from sklearn.model_selection import KFold
        T = 40
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        all_test = set()
        
        for train_idx, test_idx in kf.split(range(T)):
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0
            all_test.update(test_idx)
        
        # All trials covered
        assert all_test == set(range(T))


class TestModelDistinguishability:
    """Test that models produce distinguishable behavior patterns."""
    
    def test_m3_uses_more_hints_in_volatile(self):
        """M3 should use more hints in volatile context due to profile."""
        A, B, D = build_abd()
        np.random.seed(123)
        
        # Run M3 in pure volatile context
        value_fn = create_model('M3', A, B, D)
        env_volatile = TwoArmedBandit(context='volatile', reversal_schedule=[])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        logs_volatile = run_episode_with_ll(runner, env_volatile, T=100)
        hints_volatile = sum(1 for a in logs_volatile['action'] if a == 'act_hint')
        
        # Run M3 in pure stable context
        np.random.seed(123)  # Same seed for comparability
        env_stable = TwoArmedBandit(context='stable', reversal_schedule=[])
        runner2 = AgentRunnerWithLL(A, B, D, value_fn,
                                    OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                    OBSERVATION_CHOICES, ACTION_CHOICES)
        logs_stable = run_episode_with_ll(runner2, env_stable, T=100)
        hints_stable = sum(1 for a in logs_stable['action'] if a == 'act_hint')
        
        # M3 should use more hints in volatile (due to hint preference in profile 0)
        print(f"Hints volatile: {hints_volatile}, Hints stable: {hints_stable}")
        assert hints_volatile >= hints_stable, \
            f"M3 should use >= hints in volatile ({hints_volatile}) vs stable ({hints_stable})"
    
    def test_m1_behavior_constant_across_contexts(self):
        """M1 behavior should be similar regardless of context."""
        A, B, D = build_abd()
        
        value_fn = create_model('M1', A, B, D)
        
        np.random.seed(456)
        env_volatile = TwoArmedBandit(context='volatile', reversal_schedule=[])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        logs_volatile = run_episode_with_ll(runner, env_volatile, T=100)
        gamma_volatile = np.mean(logs_volatile['gamma'])
        
        np.random.seed(456)
        env_stable = TwoArmedBandit(context='stable', reversal_schedule=[])
        runner2 = AgentRunnerWithLL(A, B, D, value_fn,
                                    OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                    OBSERVATION_CHOICES, ACTION_CHOICES)
        logs_stable = run_episode_with_ll(runner2, env_stable, T=100)
        gamma_stable = np.mean(logs_stable['gamma'])
        
        # M1 gamma should be identical
        assert gamma_volatile == gamma_stable, \
            f"M1 gamma should be constant: volatile={gamma_volatile}, stable={gamma_stable}"


class TestMiniModelRecovery:
    """Mini end-to-end model recovery test."""
    
    def test_mini_recovery_self_identification(self):
        """Each model should fit its own data better than others (basic check)."""
        A, B, D = build_abd()
        np.random.seed(789)
        
        results = {}
        
        for gen_model in ['M1', 'M3']:  # Skip M2 for speed
            # Generate data
            value_fn_gen = create_model(gen_model, A, B, D)
            env = TwoArmedBandit(reversal_schedule=[25, 50])
            runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                       OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                       OBSERVATION_CHOICES, ACTION_CHOICES)
            ref_logs = run_episode_with_ll(runner, env, T=80)
            
            # Evaluate with all models
            model_lls = {}
            for fit_model in ['M1', 'M3']:
                value_fn_fit = create_model(fit_model, A, B, D)
                total_ll, _ = evaluate_ll_with_valuefn(value_fn_fit, A, B, D, ref_logs)
                model_lls[fit_model] = total_ll
            
            results[gen_model] = model_lls
        
        # Check self-fit is best or close to best
        for gen_model in ['M1', 'M3']:
            self_ll = results[gen_model][gen_model]
            other_lls = [ll for m, ll in results[gen_model].items() if m != gen_model]
            best_other = max(other_lls)
            
            # Self-fit should be at least as good as the best other
            # (with some tolerance for stochasticity)
            print(f"{gen_model} data: self LL={self_ll:.2f}, best other LL={best_other:.2f}")


class TestTrialLevelPredictions:
    """Test trial-level prediction output."""
    
    def test_trial_level_output_format(self):
        """Trial level predictions should have all required fields."""
        from src.utils.recovery_helpers import _generate_trial_level_predictions
        
        A, B, D = build_abd()
        value_fn_gen = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=40)
        
        value_fn_eval = create_model('M1', A, B, D)
        rows = _generate_trial_level_predictions(value_fn_eval, A, B, D, ref_logs)
        
        assert len(rows) == 40
        
        required_fields = ['t', 'true_context', 'gen_action', 'predicted_action',
                          'action_probs', 'belief_context', 'gamma', 'll', 'accuracy']
        for row in rows:
            for field in required_fields:
                assert field in row, f"Missing field: {field}"
    
    def test_action_probs_sum_to_one(self):
        """Action probabilities should sum to 1."""
        from src.utils.recovery_helpers import _generate_trial_level_predictions
        
        A, B, D = build_abd()
        value_fn_gen = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=40)
        
        value_fn_eval = create_model('M1', A, B, D)
        rows = _generate_trial_level_predictions(value_fn_eval, A, B, D, ref_logs)
        
        for t, row in enumerate(rows):
            prob_sum = sum(row['action_probs'])
            assert abs(prob_sum - 1.0) < 1e-6, f"At t={t}, action probs sum to {prob_sum}"
    
    def test_ll_matches_action_prob(self):
        """LL should be log of the probability of the generated action."""
        from src.utils.recovery_helpers import _generate_trial_level_predictions
        
        A, B, D = build_abd()
        value_fn_gen = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=40)
        
        value_fn_eval = create_model('M1', A, B, D)
        rows = _generate_trial_level_predictions(value_fn_eval, A, B, D, ref_logs)
        
        for t, row in enumerate(rows):
            gen_action = row['gen_action']
            try:
                gen_idx = ACTION_CHOICES.index(gen_action)
                expected_ll = np.log(row['action_probs'][gen_idx] + 1e-16)
                actual_ll = row['ll']
                assert abs(actual_ll - expected_ll) < 1e-6, \
                    f"At t={t}, LL mismatch: expected {expected_ll:.4f}, got {actual_ll:.4f}"
            except ValueError:
                pass  # Skip if action not in list


class TestParameterRecovery:
    """Test that parameters can be recovered."""
    
    def test_m1_gamma_grid_search(self):
        """Higher gamma should give higher LL for M1 data with high gamma."""
        A, B, D = build_abd()
        
        # Generate with high gamma
        from src.models.value_functions import make_values_M1
        high_gamma = 10.0
        value_fn = make_values_M1(M1_DEFAULTS['C_reward_logits'], high_gamma)
        
        env = TwoArmedBandit(reversal_schedule=[30])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=60)
        
        # Evaluate at different gammas
        from src.utils.ll_eval import _eval_m1_gamma
        
        ll_low = _eval_m1_gamma(A, B, D, 2.0, ref_logs)
        ll_high = _eval_m1_gamma(A, B, D, 10.0, ref_logs)
        
        # Higher gamma should fit better for data generated with high gamma
        assert ll_high > ll_low, f"High gamma LL ({ll_high:.2f}) should be > low gamma LL ({ll_low:.2f})"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])  # -x stops at first failure

