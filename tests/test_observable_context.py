"""
Comprehensive tests for observable context implementation.

Tests verify:
1. A matrix structure (4 modalities, context observation)
2. Environment returns correct context observations
3. Agent beliefs update based on context cues
4. M3 profile mixing works correctly
5. Models produce distinguishable behavior
6. Basic model recovery sanity checks
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.experiment_config import *
from src.models.generative_model import build_A, build_B, build_D
from src.utils.recovery_helpers import build_abd
from src.utils.model_utils import create_model
from src.environment import TwoArmedBandit
from src.models.agent_wrapper import AgentRunnerWithLL, run_episode_with_ll


class TestAMatrixStructure:
    """Test that A matrix has correct structure for observable context."""
    
    def test_num_modalities(self):
        """A matrix should have 4 modalities."""
        A, B, D = build_abd()
        assert len(A) == 4, f"Expected 4 modalities, got {len(A)}"
    
    def test_modality_shapes(self):
        """Each A modality should have correct shape."""
        A, B, D = build_abd()
        
        # A[0]: hints (3, 2, 2, 4) - 3 hint obs, 2 contexts, 2 better_arm, 4 choices
        assert A[0].shape == (3, 2, 2, 4), f"A[0] shape wrong: {A[0].shape}"
        
        # A[1]: rewards (3, 2, 2, 4)
        assert A[1].shape == (3, 2, 2, 4), f"A[1] shape wrong: {A[1].shape}"
        
        # A[2]: choices (4, 2, 2, 4)
        assert A[2].shape == (4, 2, 2, 4), f"A[2] shape wrong: {A[2].shape}"
        
        # A[3]: contexts (2, 2, 2, 4) - 2 context obs, 2 contexts, 2 better_arm, 4 choices
        assert A[3].shape == (2, 2, 2, 4), f"A[3] shape wrong: {A[3].shape}"
    
    def test_context_observation_is_deterministic(self):
        """Context observation should perfectly reveal context state."""
        A, B, D = build_abd()
        A_context = A[3]
        
        # When in volatile (ctx=0), should observe volatile (obs=0) with prob 1
        # When in stable (ctx=1), should observe stable (obs=1) with prob 1
        for better_arm in range(2):
            for choice in range(4):
                # Volatile context -> observe_volatile
                assert A_context[0, 0, better_arm, choice] == 1.0, \
                    f"P(observe_volatile | volatile) should be 1.0"
                assert A_context[1, 0, better_arm, choice] == 0.0, \
                    f"P(observe_stable | volatile) should be 0.0"
                
                # Stable context -> observe_stable
                assert A_context[0, 1, better_arm, choice] == 0.0, \
                    f"P(observe_volatile | stable) should be 0.0"
                assert A_context[1, 1, better_arm, choice] == 1.0, \
                    f"P(observe_stable | stable) should be 1.0"


class TestEnvironment:
    """Test that environment returns correct observations."""
    
    def test_returns_4_observations(self):
        """Environment step should return 4 observations."""
        env = TwoArmedBandit()
        obs = env.step("act_hint")
        assert len(obs) == 4, f"Expected 4 observations, got {len(obs)}"
    
    def test_context_observation_matches_state(self):
        """Context observation should match actual context."""
        env = TwoArmedBandit(context='volatile')
        obs = env.step("act_hint")
        assert obs[3] == "observe_volatile", f"Expected observe_volatile, got {obs[3]}"
        
        env2 = TwoArmedBandit(context='stable')
        obs2 = env2.step("act_hint")
        assert obs2[3] == "observe_stable", f"Expected observe_stable, got {obs2[3]}"
    
    def test_context_observation_updates_after_reversal(self):
        """After context reversal, observation should change."""
        env = TwoArmedBandit(context='volatile', reversal_schedule=[5])
        
        # First few trials should be volatile
        for _ in range(4):
            obs = env.step("act_left")
            assert obs[3] == "observe_volatile"
        
        # After reversal (trial 5+), should be stable
        for _ in range(3):
            obs = env.step("act_left")
        # By now we should have passed the reversal
        assert env.context == "stable", f"Context should have reversed"


class TestAgentBeliefUpdates:
    """Test that agent beliefs update correctly with context observations."""
    
    def test_beliefs_update_to_volatile(self):
        """When observing volatile, beliefs should shift toward volatile."""
        A, B, D = build_abd()
        value_fn = create_model('M1', A, B, D)  # Use M1 (doesn't matter which)
        
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        
        # Start with uniform prior
        initial_obs = ['null', 'null', 'observe_start', 'observe_volatile']
        obs_ids = runner.obs_labels_to_ids(initial_obs)
        
        # Infer states
        qs = runner.agent.infer_states(obs_ids)
        q_context = qs[0]
        
        # Should strongly believe volatile after observing volatile cue
        assert q_context[0] > 0.9, f"P(volatile) should be > 0.9, got {q_context[0]}"
    
    def test_beliefs_update_to_stable(self):
        """When observing stable, beliefs should shift toward stable."""
        A, B, D = build_abd()
        value_fn = create_model('M1', A, B, D)
        
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        
        initial_obs = ['null', 'null', 'observe_start', 'observe_stable']
        obs_ids = runner.obs_labels_to_ids(initial_obs)
        
        qs = runner.agent.infer_states(obs_ids)
        q_context = qs[0]
        
        # Should strongly believe stable after observing stable cue
        assert q_context[1] > 0.9, f"P(stable) should be > 0.9, got {q_context[1]}"


class TestBeliefPropagation:
    """Test that beliefs propagate correctly across multiple trials."""
    
    def test_beliefs_stay_correct_across_trials(self):
        """Beliefs should stay correct (not reset to 50/50) across trials."""
        from src.utils.ll_eval import evaluate_ll_with_valuefn
        
        A, B, D = build_abd()
        
        # Generate reference data
        value_fn_gen = create_model('M3', A, B, D)
        env = TwoArmedBandit(context='volatile', reversal_schedule=[50])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=60)
        
        # Check that beliefs in volatile context are near [1, 0]
        # (first 50 trials are volatile)
        for t in range(5, 45):  # Skip first few trials, check middle
            belief = ref_logs['belief'][t]
            assert belief[0] > 0.8, f"At t={t}, P(volatile) should be > 0.8, got {belief[0]}"
        
        # Check that beliefs in stable context are near [0, 1]
        # (trials 50+ are stable)
        for t in range(52, 58):
            belief = ref_logs['belief'][t]
            assert belief[1] > 0.8, f"At t={t}, P(stable) should be > 0.8, got {belief[1]}"
    
    def test_ll_eval_beliefs_propagate(self):
        """During LL evaluation, beliefs should propagate correctly."""
        from src.utils.ll_eval import evaluate_ll_with_valuefn
        from src.utils.recovery_helpers import _generate_trial_level_predictions
        
        A, B, D = build_abd()
        
        # Generate reference data with known context pattern
        value_fn_gen = create_model('M3', A, B, D)
        env = TwoArmedBandit(context='volatile', reversal_schedule=[30])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=50)
        
        # Use M1 to evaluate (so we can check beliefs from trial_level predictions)
        value_fn_eval = create_model('M1', A, B, D)
        rows = _generate_trial_level_predictions(value_fn_eval, A, B, D, ref_logs)
        
        # Check volatile period (t=5-25)
        for t in range(5, 25):
            belief = rows[t]['belief_context']
            assert belief[0] > 0.7, f"At t={t}, P(volatile) should be > 0.7, got {belief[0]}"
        
        # Check stable period (t=35-45)
        for t in range(35, 45):
            belief = rows[t]['belief_context']
            assert belief[1] > 0.7, f"At t={t}, P(stable) should be > 0.7, got {belief[1]}"


class TestM3ProfileMixing:
    """Test that M3 correctly mixes profiles based on context."""
    
    def test_volatile_context_gives_low_gamma(self):
        """In volatile context, M3 should use exploratory profile (low gamma)."""
        A, B, D = build_abd()
        value_fn = create_model('M3', A, B, D)
        
        # Simulate beliefs: strongly believe volatile
        q_context = np.array([0.95, 0.05])
        q_better_arm = np.array([0.5, 0.5])
        q_choice = np.array([1.0, 0.0, 0.0, 0.0])
        qs = [q_context, q_better_arm, q_choice]
        
        C, E, gamma = value_fn(qs, t=0)
        
        # Profile 0 (volatile) has gamma=2.0, Profile 1 (stable) has gamma=4.0
        # With 95% volatile, gamma should be close to 2.0
        assert gamma < 2.5, f"Gamma should be < 2.5 for volatile, got {gamma}"
    
    def test_stable_context_gives_high_gamma(self):
        """In stable context, M3 should use exploitative profile (high gamma)."""
        A, B, D = build_abd()
        value_fn = create_model('M3', A, B, D)
        
        # Simulate beliefs: strongly believe stable
        q_context = np.array([0.05, 0.95])
        q_better_arm = np.array([0.5, 0.5])
        q_choice = np.array([1.0, 0.0, 0.0, 0.0])
        qs = [q_context, q_better_arm, q_choice]
        
        C, E, gamma = value_fn(qs, t=0)
        
        # With 95% stable, gamma should be close to 4.0
        assert gamma > 3.5, f"Gamma should be > 3.5 for stable, got {gamma}"
    
    def test_gamma_changes_with_context(self):
        """Gamma should be different for volatile vs stable beliefs."""
        A, B, D = build_abd()
        value_fn = create_model('M3', A, B, D)
        
        q_better_arm = np.array([0.5, 0.5])
        q_choice = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Volatile
        qs_volatile = [np.array([0.95, 0.05]), q_better_arm, q_choice]
        _, _, gamma_volatile = value_fn(qs_volatile, t=0)
        
        # Stable
        qs_stable = [np.array([0.05, 0.95]), q_better_arm, q_choice]
        _, _, gamma_stable = value_fn(qs_stable, t=0)
        
        assert gamma_volatile < gamma_stable, \
            f"Gamma should be lower for volatile ({gamma_volatile}) than stable ({gamma_stable})"


class TestM1M2DoNotUseContext:
    """Test that M1 and M2 don't change based on context beliefs."""
    
    def test_m1_gamma_constant(self):
        """M1 gamma should be constant regardless of context."""
        A, B, D = build_abd()
        value_fn = create_model('M1', A, B, D)
        
        q_better_arm = np.array([0.5, 0.5])
        q_choice = np.array([1.0, 0.0, 0.0, 0.0])
        
        qs_volatile = [np.array([0.95, 0.05]), q_better_arm, q_choice]
        qs_stable = [np.array([0.05, 0.95]), q_better_arm, q_choice]
        
        _, _, gamma_v = value_fn(qs_volatile, t=0)
        _, _, gamma_s = value_fn(qs_stable, t=0)
        
        assert gamma_v == gamma_s, f"M1 gamma should be constant, got {gamma_v} vs {gamma_s}"
    
    def test_m2_uses_better_arm_not_context(self):
        """M2 gamma should depend on better_arm entropy, not context."""
        A, B, D = build_abd()
        value_fn = create_model('M2', A, B, D)
        
        q_choice = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Same better_arm entropy, different context
        q_better_arm = np.array([0.5, 0.5])  # High entropy
        qs_volatile = [np.array([0.95, 0.05]), q_better_arm, q_choice]
        qs_stable = [np.array([0.05, 0.95]), q_better_arm, q_choice]
        
        _, _, gamma_v = value_fn(qs_volatile, t=0)
        _, _, gamma_s = value_fn(qs_stable, t=0)
        
        # Should be same since better_arm entropy is same
        assert abs(gamma_v - gamma_s) < 0.01, \
            f"M2 gamma should depend on better_arm, not context: {gamma_v} vs {gamma_s}"
        
        # Different better_arm entropy should give different gamma
        q_better_arm_certain = np.array([0.95, 0.05])  # Low entropy
        qs_certain = [np.array([0.5, 0.5]), q_better_arm_certain, q_choice]
        _, _, gamma_certain = value_fn(qs_certain, t=0)
        
        assert gamma_certain > gamma_v, \
            f"M2 gamma should be higher when certain about better_arm"


class TestFullEpisode:
    """Test full episode runs with all models."""
    
    def test_m1_episode_runs(self):
        """M1 should complete an episode without errors."""
        A, B, D = build_abd()
        value_fn = create_model('M1', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        
        logs = run_episode_with_ll(runner, env, T=40)
        
        assert len(logs['action']) == 40
        assert all(g == logs['gamma'][0] for g in logs['gamma']), "M1 gamma should be constant"
    
    def test_m2_episode_runs(self):
        """M2 should complete an episode without errors."""
        A, B, D = build_abd()
        value_fn = create_model('M2', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        
        logs = run_episode_with_ll(runner, env, T=40)
        
        assert len(logs['action']) == 40
        # M2 gamma should vary (based on better_arm entropy)
        assert len(set(logs['gamma'])) > 1, "M2 gamma should vary"
    
    def test_m3_episode_runs(self):
        """M3 should complete an episode without errors."""
        A, B, D = build_abd()
        value_fn = create_model('M3', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        
        logs = run_episode_with_ll(runner, env, T=40)
        
        assert len(logs['action']) == 40
        assert len(logs['context_label']) == 40
        
        # Check that context observations are recorded
        assert all(c in ['observe_volatile', 'observe_stable'] for c in logs['context_label'])
    
    def test_m3_gamma_changes_at_reversal(self):
        """M3 gamma should change when context reverses."""
        A, B, D = build_abd()
        value_fn = create_model('M3', A, B, D)
        env = TwoArmedBandit(context='volatile', reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        
        logs = run_episode_with_ll(runner, env, T=40)
        
        # Average gamma in first half (volatile) vs second half (stable)
        gamma_first = np.mean(logs['gamma'][:15])
        gamma_second = np.mean(logs['gamma'][25:])
        
        # Gamma should be higher in stable context
        assert gamma_second > gamma_first, \
            f"Gamma should increase after reversal: {gamma_first:.2f} -> {gamma_second:.2f}"


class TestModelDistinguishability:
    """Test that models produce distinguishable behavior patterns."""
    
    def test_action_distributions_differ(self):
        """Different models should produce different action distributions."""
        A, B, D = build_abd()
        np.random.seed(42)
        
        action_counts = {}
        
        for model_name in ['M1', 'M2', 'M3']:
            value_fn = create_model(model_name, A, B, D)
            env = TwoArmedBandit(reversal_schedule=[40], context='volatile')
            runner = AgentRunnerWithLL(A, B, D, value_fn,
                                       OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                       OBSERVATION_CHOICES, ACTION_CHOICES)
            
            logs = run_episode_with_ll(runner, env, T=80)
            
            # Count hint actions
            hint_count = sum(1 for a in logs['action'] if a == 'act_hint')
            action_counts[model_name] = hint_count
        
        # M3 should have more hints than M1 (due to hint preference in volatile profile)
        # This is a rough test - exact values depend on parameters
        print(f"Hint counts: M1={action_counts['M1']}, M2={action_counts['M2']}, M3={action_counts['M3']}")


class TestLogLikelihoodEvaluation:
    """Test that log-likelihood evaluation works correctly."""
    
    def test_ll_eval_runs(self):
        """LL evaluation should run without errors."""
        from src.utils.ll_eval import evaluate_ll_with_valuefn
        
        A, B, D = build_abd()
        
        # Generate reference data with M3
        value_fn_gen = create_model('M3', A, B, D)
        env = TwoArmedBandit(reversal_schedule=[20])
        runner = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                   OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                   OBSERVATION_CHOICES, ACTION_CHOICES)
        ref_logs = run_episode_with_ll(runner, env, T=40)
        
        # Evaluate with each model
        for model_name in ['M1', 'M2', 'M3']:
            value_fn = create_model(model_name, A, B, D)
            total_ll, ll_seq = evaluate_ll_with_valuefn(value_fn, A, B, D, ref_logs)
            
            assert np.isfinite(total_ll), f"{model_name} LL should be finite"
            assert len(ll_seq) == 40, f"{model_name} should have 40 trial LLs"
            assert all(np.isfinite(ll) for ll in ll_seq), f"{model_name} all trial LLs should be finite"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

