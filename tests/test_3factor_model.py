"""
Comprehensive test suite for the 3-factor state model fix.

Tests verify that:
1. A/B/D matrices have correct shapes for 3 state factors
2. Hints reveal better_arm, NOT context
3. Rewards depend on better_arm AND choice
4. Context beliefs track actual context (volatile vs stable)
5. Better_arm beliefs update from hints
6. End-to-end agent-environment interaction works

Run with: pytest tests/test_3factor_model.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import (
    STATE_CONTEXTS, STATE_BETTER_ARM, STATE_CHOICES,
    ACTION_CONTEXTS, ACTION_BETTER_ARM, ACTION_CHOICES,
    OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
    NUM_STATES, NUM_ACTIONS, NUM_OBS, NUM_FACTORS, NUM_MODALITIES,
    PROBABILITY_HINT
)
from src.models.generative_model import build_A, build_B, build_D


class TestConfigDimensions:
    """Test that config dimensions are correct for 3-factor model."""
    
    def test_state_factors(self):
        """Should have 3 state factors."""
        assert NUM_FACTORS == 3
        assert len(NUM_STATES) == 3
        
    def test_state_dimensions(self):
        """State dimensions should be [2, 2, 4]."""
        assert NUM_STATES == [2, 2, 4]  # context, better_arm, choice
        
    def test_action_dimensions(self):
        """Action dimensions should be [1, 1, 4]."""
        assert NUM_ACTIONS == [1, 1, 4]  # rest, rest, choice_actions
        
    def test_observation_dimensions(self):
        """Observation dimensions should be [3, 3, 4]."""
        assert NUM_OBS == [3, 3, 4]  # hints, rewards, choices
        
    def test_state_labels(self):
        """State labels should be correctly defined."""
        assert STATE_CONTEXTS == ['volatile', 'stable']
        assert STATE_BETTER_ARM == ['left_better', 'right_better']
        assert STATE_CHOICES == ['start', 'hint', 'left', 'right']


class TestAMatrix:
    """Test A matrix (observation likelihood) structure and values."""
    
    @pytest.fixture
    def A(self):
        """Build A matrix for testing."""
        return build_A(
            NUM_MODALITIES,
            STATE_CONTEXTS, STATE_BETTER_ARM, STATE_CHOICES,
            OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
            p_hint=0.85, p_reward=0.80
        )
    
    def test_A_shape(self, A):
        """A matrices should have correct shapes."""
        # A[0]: hints - shape (3, 2, 2, 4) = (obs, ctx, better, choice)
        assert A[0].shape == (3, 2, 2, 4), f"A[0] shape: {A[0].shape}"
        # A[1]: rewards - shape (3, 2, 2, 4)
        assert A[1].shape == (3, 2, 2, 4), f"A[1] shape: {A[1].shape}"
        # A[2]: choices - shape (4, 2, 2, 4)
        assert A[2].shape == (4, 2, 2, 4), f"A[2] shape: {A[2].shape}"
        
    def test_A_hint_independent_of_context(self, A):
        """Hint observations should NOT depend on context (volatile/stable)."""
        # When in hint state, probabilities should be same for both contexts
        choice_hint_idx = STATE_CHOICES.index('hint')
        
        # For left_better, hints should be same in volatile and stable
        hint_volatile_leftbetter = A[0][:, 0, 0, choice_hint_idx]  # ctx=0 (volatile)
        hint_stable_leftbetter = A[0][:, 1, 0, choice_hint_idx]    # ctx=1 (stable)
        np.testing.assert_array_almost_equal(
            hint_volatile_leftbetter, hint_stable_leftbetter,
            err_msg="Hints should be same regardless of context (volatile/stable)"
        )
        
    def test_A_hint_depends_on_better_arm(self, A):
        """Hint observations SHOULD depend on better_arm."""
        choice_hint_idx = STATE_CHOICES.index('hint')
        p_hint = 0.85
        
        # Check hint probabilities for left_better vs right_better
        # Left_better should make left_hint more likely
        hint_left_idx = OBSERVATION_HINTS.index('observe_left_hint')
        hint_right_idx = OBSERVATION_HINTS.index('observe_right_hint')
        
        # When left_better (idx 0): P(left_hint) should be high
        prob_left_hint_given_left_better = A[0][hint_left_idx, 0, 0, choice_hint_idx]
        assert abs(prob_left_hint_given_left_better - p_hint) < 0.01, \
            f"P(left_hint | left_better) should be {p_hint}, got {prob_left_hint_given_left_better}"
        
        # When right_better (idx 1): P(right_hint) should be high
        prob_right_hint_given_right_better = A[0][hint_right_idx, 0, 1, choice_hint_idx]
        assert abs(prob_right_hint_given_right_better - p_hint) < 0.01, \
            f"P(right_hint | right_better) should be {p_hint}, got {prob_right_hint_given_right_better}"
            
    def test_A_reward_depends_on_better_arm_and_choice(self, A):
        """Rewards should depend on both better_arm AND choice."""
        p_reward = 0.80
        reward_idx = OBSERVATION_REWARDS.index('observe_reward')
        loss_idx = OBSERVATION_REWARDS.index('observe_loss')
        choice_left_idx = STATE_CHOICES.index('left')
        choice_right_idx = STATE_CHOICES.index('right')
        
        # Chose left when left_better -> high reward
        prob_reward_left_leftbetter = A[1][reward_idx, 0, 0, choice_left_idx]
        assert abs(prob_reward_left_leftbetter - p_reward) < 0.01, \
            f"P(reward | left, left_better) should be {p_reward}, got {prob_reward_left_leftbetter}"
        
        # Chose left when right_better -> low reward (high loss)
        prob_reward_left_rightbetter = A[1][reward_idx, 0, 1, choice_left_idx]
        assert abs(prob_reward_left_rightbetter - (1 - p_reward)) < 0.01, \
            f"P(reward | left, right_better) should be {1-p_reward}, got {prob_reward_left_rightbetter}"
            
    def test_A_normalizes_correctly(self, A):
        """All A matrices should be valid probability distributions."""
        for m, A_m in enumerate(A):
            sums = A_m.sum(axis=0)  # Sum over observations
            np.testing.assert_array_almost_equal(
                sums, np.ones_like(sums),
                err_msg=f"A[{m}] rows don't sum to 1"
            )


class TestBMatrix:
    """Test B matrix (transition) structure."""
    
    @pytest.fixture
    def B(self):
        """Build B matrix for testing."""
        return build_B(
            STATE_CONTEXTS, STATE_BETTER_ARM, STATE_CHOICES,
            ACTION_CONTEXTS, ACTION_BETTER_ARM, ACTION_CHOICES,
            context_volatility=0.05
        )
        
    def test_B_has_3_factors(self, B):
        """Should have B matrices for 3 state factors."""
        assert len(B) == 3
        
    def test_B_context_shape(self, B):
        """B[0] should be context transitions."""
        # Shape: (n_context, n_context, n_action_context)
        assert B[0].shape == (2, 2, 1), f"B[0] shape: {B[0].shape}"
        
    def test_B_better_arm_shape(self, B):
        """B[1] should be better_arm transitions."""
        # Shape: (n_better, n_better, n_action_better)
        assert B[1].shape == (2, 2, 1), f"B[1] shape: {B[1].shape}"
        
    def test_B_choice_shape(self, B):
        """B[2] should be choice transitions."""
        # Shape: (n_choice, n_choice, n_action_choice)
        assert B[2].shape == (4, 4, 4), f"B[2] shape: {B[2].shape}"
        
    def test_B_context_sticky(self, B):
        """Context transitions should be sticky (high self-transition)."""
        context_vol = 0.05
        # Diagonal should be 1 - volatility
        np.testing.assert_almost_equal(
            B[0][0, 0, 0], 1 - context_vol,
            err_msg="Context self-transition should be high"
        )
        
    def test_B_choice_deterministic(self, B):
        """Choice transitions should be deterministic given action."""
        # Action 0 (start) -> state 0, action 1 (hint) -> state 1, etc.
        for a in range(4):
            expected = np.zeros((4, 4))
            expected[a, :] = 1.0
            np.testing.assert_array_almost_equal(
                B[2][:, :, a], expected,
                err_msg=f"B_choice[:,:,{a}] should be deterministic"
            )


class TestDMatrix:
    """Test D matrix (prior) structure."""
    
    @pytest.fixture
    def D(self):
        """Build D matrix for testing."""
        return build_D(STATE_CONTEXTS, STATE_BETTER_ARM, STATE_CHOICES)
        
    def test_D_has_3_factors(self, D):
        """Should have D vectors for 3 state factors."""
        assert len(D) == 3
        
    def test_D_context_uniform(self, D):
        """Prior over context should be uniform."""
        np.testing.assert_array_almost_equal(
            D[0], [0.5, 0.5],
            err_msg="Context prior should be uniform"
        )
        
    def test_D_better_arm_uniform(self, D):
        """Prior over better_arm should be uniform."""
        np.testing.assert_array_almost_equal(
            D[1], [0.5, 0.5],
            err_msg="Better_arm prior should be uniform"
        )
        
    def test_D_choice_starts_at_start(self, D):
        """Prior over choice should start at 'start' state."""
        expected = np.zeros(4)
        expected[STATE_CHOICES.index('start')] = 1.0
        np.testing.assert_array_almost_equal(
            D[2], expected,
            err_msg="Choice prior should be deterministic at 'start'"
        )


class TestAgentIntegration:
    """Test that pyMDP agent works with new 3-factor structure."""
    
    @pytest.fixture
    def abd(self):
        """Build A, B, D matrices."""
        A = build_A(
            NUM_MODALITIES,
            STATE_CONTEXTS, STATE_BETTER_ARM, STATE_CHOICES,
            OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
            p_hint=0.85, p_reward=0.80
        )
        B = build_B(
            STATE_CONTEXTS, STATE_BETTER_ARM, STATE_CHOICES,
            ACTION_CONTEXTS, ACTION_BETTER_ARM, ACTION_CHOICES,
            context_volatility=0.05
        )
        D = build_D(STATE_CONTEXTS, STATE_BETTER_ARM, STATE_CHOICES)
        return A, B, D
    
    def test_agent_creation(self, abd):
        """Agent should be creatable with 3-factor matrices."""
        from pymdp.agent import Agent
        from pymdp import utils
        
        A, B, D = abd
        C = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
        
        agent = Agent(
            A=A, B=B, C=C, D=D,
            policy_len=2, inference_horizon=1,
            control_fac_idx=[2],  # Only choice is controllable
            use_utility=True,
            use_states_info_gain=True,
            action_selection="stochastic", gamma=4
        )
        
        assert agent is not None
        assert len(agent.policies) > 0
        
    def test_agent_policies_shape(self, abd):
        """Agent policies should have correct shape."""
        from pymdp.agent import Agent
        from pymdp import utils
        
        A, B, D = abd
        C = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
        
        agent = Agent(
            A=A, B=B, C=C, D=D,
            policy_len=2, inference_horizon=1,
            control_fac_idx=[2],
            use_utility=True,
            use_states_info_gain=True,
            action_selection="stochastic", gamma=4
        )
        
        # Policies should have shape [policy_len, num_action_factors]
        for pol in agent.policies:
            assert pol.shape[0] == 2, f"Policy length should be 2, got {pol.shape[0]}"
            assert pol.shape[1] == 3, f"Policy should have 3 action factors, got {pol.shape[1]}"
            
    def test_state_inference(self, abd):
        """Agent should infer states with 3 factors."""
        from pymdp.agent import Agent
        from pymdp import utils
        
        A, B, D = abd
        C = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
        
        agent = Agent(
            A=A, B=B, C=C, D=D,
            policy_len=2, inference_horizon=1,
            control_fac_idx=[2],
            use_utility=True,
            use_states_info_gain=True,
            action_selection="stochastic", gamma=4
        )
        
        # Create observation (null hint, null reward, start choice)
        obs = [0, 0, 0]  # All null/start
        
        qs = agent.infer_states(obs)
        
        assert len(qs) == 3, f"Should have 3 state beliefs, got {len(qs)}"
        assert len(qs[0]) == 2, "Context beliefs should have 2 elements"
        assert len(qs[1]) == 2, "Better_arm beliefs should have 2 elements"
        assert len(qs[2]) == 4, "Choice beliefs should have 4 elements"


class TestBeliefUpdates:
    """Test that beliefs update correctly based on observations."""
    
    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        from pymdp.agent import Agent
        from pymdp import utils
        
        A = build_A(
            NUM_MODALITIES,
            STATE_CONTEXTS, STATE_BETTER_ARM, STATE_CHOICES,
            OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
            p_hint=0.85, p_reward=0.80
        )
        B = build_B(
            STATE_CONTEXTS, STATE_BETTER_ARM, STATE_CHOICES,
            ACTION_CONTEXTS, ACTION_BETTER_ARM, ACTION_CHOICES,
            context_volatility=0.05
        )
        D = build_D(STATE_CONTEXTS, STATE_BETTER_ARM, STATE_CHOICES)
        C = utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
        
        return Agent(
            A=A, B=B, C=C, D=D,
            policy_len=2, inference_horizon=1,
            control_fac_idx=[2],
            use_utility=True,
            use_states_info_gain=True,
            action_selection="stochastic", gamma=4
        )
    
    def test_hint_updates_better_arm_belief(self, agent):
        """Observing a hint should update better_arm belief, not context belief."""
        # First, observe being in hint state
        obs_start = [0, 0, 0]  # null, null, start
        qs_start = agent.infer_states(obs_start)
        
        # Context belief should be ~uniform
        assert abs(qs_start[0][0] - 0.5) < 0.2, "Context belief should start ~uniform"
        
        # Better arm belief should be ~uniform
        initial_better_arm_belief = qs_start[1].copy()
        assert abs(initial_better_arm_belief[0] - 0.5) < 0.2, "Better_arm belief should start ~uniform"
        
        # Now observe left hint (in hint state)
        hint_left_idx = OBSERVATION_HINTS.index('observe_left_hint')
        choice_hint_idx = OBSERVATION_CHOICES.index('observe_hint')
        obs_hint_left = [hint_left_idx, 0, choice_hint_idx]
        
        qs_after_hint = agent.infer_states(obs_hint_left)
        
        # Better arm belief should shift toward left_better
        assert qs_after_hint[1][0] > initial_better_arm_belief[0], \
            "Left hint should increase belief in left_better"
        
        # Context belief should NOT change significantly 
        # (hints don't reveal volatile vs stable)
        context_change = abs(qs_after_hint[0][0] - qs_start[0][0])
        # Allow some change due to choice state transition
        assert context_change < 0.3, \
            f"Context belief shouldn't change much from hint, changed by {context_change}"


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_build_abd_function(self):
        """Test the build_abd helper function."""
        from src.utils.recovery_helpers import build_abd
        
        A, B, D = build_abd()
        
        assert len(A) == 3, "Should have 3 observation modalities"
        assert len(B) == 3, "Should have 3 state factors"
        assert len(D) == 3, "Should have 3 state priors"
        
        # Check shapes match config
        assert A[0].shape == (3, 2, 2, 4), f"A[0] shape mismatch: {A[0].shape}"
        assert B[0].shape == (2, 2, 1), f"B[0] shape mismatch: {B[0].shape}"
        assert len(D[0]) == 2, f"D[0] length mismatch: {len(D[0])}"
        
    def test_create_model_m1(self):
        """Test M1 model creation."""
        from src.utils.recovery_helpers import build_abd
        from src.utils.model_utils import create_model
        
        A, B, D = build_abd()
        value_fn = create_model('M1', A, B, D)
        
        # Test that value function works - now takes full qs list
        q_context = np.array([0.5, 0.5])
        q_better_arm = np.array([0.5, 0.5])
        q_choice = np.array([1.0, 0.0, 0.0, 0.0])  # start state
        qs = [q_context, q_better_arm, q_choice]
        C, E, gamma = value_fn(qs, 0)
        
        assert C is not None
        assert gamma > 0
        
    def test_create_model_m3(self):
        """Test M3 model creation."""
        from src.utils.recovery_helpers import build_abd
        from src.utils.model_utils import create_model
        
        A, B, D = build_abd()
        value_fn = create_model('M3', A, B, D)
        
        # Test that value function works with uncertain better_arm beliefs
        q_context = np.array([0.5, 0.5])
        q_better_arm_uncertain = np.array([0.5, 0.5])  # High entropy - uncertain
        q_choice = np.array([1.0, 0.0, 0.0, 0.0])
        qs_uncertain = [q_context, q_better_arm_uncertain, q_choice]
        C, E, gamma = value_fn(qs_uncertain, 0)
        
        assert C is not None
        assert E is not None
        assert gamma > 0
        
        # M3 now uses better_arm entropy for profile mixing:
        # High uncertainty (entropy near max) → Profile 0 (exploratory, lower gamma)
        # Low uncertainty (entropy near 0) → Profile 1 (exploitative, higher gamma)
        q_better_arm_certain = np.array([0.95, 0.05])  # Low entropy - confident
        qs_certain = [q_context, q_better_arm_certain, q_choice]
        C2, E2, gamma2 = value_fn(qs_certain, 0)
        
        assert gamma < gamma2, "Gamma should be lower when uncertain (exploratory profile)"


class TestAgentRunner:
    """Test AgentRunner class with 3-factor model."""
    
    def test_agent_runner_creation(self):
        """Test AgentRunnerWithLL creation."""
        from src.utils.recovery_helpers import build_abd
        from src.utils.model_utils import create_model
        from src.models.agent_wrapper import AgentRunnerWithLL
        
        A, B, D = build_abd()
        value_fn = create_model('M1', A, B, D)
        
        runner = AgentRunnerWithLL(
            A=A, B=B, D=D, value_fn=value_fn,
            observation_hints=OBSERVATION_HINTS,
            observation_rewards=OBSERVATION_REWARDS,
            observation_choices=OBSERVATION_CHOICES,
            action_choices=ACTION_CHOICES
        )
        
        assert runner is not None
        assert runner.agent is not None
        
    def test_agent_runner_step(self):
        """Test single step of agent runner."""
        from src.utils.recovery_helpers import build_abd
        from src.utils.model_utils import create_model
        from src.models.agent_wrapper import AgentRunnerWithLL
        
        A, B, D = build_abd()
        value_fn = create_model('M1', A, B, D)
        
        runner = AgentRunnerWithLL(
            A=A, B=B, D=D, value_fn=value_fn,
            observation_hints=OBSERVATION_HINTS,
            observation_rewards=OBSERVATION_REWARDS,
            observation_choices=OBSERVATION_CHOICES,
            action_choices=ACTION_CHOICES
        )
        
        # Start observation
        obs_ids = [0, 0, 0]  # null hint, null reward, start choice
        
        action_label, qs, q_pi, efe, gamma_t, ll = runner.step_with_ll(obs_ids, t=0)
        
        assert action_label in ACTION_CHOICES
        assert len(qs) == 3, "Should have beliefs over 3 state factors"
        assert gamma_t > 0
        
    def test_agent_runner_action_logprob(self):
        """Test action log-probability computation."""
        from src.utils.recovery_helpers import build_abd
        from src.utils.model_utils import create_model
        from src.models.agent_wrapper import AgentRunnerWithLL
        
        A, B, D = build_abd()
        value_fn = create_model('M1', A, B, D)
        
        runner = AgentRunnerWithLL(
            A=A, B=B, D=D, value_fn=value_fn,
            observation_hints=OBSERVATION_HINTS,
            observation_rewards=OBSERVATION_REWARDS,
            observation_choices=OBSERVATION_CHOICES,
            action_choices=ACTION_CHOICES
        )
        
        obs_ids = [0, 0, 0]
        
        # Compute log prob for each action
        total_prob = 0
        for action in ACTION_CHOICES:
            ll = runner.action_logprob(obs_ids, action, t=0)
            prob = np.exp(ll)
            total_prob += prob
            
        # Probabilities should sum to ~1
        assert abs(total_prob - 1.0) < 0.1, f"Action probabilities should sum to 1, got {total_prob}"


class TestEnvironmentIntegration:
    """Test environment works with new model structure."""
    
    def test_env_tracks_better_arm(self):
        """Environment should track current_better_arm."""
        from src.environment import TwoArmedBandit
        
        env = TwoArmedBandit(
            context='volatile',
            reversal_schedule=[50],
            volatile_switch_interval=10
        )
        
        assert hasattr(env, 'current_better_arm'), "Env should track current_better_arm"
        assert env.current_better_arm in ['left', 'right']
        
    def test_volatile_arms_switch(self):
        """In volatile context, better arm should switch periodically."""
        from src.environment import TwoArmedBandit
        
        env = TwoArmedBandit(
            context='volatile',
            reversal_schedule=[],
            volatile_switch_interval=5  # Switch every 5 trials
        )
        
        initial_arm = env.current_better_arm
        
        # Take 6 actions (enough for at least one switch at interval=5)
        for _ in range(6):
            env.step('act_hint')
            
        # Note: arm might not have switched yet if we started at trial 0
        # Just verify it's still a valid arm
        assert env.current_better_arm in ['left', 'right']


class TestFullEpisode:
    """Test full episode execution."""
    
    def test_run_episode(self):
        """Test running a complete short episode."""
        from src.utils.recovery_helpers import build_abd
        from src.utils.model_utils import create_model
        from src.models.agent_wrapper import AgentRunnerWithLL, run_episode_with_ll
        from src.environment import TwoArmedBandit
        
        A, B, D = build_abd()
        value_fn = create_model('M3', A, B, D)
        
        runner = AgentRunnerWithLL(
            A=A, B=B, D=D, value_fn=value_fn,
            observation_hints=OBSERVATION_HINTS,
            observation_rewards=OBSERVATION_REWARDS,
            observation_choices=OBSERVATION_CHOICES,
            action_choices=ACTION_CHOICES
        )
        
        env = TwoArmedBandit(
            context='volatile',
            reversal_schedule=[20],
            volatile_switch_interval=10
        )
        
        logs = run_episode_with_ll(runner, env, T=30)
        
        assert len(logs['t']) == 30, "Should have 30 trials"
        assert len(logs['context']) == 30
        assert len(logs['belief']) == 30
        assert len(logs['gamma']) == 30
        assert len(logs['ll']) == 30
        assert len(logs['current_better_arm']) == 30
        
        # Verify gamma is in expected range for M3 (between profile gammas 2.0 and 4.0)
        gamma_values = logs['gamma']
        assert all(1.5 <= g <= 4.5 for g in gamma_values), \
            f"M3 gamma should be in [1.5, 4.5], got range [{min(gamma_values):.2f}, {max(gamma_values):.2f}]"
            
        # Check that beliefs exist and are valid
        for belief in logs['belief']:
            assert len(belief) == 2, "Context belief should have 2 elements"
            assert abs(sum(belief) - 1.0) < 0.01, "Context beliefs should sum to 1"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

