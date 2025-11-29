"""
Active inference models for the changepoint behavioral task.

This module implements proper active inference agents that:
- Track beliefs over helicopter position (discretized state space)
- Generate bucket position predictions from beliefs
- Use profile-based precision control (M3) for adaptive strategies
- Evaluate on prediction log-likelihood (matching RL baselines)
"""

from .state_space import (
    PositionStateSpace,
    VolatilityStateSpace,
    create_state_space,
)
from .generative_model import (
    build_generative_model,
    build_A,
    build_B,
    build_D,
)
from .value_functions import (
    make_value_fn,
    make_values_M1,
    make_values_M2,
    make_values_M3,
    make_values_M3_with_volatility,
)
from .agent_pymdp import (
    ChangepointAgentPymdp,
    AgentConfig,
)
from .evaluation import (
    evaluate_agent,
    evaluate_agent_on_dataframe,
    compute_prediction_ll,
)
from .profile_utils import (
    create_exploit_phi_logits,
    create_dynamic_phi_generator,
)

__all__ = [
    # State space
    "PositionStateSpace",
    "VolatilityStateSpace",
    "create_state_space",
    # Generative model
    "build_generative_model",
    "build_A",
    "build_B",
    "build_D",
    # Value functions
    "make_value_fn",
    "make_values_M1",
    "make_values_M2",
    "make_values_M3",
    "make_values_M3_with_volatility",
    # Agent
    "ChangepointAgentPymdp",
    "AgentConfig",
    # Evaluation
    "evaluate_agent",
    "evaluate_agent_on_dataframe",
    "compute_prediction_ll",
    # Profile utilities
    "create_exploit_phi_logits",
    "create_dynamic_phi_generator",
]

