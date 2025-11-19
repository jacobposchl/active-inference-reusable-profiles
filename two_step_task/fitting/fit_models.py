"""Fitting scaffolds for two-step task models.

This module includes an optimizer loop and a runnable
`compute_log_likelihood` that uses `TwoStepAgent` to step through a
participant's trials. The agent maintains Bayesian reward estimates per
planet/alien and computes action probabilities using model-specific
precision rules (M1/M2/M3).
"""
from typing import Dict, Tuple
import numpy as np
from scipy.optimize import minimize

from pymdp.agent import Agent
from pymdp import utils

from two_step_task.models.generative import build_A_matrices, build_B_matrices
from two_step_task.ai.agent_runner import AgentRunnerWithLL
from two_step_task.models.value_functions import make_value_fn_for_model
from two_step_task.data.data_loader import planet_to_obs_idx, row_to_obs_ids


def sample_random_params(param_bounds: Dict[str, Tuple[float, float]]):
    params = []
    for (low, high) in param_bounds.values():
        params.append(np.random.uniform(low, high))
    return np.array(params)


def bounds_from_paramdict(param_bounds: Dict[str, Tuple[float, float]]):
    return list(param_bounds.values())


def param_vector_to_kwargs(param_bounds: Dict[str, Tuple[float, float]], x: np.ndarray):
    """Map parameter vector `x` to a dict keyed by param_bounds keys.

    The ordering of `x` must match the iteration order of
    `param_bounds.keys()`.
    """
    keys = list(param_bounds.keys())
    if len(keys) != len(x):
        raise ValueError("Parameter vector length does not match bounds")
    return {k: float(v) for k, v in zip(keys, x)}


def set_model_params_from_vector(model, param_bounds: Dict[str, Tuple[float, float]], x: np.ndarray):
    """Assign parameter values from vector `x` to attributes on `model`.

    Matches names directly (i.e., a key 'gamma' sets `model.gamma`).
    """
    kwargs = param_vector_to_kwargs(param_bounds, x)
    for k, v in kwargs.items():
        setattr(model, k, v)


def fit_model_to_participant(model, participant_df, n_restarts: int = 5):
    param_bounds = model.free_parameters()
    best_ll = -np.inf
    best_params = None

    for restart in range(n_restarts):
        x0 = sample_random_params(param_bounds)
        try:
            res = minimize(
                fun=lambda x: -compute_log_likelihood(model, x, participant_df),
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds_from_paramdict(param_bounds),
            )
        except Exception as e:
            print(f"Optimization failed on restart {restart}: {e}")
            continue

        ll = -res.fun
        if ll > best_ll:
            best_ll = ll
            best_params = res.x

    return best_params, best_ll


def compute_log_likelihood(model, params, participant_df):
    """Compute log-likelihood of participant's choices under a model.

    Implementation details:
    - `params` is a 1D numpy vector whose order matches
      `model.free_parameters()` iteration.
    - The function sets parameters on the `model`, instantiates a
      `TwoStepAgent`, and runs the trial loop accumulating log-prob.
    """
    param_bounds = model.free_parameters()
    # Set model parameters from params vector
    set_model_params_from_vector(model, param_bounds, params)

    # Build generative matrices and pymdp.Agent
    A_list = build_A_matrices(num_contexts=4)
    B_list = build_B_matrices(context_volatility=0.01)

    # pymdp expects object-array containers for modality matrices
    import numpy as _np
    A_list = _np.array(A_list, dtype=object)
    B_list = _np.array(B_list, dtype=object)

    # Normalize B matrices along the "next state" axis (axis=0) to ensure
    # they represent valid conditional distributions P(s' | s, a).
    for i in range(len(B_list)):
        B = B_list[i]
        try:
            sums = B.sum(axis=0, keepdims=True)
            # For any conditional column with zero total probability, replace
            # with a uniform distribution across next-states before normalizing.
            zero_mask = (sums == 0)
            if zero_mask.any():
                # squeeze to index into remaining axes
                zm = zero_mask.squeeze(axis=0)
                # iterate over indices where zero and set uniform
                it = __import__('numpy').ndindex(*zm.shape)
                for idx in it:
                    if zm[idx]:
                        # build full slice for assignment
                        sl = (slice(None),) + idx
                        B[sl] = 1.0 / float(B.shape[0])
                # recompute sums after filling
                sums = B.sum(axis=0, keepdims=True)
            B = B / sums
            B_list[i] = B
        except Exception:
            # if B doesn't have an axis 0 to sum over, skip normalization
            pass

    # Ensure B matrices have a control/action axis (3 dims). Some builders
    # return 2D matrices for uncontrolled factors; pymdp expects a 3rd
    # axis indexing controls. Expand dims where necessary.
    for i in range(len(B_list)):
        B = B_list[i]
        try:
            if getattr(B, 'ndim', None) == 2:
                B = B[:, :, None]
                B_list[i] = B
        except Exception:
            pass

    # initial priors
    n_contexts = A_list[1].shape[1]
    D_context = np.ones(n_contexts) / float(n_contexts)
    n_stages = A_list[0].shape[2]
    D_stage = np.zeros(n_stages)
    D_stage[0] = 1.0
    D = [D_context, D_stage]
    import numpy as _np
    D = _np.array(D, dtype=object)

    C0 = utils.obj_array_zeros([(A_list[m].shape[0],) for m in range(len(A_list))])

    agent = Agent(
        A=A_list,
        B=B_list,
        C=C0,
        D=D,
        policy_len=2,
        inference_horizon=1,
        control_fac_idx=[1],
        use_utility=True,
        use_states_info_gain=True,
        action_selection="stochastic",
        gamma=1.0,
    )

    # Create runner with value function derived from model
    value_fn = make_value_fn_for_model(model)
    runner = AgentRunnerWithLL(agent, reward_mod_idx=len(A_list) - 1, value_fn=value_fn, learning_mode="dirichlet")

    ll = 0.0

    # ensure participant_df sorted by trial
    df = participant_df.sort_values("trial")

    for row in df.itertuples(index=False):
        # Stage 1: spaceship choice
        try:
            actual_s1 = int(getattr(row, "stage1_choice"))
        except Exception:
            raise ValueError("participant_df missing 'stage1_choice' column")

        # Stage1 log-likelihood from agent posterior over policies
        # build centralized obs mapping for this trial
        full_obs = row_to_obs_ids(row)

        # At stage1 the agent has not observed the planet/alien/reward yet; keep
        # the modalities unobserved to mirror previous behavior.
        obs_ids = [0] * len(A_list)
        t = int(getattr(row, "trial"))
        ll += runner.log_likelihood_of_action(obs_ids, action_stage=0, action_choice=actual_s1, t=t)

        # Observe transition / planet (participant provides 'planet')
        planet = getattr(row, "planet", None)
        if planet is None:
            raise ValueError("participant_df missing 'planet' column")
        # Stage 2: alien choice
        try:
            actual_s2 = int(getattr(row, "stage2_choice"))
        except Exception:
            raise ValueError("participant_df missing 'stage2_choice' column")

        # Map planet string to planet observation index used in A matrices
        # Use centralized mapping helper to ensure consistency
        obs_ids = [0] * len(A_list)
        # prefer the centralized full_obs mapping but only expose the planet
        # modality at stage2 (agent sees the planet before choosing alien)
        obs_ids[1] = int(full_obs[1])
        ll += runner.log_likelihood_of_action(obs_ids, action_stage=1, action_choice=actual_s2, t=t)

        # Feedback / reward
        try:
            reward = int(getattr(row, "reward"))
        except Exception:
            raise ValueError("participant_df missing 'reward' column")

        # Pass the centralized observation mapping to the runner for learning
        runner.update_after_feedback(actual_s2, reward, planet, obs_ids=full_obs)

    return float(ll)
