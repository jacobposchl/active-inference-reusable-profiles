"""Shared helpers for recovery-style experiments.

This module contains routines reused by `recovery.py` and `cv_recovery.py`:
- building A/B/D matrices
- generating reference runs
- fitting small parameter grids for M1/M2/M3
- evaluating fitted params on held-out runs

Keep these helpers focused and side-effect free where possible.
"""
import numpy as np
import logging

from src.models import build_A, build_B, build_D, make_value_fn
from src.utils.model_utils import create_model
from src.utils.simulate import simulate_baseline_run
from src.models.agent_wrapper import run_episode_with_ll
from src.utils.ll_eval import evaluate_ll_with_valuefn
from config.experiment_config import *
# Import pymdp lazily inside functions that need it to avoid import-time
# dependency failures when pymdp is not installed (tests may skip runtime tests).


def build_abd():
    """Build and return (A, B, D) using config constants."""
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)
    return A, B, D


def generate_all_runs(generators, runs_per_generator, num_trials, seed, reversal_interval=None):
    """Generate reference rollouts for a set of generators.

    Returns (A, B, D, refs) where refs is a list of dicts
    {'gen', 'run_idx', 'seed', 'ref_logs'}.
    """
    A, B, D = build_abd()

    refs = []
    for gen in generators:
        for r in range(runs_per_generator):
            run_seed = int(seed + r + (abs(hash(gen)) % 1000))
            if reversal_interval is not None:
                reversal_schedule = [i for i in range(reversal_interval, num_trials, reversal_interval)]
            else:
                reversal_schedule = DEFAULT_REVERSAL_SCHEDULE

            env = __import__('src.environment', fromlist=['TwoArmedBandit']).environment.TwoArmedBandit(
                probability_hint=PROBABILITY_HINT, probability_reward=PROBABILITY_REWARD,
                reversal_schedule=reversal_schedule
            )

            if gen in ('M1', 'M2', 'M3'):
                value_fn = create_model(gen, A, B, D)
                runner = __import__('src.models', fromlist=['AgentRunnerWithLL']).models.AgentRunnerWithLL(
                    A, B, D, value_fn,
                    OBSERVATION_HINTS, OBSERVATION_REWARDS,
                    OBSERVATION_CHOICES, ACTION_CHOICES,
                    reward_mod_idx=1
                )
                try:
                    runner.agent.gamma = 8.0
                except Exception as e:
                    logging.warning("Could not configure runner.agent for %s generator: %s", gen, e)
                ref_logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)

            elif gen in ('egreedy', 'softmax'):
                ref_logs = simulate_baseline_run(env, policy_type=gen, T=num_trials, seed=run_seed,
                                                 epsilon=0.1, temp=1.0, alpha=0.1)
            else:
                value_fn = create_model('M3', A, B, D)
                runner = __import__('src.models', fromlist=['AgentRunnerWithLL']).models.AgentRunnerWithLL(
                    A, B, D, value_fn,
                    OBSERVATION_HINTS, OBSERVATION_REWARDS,
                    OBSERVATION_CHOICES, ACTION_CHOICES,
                    reward_mod_idx=1
                )
                ref_logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)

            refs.append({'gen': gen, 'run_idx': r, 'seed': run_seed, 'ref_logs': ref_logs})

    return A, B, D, refs


def _make_temp_agent_and_policies(A, B, D):
    """Create a temporary pymdp Agent to extract policies for M3 value function."""
    # Import pymdp lazily so this module can be imported in environments
    # without pymdp (tests that only check imports should not require it).
    from pymdp.agent import Agent
    from pymdp import utils as pymdp_utils

    C_temp = pymdp_utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
    temp_agent = Agent(A=A, B=B, C=C_temp, D=D,
                      policy_len=2, inference_horizon=1,
                      control_fac_idx=[1], use_utility=True,
                      use_states_info_gain=True,
                      action_selection="stochastic", gamma=16)
    policies = temp_agent.policies
    num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
    return policies, num_actions_per_factor


def fit_model_on_runs(model_name, A, B, D, ref_logs_list):
    """Fit model via small grid search on provided reference logs (list).

    Returns best_params (dict).
    """
    best_ll = -np.inf
    best_params = None

    if model_name == 'M1':
        for g in [1.0, 2.5, 8.0, 16.0]:
            value_fn = make_value_fn('M1', C_reward_logits=M1_DEFAULTS['C_reward_logits'], gamma=g)
            total = 0.0
            for ref in ref_logs_list:
                ll, _ = evaluate_ll_with_valuefn(value_fn, A, B, D, ref)
                total += ll
            if total > best_ll:
                best_ll = total
                best_params = {'gamma': g}

    elif model_name == 'M2':
        for g_base in [1.0, 2.5, 8.0]:
            for k in [0.1, 1.0, 4.0]:
                def gamma_schedule(q, t, g_base=g_base, k=k):
                    p = np.clip(np.asarray(q, float), 1e-12, 1.0)
                    H = -(p * np.log(p)).sum()
                    return g_base / (1.0 + k * H)
                value_fn = make_value_fn('M2', C_reward_logits=M2_DEFAULTS['C_reward_logits'], gamma_schedule=gamma_schedule)
                total = 0.0
                for ref in ref_logs_list:
                    ll, _ = evaluate_ll_with_valuefn(value_fn, A, B, D, ref)
                    total += ll
                if total > best_ll:
                    best_ll = total
                    best_params = {'gamma_base': g_base, 'entropy_k': k}

    elif model_name == 'M3':
        policies, num_actions_per_factor = _make_temp_agent_and_policies(A, B, D)
        for g in [1.0, 2.5, 8.0]:
            for xi_scale in [0.5, 1.0, 2.0]:
                profiles = []
                for p in M3_DEFAULTS['profiles']:
                    prof = dict(p)
                    prof['gamma'] = g
                    prof['xi_logits'] = (np.array(p['xi_logits'], float) * xi_scale).tolist()
                    profiles.append(prof)
                value_fn = make_value_fn('M3', profiles=profiles, Z=np.array(M3_DEFAULTS['Z']), policies=policies, num_actions_per_factor=num_actions_per_factor)
                total = 0.0
                for ref in ref_logs_list:
                    ll, _ = evaluate_ll_with_valuefn(value_fn, A, B, D, ref)
                    total += ll
                if total > best_ll:
                    best_ll = total
                    best_params = {'gamma': g, 'xi_scale': xi_scale}

    return best_params


def evaluate_on_test(model_name, A, B, D, params, ref_logs_list):
    """Evaluate fitted `params` for `model_name` on a list of reference logs.

    Returns total log-likelihood across provided refs.
    """
    if model_name == 'M1':
        value_fn = make_value_fn('M1', C_reward_logits=M1_DEFAULTS['C_reward_logits'], gamma=params['gamma'])
    elif model_name == 'M2':
        def gamma_schedule(q, t, g_base=params['gamma_base'], k=params['entropy_k']):
            p = np.clip(np.asarray(q, float), 1e-12, 1.0)
            H = -(p * np.log(p)).sum()
            return g_base / (1.0 + k * H)
        value_fn = make_value_fn('M2', C_reward_logits=M2_DEFAULTS['C_reward_logits'], gamma_schedule=gamma_schedule)
    else:
        policies, num_actions_per_factor = _make_temp_agent_and_policies(A, B, D)
        profiles = []
        for p in M3_DEFAULTS['profiles']:
            prof = dict(p)
            prof['gamma'] = params['gamma']
            prof['xi_logits'] = (np.array(p['xi_logits'], float) * params['xi_scale']).tolist()
            profiles.append(prof)
        value_fn = make_value_fn('M3', profiles=profiles, Z=np.array(M3_DEFAULTS['Z']), policies=policies, num_actions_per_factor=num_actions_per_factor)

    total = 0.0
    for ref in ref_logs_list:
        ll, _ = evaluate_ll_with_valuefn(value_fn, A, B, D, ref)
        total += ll
    return total
