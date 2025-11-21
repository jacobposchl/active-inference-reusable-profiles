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
import os
import concurrent.futures

from src.utils.ll_eval import _worker_init, _eval_m1_gamma, _eval_m2_params, _eval_m3_params_per_profile

from src.models import build_A, build_B, build_D, make_value_fn
from src.utils.model_utils import create_model
from src.utils.simulate import simulate_baseline_run
from src.models.agent_wrapper import run_episode_with_ll
from src.utils.ll_eval import evaluate_ll_with_valuefn
from config.experiment_config import *
from src.environment import TwoArmedBandit
from src.models import AgentRunnerWithLL
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

            env = TwoArmedBandit(
                probability_hint=PROBABILITY_HINT, probability_reward=PROBABILITY_REWARD,
                reversal_schedule=reversal_schedule
            )

            if gen in ('M1', 'M2', 'M3'):
                value_fn = create_model(gen, A, B, D)
                runner = AgentRunnerWithLL(
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
                runner = AgentRunnerWithLL(
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
    # Number of worker processes for parallel sections (can be set via env)
    max_workers = int(os.environ.get('MODEL_COMP_MAX_WORKERS', os.cpu_count() or 1))

    if model_name == 'M1':
        # Two-stage search for M1: coarse grid then local refinement.
        # Parallelize evaluations across CPU cores using worker helpers.
        coarse_g = [0.5, 1.0, 1.5, 2.5, 4.0, 8.0, 12.0, 16.0]
        max_workers = int(os.environ.get('MODEL_COMP_MAX_WORKERS', os.cpu_count() or 1))

        # Submit (g, ref) jobs to workers which use A/B/D from initializer.
        coarse_scores = {g: 0.0 for g in coarse_g}
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init, initargs=(A, B, D)) as exe:
            future_map = {}
            for g in coarse_g:
                for ref in ref_logs_list:
                    fut = exe.submit(_eval_m1_gamma, None, None, None, g, ref)
                    future_map[fut] = g
            for fut in concurrent.futures.as_completed(future_map):
                g = future_map[fut]
                try:
                    val = fut.result()
                except Exception:
                    val = -np.inf
                coarse_scores[g] += float(val)

        # Determine best coarse gamma
        best_g = max(coarse_scores, key=coarse_scores.get)
        best_ll = coarse_scores[best_g]
        best_params = {'gamma': best_g}

        # Local refinement around best coarse point (run sequentially, small grid)
        best_idx = int(coarse_g.index(best_g))
        lo_idx = max(0, best_idx - 1)
        hi_idx = min(len(coarse_g) - 1, best_idx + 1)
        lo = coarse_g[lo_idx]
        hi = coarse_g[hi_idx]
        if best_idx == 0:
            lo = max(0.1, coarse_g[0] * 0.5)
        if best_idx == len(coarse_g) - 1:
            hi = coarse_g[-1] * 1.25

        fine_g = np.linspace(lo, hi, 7)
        for g in fine_g:
            value_fn = make_value_fn('M1', C_reward_logits=M1_DEFAULTS['C_reward_logits'], gamma=float(g))
            total = 0.0
            for ref in ref_logs_list:
                ll, _ = evaluate_ll_with_valuefn(value_fn, A, B, D, ref)
                total += ll
            if total > best_ll:
                best_ll = total
                best_params = {'gamma': float(g)}

    elif model_name == 'M2':
        # Two-stage search for M2: coarse grid then local refinement
        coarse_g_base = [0.5, 1.0, 1.5, 2.5, 4.0, 8.0]
        coarse_k = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
        coarse_scores = np.zeros((len(coarse_g_base), len(coarse_k)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init, initargs=(A, B, D)) as exe:
            future_map = {}
            for i, g_base in enumerate(coarse_g_base):
                for j, k in enumerate(coarse_k):
                    for ref in ref_logs_list:
                        fut = exe.submit(_eval_m2_params, None, None, None, g_base, k, ref)
                        future_map[fut] = (i, j)
            for fut in concurrent.futures.as_completed(future_map):
                i, j = future_map[fut]
                try:
                    val = fut.result()
                except Exception:
                    val = -np.inf
                coarse_scores[i, j] += float(val)

        # Find best coarse indices and define local refinement bounds
        best_flat = np.argmax(coarse_scores)
        bi, bj = np.unravel_index(best_flat, coarse_scores.shape)
        # bounds for g_base
        gi_lo = max(0, bi - 1)
        gi_hi = min(len(coarse_g_base) - 1, bi + 1)
        gb_lo = coarse_g_base[gi_lo]
        gb_hi = coarse_g_base[gi_hi]
        if bi == 0:
            gb_lo = max(0.1, coarse_g_base[0] * 0.5)
        if bi == len(coarse_g_base) - 1:
            gb_hi = coarse_g_base[-1] * 1.25

        # bounds for k
        kj_lo = max(0, bj - 1)
        kj_hi = min(len(coarse_k) - 1, bj + 1)
        k_lo = coarse_k[kj_lo]
        k_hi = coarse_k[kj_hi]
        if bj == 0:
            k_lo = max(1e-3, coarse_k[0] * 0.5)
        if bj == len(coarse_k) - 1:
            k_hi = coarse_k[-1] * 1.5

        fine_gb = np.linspace(gb_lo, gb_hi, 6)
        fine_k = np.linspace(k_lo, k_hi, 6)
        for g_base in fine_gb:
            for k in fine_k:
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
                    best_params = {'gamma_base': float(g_base), 'entropy_k': float(k)}

    elif model_name == 'M3':
        # More flexible M3 grid search: allow per-profile gammas and per-profile
        # xi scaling for the non-start actions (hint, left, right). This keeps
        # the search expressive while remaining manageable.
        policies, num_actions_per_factor = _make_temp_agent_and_policies(A, B, D)

        # Grid definitions (expandable)
        gamma_vals = [1.0, 2.5, 8.0]
        xi_scale_values = [0.5, 1.0, 2.0]

        # For each profile we'll search independent gammas and xi scales for
        # the three actionable indices (skip the 'start' action at idx 0).
        from itertools import product

        num_profiles = len(M3_DEFAULTS['profiles'])

        # Parallel coarse evaluation across candidate combinations using worker helper
        per_profile_xi_combos = list(product(xi_scale_values, repeat=3))
        # Build list of all candidate combinations (may be large)
        candidates = []
        for gammas in product(gamma_vals, repeat=num_profiles):
            for xi_combo_all in product(per_profile_xi_combos, repeat=num_profiles):
                candidates.append((list(gammas), [list(x) for x in xi_combo_all]))

        # Evaluate each candidate across all training refs in parallel
        max_workers = int(os.environ.get('MODEL_COMP_MAX_WORKERS', os.cpu_count() or 1))
        candidate_scores = [0.0] * len(candidates)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init, initargs=(A, B, D)) as exe:
            future_map = {}
            for c_idx, (gammas_c, xi_scales_c) in enumerate(candidates):
                for ref in ref_logs_list:
                    fut = exe.submit(_eval_m3_params_per_profile, None, None, None, gammas_c, xi_scales_c, ref)
                    future_map[fut] = c_idx
            for fut in concurrent.futures.as_completed(future_map):
                c_idx = future_map[fut]
                try:
                    val = fut.result()
                except Exception:
                    val = -np.inf
                candidate_scores[c_idx] += float(val)

        # pick best candidate
        best_idx = int(np.nanargmax(candidate_scores))
        best_ll = candidate_scores[best_idx]
        best_gammas, best_xi_scales = candidates[best_idx]
        best_params = {'gamma_profile': best_gammas, 'xi_scales_profile': best_xi_scales}

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
        # Backward-compatible: support old params format with single 'gamma' and 'xi_scale'
        if params is None:
            params = {}

        if 'gamma' in params and 'xi_scale' in params:
            # legacy format: apply same gamma and xi_scale to all profiles
            for p in M3_DEFAULTS['profiles']:
                prof = dict(p)
                prof['gamma'] = params['gamma']
                prof['xi_logits'] = (np.array(p['xi_logits'], float) * params['xi_scale']).tolist()
                profiles.append(prof)
        elif 'gamma_profile' in params and 'xi_scales_profile' in params:
            # new format: per-profile gammas and per-profile xi triple scales
            gamma_profile = params['gamma_profile']
            xi_scales_profile = params['xi_scales_profile']
            for p_idx, p in enumerate(M3_DEFAULTS['profiles']):
                prof = dict(p)
                prof['gamma'] = float(gamma_profile[p_idx])
                orig_xi = np.array(p['xi_logits'], float)
                scales3 = xi_scales_profile[p_idx]
                new_xi = orig_xi.copy()
                new_xi[1] = orig_xi[1] * float(scales3[0])
                new_xi[2] = orig_xi[2] * float(scales3[1])
                new_xi[3] = orig_xi[3] * float(scales3[2])
                prof['xi_logits'] = new_xi.tolist()
                profiles.append(prof)
        else:
            # fallback: use defaults
            for p in M3_DEFAULTS['profiles']:
                profiles.append(dict(p))

        value_fn = make_value_fn('M3', profiles=profiles, Z=np.array(M3_DEFAULTS['Z']), policies=policies, num_actions_per_factor=num_actions_per_factor)

    total = 0.0
    for ref in ref_logs_list:
        ll, _ = evaluate_ll_with_valuefn(value_fn, A, B, D, ref)
        total += ll
    return total
