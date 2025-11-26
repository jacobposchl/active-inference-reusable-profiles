"""Helpers for recovery-style experiments.

- building A/B/D matrices
- generating reference runs
- fitting small parameter grids for M1/M2/M3
- evaluating fitted params on held-out runs

Keep these helpers focused and side-effect free where possible.
"""
import os
import logging
import json
import time
import concurrent.futures
from collections import Counter
import numpy as np

from src.utils.ll_eval import (
    compute_sequence_ll_for_model,
    evaluate_ll_with_valuefn,
    _worker_init,
    compute_sequence_ll_for_model_worker,
    _eval_m1_gamma,
    _eval_m2_params,
    _eval_m3_params,
    _eval_m1_gamma_masked,
    _eval_m2_params_masked,
    _eval_m3_params_masked,
    _eval_m3_params_per_profile,
    _eval_m3_params_per_profile_masked,
    evaluate_ll_with_valuefn_masked,
)

from src.models import build_A, build_B, build_D, make_value_fn
from src.utils.model_utils import create_model, get_num_parameters
from src.utils.simulate import simulate_baseline_run
from src.models.agent_wrapper import run_episode_with_ll
from config.experiment_config import *
from src.environment import TwoArmedBandit
from src.models import AgentRunnerWithLL
from src.utils.helpers import compute_entropy, find_reversals

MODEL_RECOVERY_BASE = os.path.join('results', 'model_recovery')


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _trial_level_path(generator, model_name, run_idx, base_dir=MODEL_RECOVERY_BASE):
    return os.path.join(
        base_dir,
        'trial_level',
        f'gen_{generator}',
        f'model_{model_name}',
        f'run_{int(run_idx):03d}.csv'
    )


def _fold_level_path(generator, model_name, run_idx, base_dir=MODEL_RECOVERY_BASE):
    return os.path.join(
        base_dir,
        'fold_level',
        f'gen_{generator}',
        f'model_{model_name}',
        f'run_{int(run_idx):03d}.csv'
    )


def _run_summary_path(generator, model_name, base_dir=MODEL_RECOVERY_BASE):
    return os.path.join(
        base_dir,
        'run_summary',
        f'gen_{generator}',
        f'model_{model_name}.csv'
    )


def _grid_eval_path(generator, model_name, run_idx, fold_idx, base_dir=MODEL_RECOVERY_BASE):
    return os.path.join(
        base_dir,
        'grid_evals',
        f'gen_{generator}',
        f'model_{model_name}',
        f'run_{int(run_idx):03d}_fold_{int(fold_idx)}.csv'
    )


def _confusion_matrix_path(metric, statistic, base_dir=MODEL_RECOVERY_BASE):
    return os.path.join(
        base_dir,
        'confusion',
        f'{metric}_{statistic}.csv'
    )


def _metadata_path(base_dir=MODEL_RECOVERY_BASE):
    return os.path.join(base_dir, 'metadata', 'experiment_summary.json')


def _to_native(obj):
    """Convert numpy scalars/arrays to plain Python types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    return obj


def _write_dict_csv(path, fieldnames, rows, mode='w'):
    import csv
    file_exists = os.path.exists(path)
    _ensure_parent(path)
    with open(path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w' or not file_exists:
            writer.writeheader()
        writer.writerows(rows if isinstance(rows, list) else [rows])


def build_abd():
    """Build and return (A, B, D) using config constants.
    
    Note: With the new volatile/stable contexts, the agent's A matrix represents
    beliefs about observation likelihoods. We use average reward probabilities
    since the agent doesn't know the exact probabilities in each context a priori.
    """
    # Use average reward probability for agent's beliefs
    # Agent learns the actual probabilities through experience
    avg_reward_prob = 0.80  # Reasonable middle ground between contexts
    
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, avg_reward_prob)
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

            # Create environment with new volatile/stable parameters
            env = TwoArmedBandit(
                probability_hint=PROBABILITY_HINT,
                volatile_reward_better=VOLATILE_REWARD_BETTER,
                volatile_reward_worse=VOLATILE_REWARD_WORSE,
                stable_reward_better=STABLE_REWARD_BETTER,
                stable_reward_worse=STABLE_REWARD_WORSE,
                volatile_switch_interval=VOLATILE_SWITCH_INTERVAL,
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


def fit_model_on_runs(model_name, A, B, D, ref_logs_list, save_grid=False, save_dir=None):
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
        coarse_evals = []
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
                coarse_evals.append((g, float(val)))

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
        fine_evals = []
        # Parallelize fine-grid evaluation using worker helpers to reuse A/B/D in workers
        max_workers = int(os.environ.get('MODEL_COMP_MAX_WORKERS', os.cpu_count() or 1))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init, initargs=(A, B, D)) as exe:
            fut_map = {}
            for g in fine_g:
                for ref in ref_logs_list:
                    fut = exe.submit(_eval_m1_gamma, None, None, None, float(g), ref)
                    fut_map[fut] = float(g)
            # accumulate totals per gamma
            fine_totals = {float(g): 0.0 for g in fine_g}
            for fut in concurrent.futures.as_completed(fut_map):
                g = fut_map[fut]
                try:
                    val = fut.result()
                except Exception:
                    val = -np.inf
                fine_totals[g] += float(val)
        for g, total in fine_totals.items():
            fine_evals.append((float(g), float(total)))
            if total > best_ll:
                best_ll = total
                best_params = {'gamma': float(g)}
        # optionally save grid evaluations
        if save_grid:
            if save_dir is None:
                save_dir = os.path.join('results', 'csv')
            os.makedirs(save_dir, exist_ok=True)
            import csv
            path_coarse = os.path.join(save_dir, 'grid_m1_coarse.csv')
            with open(path_coarse, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['gamma', 'aggregate_ll'])
                for gg, vv in coarse_scores.items():
                    w.writerow([gg, vv])
            path_fine = os.path.join(save_dir, 'grid_m1_fine.csv')
            with open(path_fine, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['gamma', 'total_ll'])
                for gg, vv in fine_evals:
                    w.writerow([gg, vv])

    elif model_name == 'M2':
        # Two-stage search for M2: coarse grid then local refinement
        coarse_g_base = [0.5, 1.0, 1.5, 2.5, 4.0, 8.0]
        coarse_k = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
        coarse_scores = np.zeros((len(coarse_g_base), len(coarse_k)))
        coarse_evals = []
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
                coarse_evals.append((i, j, float(val)))

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
        fine_evals = []
        # Parallelize fine M2 grid using worker helpers
        max_workers = int(os.environ.get('MODEL_COMP_MAX_WORKERS', os.cpu_count() or 1))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init, initargs=(A, B, D)) as exe:
            fut_map = {}
            for g_base in fine_gb:
                for k in fine_k:
                    for ref in ref_logs_list:
                        fut = exe.submit(_eval_m2_params, None, None, None, float(g_base), float(k), ref)
                        fut_map[fut] = (float(g_base), float(k))
            # accumulate totals
            fine_totals = {(float(g), float(k)): 0.0 for g in fine_gb for k in fine_k}
            for fut in concurrent.futures.as_completed(fut_map):
                g_base, k_val = fut_map[fut]
                try:
                    val = fut.result()
                except Exception:
                    val = -np.inf
                fine_totals[(g_base, k_val)] += float(val)
        for (gb, kk), total in fine_totals.items():
            fine_evals.append((float(gb), float(kk), float(total)))
            if total > best_ll:
                best_ll = total
                best_params = {'gamma_base': float(gb), 'entropy_k': float(kk)}
        if save_grid:
            if save_dir is None:
                save_dir = os.path.join('results', 'csv')
            os.makedirs(save_dir, exist_ok=True)
            import csv
            path_coarse = os.path.join(save_dir, 'grid_m2_coarse.csv')
            with open(path_coarse, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['g_base_idx', 'k_idx', 'aggregate_ll'])
                for i, j, v in coarse_evals:
                    w.writerow([i, j, v])
            path_fine = os.path.join(save_dir, 'grid_m2_fine.csv')
            with open(path_fine, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['g_base', 'k', 'total_ll'])
                for gb, kk, vv in fine_evals:
                    w.writerow([gb, kk, vv])

    elif model_name == 'M3':
        # SMART CONSTRAINED SEARCH: Exploit M3's symmetric profile structure
        # Assumption: Xi scales are symmetric (same magnitude, opposite signs for arms)
        # Allows: Independent gamma per profile (no assumption of equal precision)
        # This reduces search space from 6,561 to 108 candidates.
        policies, num_actions_per_factor = _make_temp_agent_and_policies(A, B, D)

        # Grid definitions
        gamma_vals = [1.0, 2.5, 5.0]  # Each profile can have different gamma
        xi_scale_hint = [0.5, 1.0, 2.0, 4.0]  # Hint scale (symmetric across profiles)
        xi_scale_arm = [0.5, 1.0, 2.0]  # Arm bias magnitude (symmetric)
        
        from itertools import product
        num_profiles = len(M3_DEFAULTS['profiles'])

        # Build constrained candidates with symmetric xi but independent gammas
        candidates = []
        for gamma_p0 in gamma_vals:
            for gamma_p1 in gamma_vals:
                for hint_scale in xi_scale_hint:
                    for arm_scale in xi_scale_arm:
                        # Allow different gamma per profile
                        gammas = [gamma_p0, gamma_p1]
                        
                        # Symmetric xi scaling across profiles
                        xi_scales = [
                            [hint_scale, arm_scale, arm_scale],  # Profile 0
                            [hint_scale, arm_scale, arm_scale]   # Profile 1
                        ]
                        candidates.append((gammas, xi_scales))

        # Evaluate each candidate across all training refs in parallel
        max_workers = int(os.environ.get('MODEL_COMP_MAX_WORKERS', os.cpu_count() or 1))
        candidate_scores = [0.0] * len(candidates)
        candidate_evals = []
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
                candidate_evals.append((c_idx, float(val)))

        # pick best candidate
        best_idx = int(np.nanargmax(candidate_scores))
        best_ll = candidate_scores[best_idx]
        best_gammas, best_xi_scales = candidates[best_idx]
        best_params = {'gamma_profile': best_gammas, 'xi_scales_profile': best_xi_scales}
        if save_grid:
            if save_dir is None:
                save_dir = os.path.join('results', 'csv')
            os.makedirs(save_dir, exist_ok=True)
            import csv
            path_candidates = os.path.join(save_dir, 'grid_m3_candidates.csv')
            with open(path_candidates, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['candidate_idx', 'gamma_profile', 'xi_scales_profile', 'score'])
                for idx, score in enumerate(candidate_scores):
                    gam, xi = candidates[idx]
                    w.writerow([idx, str(gam), str(xi), score])

    # Save metadata about this fit call if requested
    try:
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            meta = {
                'model_name': model_name,
                'n_refs': len(ref_logs_list),
                'save_grid': bool(save_grid),
                'timestamp': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
                'coarse_grid_counts': {
                    'M1_coarse': len(coarse_g) if model_name == 'M1' else None,
                    'M2_coarse': (len(coarse_g_base), len(coarse_k)) if model_name == 'M2' else None,
                    'M3_candidates': len(candidates) if model_name == 'M3' else None
                },
                'python_version': __import__('platform').python_version()
            }
            # attempt to add pymdp version
            try:
                import pymdp
                meta['pymdp_version'] = getattr(pymdp, '__version__', None)
            except Exception:
                meta['pymdp_version'] = None

            meta_path = os.path.join(save_dir, f'metadata_fit_{model_name}.json')
            with open(meta_path, 'w') as mf:
                json.dump(meta, mf, indent=2)
    except Exception:
        # do not fail fitting if metadata save fails
        pass

    return best_params




def cv_fit_single_run(
    model_name,
    A,
    B,
    D,
    ref_logs,
    K=5,
    run_id=None,
    generator=None,
    seed=None,
    artifact_base_dir=MODEL_RECOVERY_BASE,
    save_artifacts=True,
    record_grid=False,
):
    """Perform within-run K-fold CV for a single reference run with rich logging."""
    N = len(ref_logs['action'])
    idx = np.arange(N)
    folds = np.array_split(idx, K)

    fold_results = []
    trial_rows_all = []
    total_grid_evals = 0
    run_start = time.time()
    
    # Import tqdm for fold progress (optional, graceful fallback)
    try:
        from tqdm import tqdm as _tqdm
        use_progress = True
    except ImportError:
        use_progress = False

    fold_iter = range(K)
    if use_progress and K > 1:
        fold_iter = _tqdm(fold_iter, desc=f"    ├─ Folds ({model_name})", leave=False, position=2)

    for k in fold_iter:
        test_idx = [int(i) for i in folds[k]]
        test_idx_set = set(test_idx)
        train_idx = [int(i) for i in idx if int(i) not in test_idx_set]
        best_ll = -np.inf
        best_params = None
        fold_grid_records = []
        fold_grid_evals = 0

        if model_name == 'M1':
            coarse_g = [0.5, 1.0, 1.5, 2.5, 4.0, 8.0, 12.0, 16.0]
            max_workers = int(os.environ.get('MODEL_COMP_MAX_WORKERS', os.cpu_count() or 1))
            coarse_scores = {}
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers, initializer=_worker_init, initargs=(A, B, D)
            ) as exe:
                fut_map = {}
                for g in coarse_g:
                    fold_grid_evals += 1
                    fut = exe.submit(
                        _eval_m1_gamma_masked, None, None, None, float(g), ref_logs, train_idx
                    )
                    fut_map[fut] = float(g)
                for fut in concurrent.futures.as_completed(fut_map):
                    g = fut_map[fut]
                    try:
                        val = fut.result()
                    except Exception:
                        val = -np.inf
                    coarse_scores[g] = float(val)
                    if record_grid:
                        fold_grid_records.append(
                            {'fold': k, 'stage': 'coarse', 'gamma': float(g), 'll': float(val)}
                        )

            best_g = max(coarse_scores, key=coarse_scores.get)
            best_ll = coarse_scores[best_g]
            best_params = {'gamma': best_g}

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
            fine_scores = {}
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers, initializer=_worker_init, initargs=(A, B, D)
            ) as exe:
                fut_map = {}
                for g in fine_g:
                    fold_grid_evals += 1
                    fut = exe.submit(
                        _eval_m1_gamma_masked, None, None, None, float(g), ref_logs, train_idx
                    )
                    fut_map[fut] = float(g)
                for fut in concurrent.futures.as_completed(fut_map):
                    g = fut_map[fut]
                    try:
                        val = fut.result()
                    except Exception:
                        val = -np.inf
                    fine_scores[float(g)] = float(val)
                    if record_grid:
                        fold_grid_records.append(
                            {'fold': k, 'stage': 'fine', 'gamma': float(g), 'll': float(val)}
                        )
            for g, total in fine_scores.items():
                if total > best_ll:
                    best_ll = total
                    best_params = {'gamma': float(g)}

        elif model_name == 'M2':
            coarse_g_base = [0.5, 1.0, 1.5, 2.5, 4.0, 8.0]
            coarse_k = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
            max_workers = int(os.environ.get('MODEL_COMP_MAX_WORKERS', os.cpu_count() or 1))
            coarse_scores = np.zeros((len(coarse_g_base), len(coarse_k)))
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers, initializer=_worker_init, initargs=(A, B, D)
            ) as exe:
                fut_map = {}
                for i, g_base in enumerate(coarse_g_base):
                    for j, k_val in enumerate(coarse_k):
                        fold_grid_evals += 1
                        fut = exe.submit(
                            _eval_m2_params_masked,
                            None,
                            None,
                            None,
                            float(g_base),
                            float(k_val),
                            ref_logs,
                            train_idx,
                        )
                        fut_map[fut] = (i, j, float(g_base), float(k_val))
                for fut in concurrent.futures.as_completed(fut_map):
                    i, j, gb, kv = fut_map[fut]
                    try:
                        val = fut.result()
                    except Exception:
                        val = -np.inf
                    coarse_scores[i, j] += float(val)
                    if record_grid:
                        fold_grid_records.append(
                            {
                                'fold': k,
                                'stage': 'coarse',
                                'gamma_base': gb,
                                'entropy_k': kv,
                                'll': float(val),
                            }
                        )

            best_flat = np.argmax(coarse_scores)
            bi, bj = np.unravel_index(best_flat, coarse_scores.shape)
            best_ll = coarse_scores[bi, bj]
            best_params = {'gamma_base': float(coarse_g_base[bi]), 'entropy_k': float(coarse_k[bj])}

            gi_lo = max(0, bi - 1)
            gi_hi = min(len(coarse_g_base) - 1, bi + 1)
            gb_lo = coarse_g_base[gi_lo]
            gb_hi = coarse_g_base[gi_hi]
            if bi == 0:
                gb_lo = max(0.1, coarse_g_base[0] * 0.5)
            if bi == len(coarse_g_base) - 1:
                gb_hi = coarse_g_base[-1] * 1.25

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
            fine_scores = {}
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers, initializer=_worker_init, initargs=(A, B, D)
            ) as exe:
                fut_map = {}
                for g_base in fine_gb:
                    for k_val in fine_k:
                        fold_grid_evals += 1
                        fut = exe.submit(
                            _eval_m2_params_masked,
                            None,
                            None,
                            None,
                            float(g_base),
                            float(k_val),
                            ref_logs,
                            train_idx,
                        )
                        fut_map[fut] = (float(g_base), float(k_val))
                for fut in concurrent.futures.as_completed(fut_map):
                    g_base, k_val = fut_map[fut]
                    try:
                        val = fut.result()
                    except Exception:
                        val = -np.inf
                    fine_scores[(g_base, k_val)] = float(val)
                    if record_grid:
                        fold_grid_records.append(
                            {
                                'fold': k,
                                'stage': 'fine',
                                'gamma_base': g_base,
                                'entropy_k': k_val,
                                'll': float(val),
                            }
                        )
            for (gb, kk), total in fine_scores.items():
                if total > best_ll:
                    best_ll = total
                    best_params = {'gamma_base': float(gb), 'entropy_k': float(kk)}

        elif model_name == 'M3':
            policies, num_actions_per_factor = _make_temp_agent_and_policies(A, B, D)
            # SMART CONSTRAINED SEARCH (same as non-CV version)
            gamma_vals = [1.0, 2.5, 5.0]
            xi_scale_hint = [0.5, 1.0, 2.0, 4.0]  # Expanded to include 4.0
            xi_scale_arm = [0.5, 1.0, 2.0]  # Removed 4.0
            
            from itertools import product
            num_profiles = len(M3_DEFAULTS['profiles'])
            
            candidates = []
            for gamma_p0 in gamma_vals:
                for gamma_p1 in gamma_vals:
                    for hint_scale in xi_scale_hint:
                        for arm_scale in xi_scale_arm:
                            gammas = [gamma_p0, gamma_p1]
                            xi_scales = [
                                [hint_scale, arm_scale, arm_scale],
                                [hint_scale, arm_scale, arm_scale]
                            ]
                            candidates.append((list(gammas), xi_scales))

            # Parallelize M3 grid evaluation across workers
            max_workers = int(os.environ.get('MODEL_COMP_MAX_WORKERS', os.cpu_count() or 1))
            candidate_scores = {}
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers, initializer=_worker_init, initargs=(A, B, D)
            ) as exe:
                fut_map = {}
                for gammas_c, xi_scales_c in candidates:
                    fold_grid_evals += 1
                    fut = exe.submit(
                        _eval_m3_params_per_profile_masked,
                        None, None, None,
                        gammas_c,
                        xi_scales_c,
                        ref_logs,
                        train_idx
                    )
                    fut_map[fut] = (gammas_c, xi_scales_c)
                
                for fut in concurrent.futures.as_completed(fut_map):
                    gammas_c, xi_scales_c = fut_map[fut]
                    try:
                        tr_ll = fut.result()
                    except Exception:
                        tr_ll = -np.inf
                    
                    candidate_scores[(tuple(gammas_c), tuple(tuple(x) for x in xi_scales_c))] = float(tr_ll)
                    
                    if record_grid:
                        fold_grid_records.append(
                            {
                                'fold': k,
                                'stage': 'grid',
                                'gamma_profile': json.dumps(gammas_c),
                                'xi_scales': json.dumps(xi_scales_c),
                                'll': float(tr_ll),
                            }
                        )
                    
                    if tr_ll > best_ll:
                        best_ll = tr_ll
                        best_params = {'gamma_profile': list(gammas_c), 'xi_scales_profile': xi_scales_c}

        else:
            raise ValueError(f"Unknown model for CV: {model_name}")

        if best_params is None:
            train_ll = float('-inf')
            test_ll = float('-inf')
            value_fn_best = None
        else:
            if model_name == 'M1':
                value_fn_best = make_value_fn(
                    'M1', C_reward_logits=M1_DEFAULTS['C_reward_logits'], gamma=best_params['gamma']
                )
            elif model_name == 'M2':

                def gamma_schedule(q, t, g_base=best_params['gamma_base'], k=best_params['entropy_k']):
                    p = np.clip(np.asarray(q, float), 1e-12, 1.0)
                    H = -(p * np.log(p)).sum()
                    return g_base / (1.0 + k * H)

                value_fn_best = make_value_fn(
                    'M2', C_reward_logits=M2_DEFAULTS['C_reward_logits'], gamma_schedule=gamma_schedule
                )
            else:
                policies, num_actions_per_factor = _make_temp_agent_and_policies(A, B, D)
                profiles = []
                if 'gamma_profile' in best_params and 'xi_scales_profile' in best_params:
                    gamma_profile = best_params['gamma_profile']
                    xi_scales_profile = best_params['xi_scales_profile']
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
                    for p in M3_DEFAULTS['profiles']:
                        profiles.append(dict(p))
                value_fn_best = make_value_fn(
                    'M3',
                    profiles=profiles,
                    Z=np.array(M3_DEFAULTS['Z']),
                    policies=policies,
                    num_actions_per_factor=num_actions_per_factor,
                )

            train_ll, _ = evaluate_ll_with_valuefn_masked(value_fn_best, A, B, D, ref_logs, train_idx)
            test_ll, _ = evaluate_ll_with_valuefn_masked(value_fn_best, A, B, D, ref_logs, test_idx)

        fold_entry = {
            'fold': k,
            'best_params': best_params,
            'train_ll': float(train_ll),
            'test_ll': float(test_ll),
            'train_idx': train_idx,
            'test_idx': test_idx,
            'grid_evals': fold_grid_evals,
        }

        rows = []
        if value_fn_best is not None:
            rows = _generate_trial_level_predictions(value_fn_best, A, B, D, ref_logs)
            role_mask = set(test_idx)
            for r in rows:
                r['fold'] = k
                r['role'] = 'test' if r['t'] in role_mask else 'train'
            trial_rows_all.extend(rows)
            train_rows = [r for r in rows if r['role'] == 'train']
            test_rows = [r for r in rows if r['role'] == 'test']
            if train_rows:
                fold_entry['train_acc'] = float(np.mean([r['accuracy'] for r in train_rows]))
            if test_rows:
                fold_entry['test_acc'] = float(np.mean([r['accuracy'] for r in test_rows]))
        fold_results.append(fold_entry)
        total_grid_evals += fold_grid_evals

        if record_grid and fold_grid_records and generator and run_id is not None and save_artifacts:
            grid_path = _grid_eval_path(generator, model_name, run_id, k, artifact_base_dir)
            fieldnames = sorted({key for rec in fold_grid_records for key in rec.keys()})
            rows_to_write = []
            for rec in fold_grid_records:
                row = {key: rec.get(key, '') for key in fieldnames}
                rows_to_write.append(row)
            _write_dict_csv(grid_path, fieldnames, rows_to_write, mode='w')

    runtime_sec = time.time() - run_start
    train_lls = np.array([fr['train_ll'] for fr in fold_results])
    test_lls = np.array([fr['test_ll'] for fr in fold_results])
    train_accs = np.array([fr.get('train_acc', np.nan) for fr in fold_results])
    test_accs = np.array([fr.get('test_acc', np.nan) for fr in fold_results])

    def _mean_std(arr):
        arr = arr.astype(float)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    mean_train_ll, std_train_ll = _mean_std(train_lls)
    mean_test_ll, std_test_ll = _mean_std(test_lls)
    mean_train_acc, std_train_acc = _mean_std(train_accs)
    mean_test_acc, std_test_acc = _mean_std(test_accs)
    best_train_ll = float(np.nanmax(train_lls)) if train_lls.size else float('nan')

    action_counts = Counter(ref_logs['action'])
    total_actions = sum(action_counts.values()) or 1
    action_distribution = {k: v / total_actions for k, v in action_counts.items()}
    beliefs = ref_logs.get('belief', [])
    if beliefs:
        entropy_vals = [compute_entropy(b) for b in beliefs if isinstance(b, (list, np.ndarray))]
        mean_entropy = float(np.mean(entropy_vals)) if entropy_vals else float('nan')
    else:
        mean_entropy = float('nan')
    reversal_count = len(find_reversals(ref_logs['context']))
    num_trials = len(ref_logs['action'])
    k_params = get_num_parameters(model_name)
    aic = 2 * k_params - 2 * mean_test_ll
    bic = k_params * np.log(max(1, num_trials)) - 2 * mean_test_ll

    summary = {
        'mean_train_ll': mean_train_ll,
        'std_train_ll': std_train_ll,
        'mean_test_ll': mean_test_ll,
        'std_test_ll': std_test_ll,
        'mean_train_acc': mean_train_acc,
        'std_train_acc': std_train_acc,
        'mean_test_acc': mean_test_acc,
        'std_test_acc': std_test_acc,
        'runtime_sec': runtime_sec,
        'grid_evals': total_grid_evals,
        'aic': aic,
        'bic': bic,
    }

    trial_csv = fold_csv = run_summary_csv = None
    if save_artifacts and generator is not None and run_id is not None:
        trial_csv = _trial_level_path(generator, model_name, run_id, artifact_base_dir)
        save_trial_level_csv(trial_rows_all, trial_csv)

        fold_csv = _fold_level_path(generator, model_name, run_id, artifact_base_dir)
        fold_rows = []
        for fr in fold_results:
            train_indices = [int(i) for i in fr['train_idx']]
            test_indices = [int(i) for i in fr['test_idx']]
            best_params_native = _to_native(fr['best_params'])
            fold_rows.append(
                {
                    'generator': generator,
                    'model': model_name,
                    'run_idx': run_id,
                    'fold': fr['fold'],
                    'train_indices': json.dumps(train_indices),
                    'test_indices': json.dumps(test_indices),
                    'train_ll': fr['train_ll'],
                    'test_ll': fr['test_ll'],
                    'train_acc': fr.get('train_acc', ''),
                    'test_acc': fr.get('test_acc', ''),
                    'grid_evals': fr['grid_evals'],
                    'best_params': json.dumps(best_params_native),
                }
            )
        _write_dict_csv(
            fold_csv,
            [
                'generator',
                'model',
                'run_idx',
                'fold',
                'train_indices',
                'test_indices',
                'train_ll',
                'test_ll',
                'train_acc',
                'test_acc',
                'grid_evals',
                'best_params',
            ],
            fold_rows,
            mode='w',
        )

        run_summary_csv = _run_summary_path(generator, model_name, artifact_base_dir)
        best_params_fold_native = [_to_native(fr['best_params']) for fr in fold_results]
        run_row = {
            'generator': generator,
            'model': model_name,
            'run_idx': run_id,
            'seed': seed if seed is not None else '',
            'runtime_sec': runtime_sec,
            'grid_evals': total_grid_evals,
            'mean_train_ll': mean_train_ll,
            'std_train_ll': std_train_ll,
            'mean_test_ll': mean_test_ll,
            'std_test_ll': std_test_ll,
            'mean_train_acc': mean_train_acc,
            'std_train_acc': std_train_acc,
            'mean_test_acc': mean_test_acc,
            'std_test_acc': std_test_acc,
            'best_params_per_fold': json.dumps(best_params_fold_native),
            'reversal_count': reversal_count,
            'action_distribution': json.dumps(_to_native(action_distribution)),
            'mean_belief_entropy': mean_entropy,
            'best_train_ll': best_train_ll,
            'aic': aic,
            'bic': bic,
        }
        _write_dict_csv(
            run_summary_csv,
            [
                'generator',
                'model',
                'run_idx',
                'seed',
                'runtime_sec',
                'grid_evals',
                'mean_train_ll',
                'std_train_ll',
                'mean_test_ll',
                'std_test_ll',
                'mean_train_acc',
                'std_train_acc',
                'mean_test_acc',
                'std_test_acc',
                'best_params_per_fold',
                'reversal_count',
                'action_distribution',
                'mean_belief_entropy',
                'best_train_ll',
                'aic',
                'bic',
            ],
            run_row,
            mode='a',
        )

    return {
        'generator': generator,
        'model': model_name,
        'run_idx': run_id,
        'seed': seed,
        'folds': fold_results,
        'summary': summary,
        'trial_csv': trial_csv,
        'fold_csv': fold_csv,
        'run_summary_csv': run_summary_csv,
    }


def _generate_trial_level_predictions(value_fn, A, B, D, ref_logs):
    """Run teacher-forced predictions for a model and return per-trial records.

    Returns list of dict rows with per-trial fields used for CSV export.
    """
    from src.models.agent_wrapper import AgentRunnerWithLL
    from src.utils.helpers import find_reversals

    runner = AgentRunnerWithLL(A, B, D, value_fn,
                               OBSERVATION_HINTS, OBSERVATION_REWARDS,
                               OBSERVATION_CHOICES, ACTION_CHOICES,
                               reward_mod_idx=1)

    T = len(ref_logs['action'])
    rows = []

    reversals = set(find_reversals(ref_logs['context']))

    # initial observation
    initial_obs_labels = ['null', 'null', 'observe_start']
    obs_ids = runner.obs_labels_to_ids(initial_obs_labels)

    for t in range(T):
        # Infer states using current obs
        qs = runner.agent.infer_states(obs_ids)
        q_context = qs[0].copy()

        # Update value_fn derived params for this trial
        C_t, E_t, gamma_t = value_fn(q_context, t)
        runner.agent.C[1] = C_t
        if E_t is not None and len(E_t) == len(runner.agent.policies):
            runner.agent.E = E_t
        runner.agent.gamma = float(gamma_t)

        # Infer policies and get posterior over policies
        q_pi, efe = runner.agent.infer_policies()

        # Compute action probabilities by summing q_pi over policies whose
        # first action for choice factor matches each action index
        num_actions = len(ACTION_CHOICES)
        action_probs = np.zeros(num_actions)
        for pi_idx, policy in enumerate(runner.agent.policies):
            a_idx = int(policy[0, 1])
            action_probs[a_idx] += float(q_pi[pi_idx])

        # Normalize (should already sum to 1)
        if action_probs.sum() > 0:
            action_probs = action_probs / action_probs.sum()

        pred_a_idx = int(np.argmax(action_probs))
        pred_action = ACTION_CHOICES[pred_a_idx]

        # Generator's actual action for this trial
        gen_action = ref_logs['action'][t]
        try:
            gen_a_idx = ACTION_CHOICES.index(gen_action)
            gen_action_prob = float(action_probs[gen_a_idx])
        except ValueError:
            gen_a_idx = None
            gen_action_prob = 0.0

        # log-likelihood for this trial (log prob assigned to the taken action)
        ll_t = np.log(gen_action_prob + 1e-16)

        # accuracy: whether predicted action chooses the currently better arm
        # For volatile/stable contexts, we need to know which arm is actually better
        current_better_arm = ref_logs.get('current_better_arm', [None]*T)[t]
        
        if current_better_arm is not None:
            # New volatile/stable task: check against current_better_arm
            if current_better_arm == 'left':
                acc = 1 if ('left' in pred_action) else 0
            elif current_better_arm == 'right':
                acc = 1 if ('right' in pred_action) else 0
            else:
                acc = 0
        else:
            # Fallback for old left_better/right_better task
            true_context = ref_logs['context'][t]
            if true_context == 'left_better':
                acc = 1 if ('left' in pred_action) else 0
            elif true_context == 'right_better':
                acc = 1 if ('right' in pred_action) else 0
            else:
                acc = 0

        # flags
        is_reversal = int(t in reversals)
        hint_flag = 1 if ref_logs.get('hint_label', [None]*T)[t] != 'null' else 0
        is_reward = 1 if ref_logs.get('reward_label', [None]*T)[t] == 'observe_reward' else 0
        is_loss = 1 if ref_logs.get('reward_label', [None]*T)[t] == 'observe_loss' else 0

        row = {
            't': t,
            'true_context': true_context,
            'gen_action': gen_action,
            'hint_label': ref_logs.get('hint_label', ['null']*T)[t],
            'reward_label': ref_logs.get('reward_label', ['null']*T)[t],
            'choice_label': ref_logs.get('choice_label', ['observe_start']*T)[t],
            'predicted_action': pred_action,
            'action_probs': action_probs.tolist(),
            'belief_context': q_context.tolist(),
            'gamma': float(runner.agent.gamma),
            'll': float(ll_t),
            'accuracy': int(acc),
            'is_reversal': is_reversal,
            'hint_flag': hint_flag,
            'is_reward': is_reward,
            'is_loss': is_loss
        }

        rows.append(row)

        # advance obs to next trial using generator's recorded obs
        if 'hint_label' in ref_logs and ref_logs['hint_label']:
            next_obs = [ref_logs['hint_label'][t], ref_logs['reward_label'][t], ref_logs['choice_label'][t]]
        else:
            next_obs = ['null', ref_logs['reward_label'][t], ref_logs['choice_label'][t]]
        obs_ids = runner.obs_labels_to_ids(next_obs)

    return rows


def save_trial_level_csv(rows, out_path):
    """Save trial-level prediction rows to CSV.

    `rows` is a list of dicts returned by `_generate_trial_level_predictions`
    with 'fold' and 'role' keys populated upstream.
    """
    import csv
    import json as _json

    header = [
        't',
        'fold',
        'role',
        'true_context',
        'gen_action',
        'hint_label',
        'reward_label',
        'choice_label',
        'predicted_action',
        'action_probs',
        'belief_context',
        'gamma',
        'll',
        'accuracy',
        'is_reversal',
        'hint_flag',
        'is_reward',
        'is_loss'
    ]
    _ensure_parent(out_path)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow([
                r['t'],
                r.get('fold', ''),
                r.get('role', ''),
                r['true_context'],
                r['gen_action'],
                r['hint_label'],
                r['reward_label'],
                r['choice_label'],
                r['predicted_action'],
                _json.dumps(r['action_probs']),
                _json.dumps(r['belief_context']),
                f"{r['gamma']:.6f}",
                f"{r['ll']:.6f}",
                r['accuracy'],
                r['is_reversal'],
                r['hint_flag'],
                r['is_reward'],
                r['is_loss']
            ])


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


def write_per_run_metrics(per_run_stats, base_dir=MODEL_RECOVERY_BASE):
    """Persist per-run metrics aggregated across models."""
    fields = [
        'generator',
        'model',
        'run_idx',
        'seed',
        'mean_test_ll',
        'std_test_ll',
        'mean_test_acc',
        'std_test_acc',
        'aic',
        'bic',
        'runtime_sec',
        'grid_evals',
    ]
    path = os.path.join(base_dir, 'per_run_metrics.csv')
    _write_dict_csv(path, fields, per_run_stats, mode='w')
    return path


def write_confusion_tables(per_run_stats, models, base_dir=MODEL_RECOVERY_BASE):
    """Create confusion matrices (mean & SE) for LL, accuracy, AIC, and BIC."""
    import csv

    generators = sorted({row['generator'] for row in per_run_stats})
    metrics = {
        'll': 'mean_test_ll',
        'acc': 'mean_test_acc',
        'aic': 'aic',
        'bic': 'bic',
    }

    for metric_key, field in metrics.items():
        mean_path = _confusion_matrix_path(metric_key, 'mean', base_dir)
        se_path = _confusion_matrix_path(metric_key, 'se', base_dir)
        _ensure_parent(mean_path)
        _ensure_parent(se_path)
        with open(mean_path, 'w', newline='') as mean_f, open(se_path, 'w', newline='') as se_f:
            mean_writer = csv.writer(mean_f)
            se_writer = csv.writer(se_f)
            header = ['generator'] + list(models)
            mean_writer.writerow(header)
            se_writer.writerow(header)
            for gen in generators:
                mean_row = [gen]
                se_row = [gen]
                for model in models:
                    vals = [
                        float(row[field])
                        for row in per_run_stats
                        if row['generator'] == gen and row['model'] == model
                    ]
                    if vals:
                        arr = np.array(vals, dtype=float)
                        mean_val = float(np.mean(arr))
                        if len(arr) > 1:
                            se_val = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
                        else:
                            se_val = 0.0
                        mean_row.append(mean_val)
                        se_row.append(se_val)
                    else:
                        mean_row.append('')
                        se_row.append('')
                mean_writer.writerow(mean_row)
                se_writer.writerow(se_row)


def write_experiment_metadata(metadata, base_dir=MODEL_RECOVERY_BASE):
    """Write experiment metadata JSON for reproducibility."""
    path = _metadata_path(base_dir)
    _ensure_parent(path)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(metadata, fh, indent=2)
    return path
