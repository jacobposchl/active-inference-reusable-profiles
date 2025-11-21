"""K-fold cross-validation for model recovery.

Re-simulates the same generators/runs used in the recovery tests, fits each
model on training folds (grid search over the same small parameter sets),
evaluates held-out runs, and reports per-model test LL and paired ΔELPD
with standard errors across folds.
"""
import os
import numpy as np
from tqdm import tqdm
from src.experiments.model_comparison import (
    create_model, simulate_baseline_run, evaluate_ll_with_valuefn,
    build_A, build_B, build_D, run_episode_with_ll, make_value_fn, AgentRunnerWithLL
)
from config.experiment_config import *
from src.environment import TwoArmedBandit


def generate_all_runs(generators, runs_per_generator, num_trials, seed, reversal_interval=None):
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)

    refs = []  # list of dicts: {'gen':, 'run_idx':, 'ref_logs': }
    for gen in generators:
        for r in range(runs_per_generator):
            run_seed = int(seed + r + (abs(hash(gen)) % 1000))
            if reversal_interval is not None:
                reversal_schedule = [i for i in range(reversal_interval, num_trials, reversal_interval)]
            else:
                reversal_schedule = DEFAULT_REVERSAL_SCHEDULE

            env = TwoArmedBandit(probability_hint=PROBABILITY_HINT, probability_reward=PROBABILITY_REWARD,
                                 reversal_schedule=reversal_schedule)

            if gen in ('M1', 'M2', 'M3'):
                value_fn = create_model(gen, A, B, D)
                runner = AgentRunnerWithLL(A, B, D, value_fn,
                                           OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                           OBSERVATION_CHOICES, ACTION_CHOICES,
                                           reward_mod_idx=1)
                try:
                    runner.agent.gamma = 8.0
                except Exception:
                    pass
                ref_logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)
            elif gen in ('egreedy', 'softmax'):
                ref_logs = simulate_baseline_run(env, policy_type=gen, T=num_trials, seed=run_seed,
                                                 epsilon=0.1, temp=1.0, alpha=0.1)
            else:
                value_fn = create_model('M3', A, B, D)
                runner = AgentRunnerWithLL(A, B, D, value_fn,
                                           OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                           OBSERVATION_CHOICES, ACTION_CHOICES,
                                           reward_mod_idx=1)
                ref_logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)

            refs.append({'gen': gen, 'run_idx': r, 'seed': run_seed, 'ref_logs': ref_logs})

    return A, B, D, refs


def fit_model_on_runs(model_name, A, B, D, ref_logs_list):
    """Fit model via small grid search on provided reference logs (list)."""
    best_ll = -np.inf
    best_params = None
    # M1
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
    # Build value_fn from params
    if model_name == 'M1':
        value_fn = make_value_fn('M1', C_reward_logits=M1_DEFAULTS['C_reward_logits'], gamma=params['gamma'])
    elif model_name == 'M2':
        def gamma_schedule(q, t, g_base=params['gamma_base'], k=params['entropy_k']):
            p = np.clip(np.asarray(q, float), 1e-12, 1.0)
            H = -(p * np.log(p)).sum()
            return g_base / (1.0 + k * H)
        value_fn = make_value_fn('M2', C_reward_logits=M2_DEFAULTS['C_reward_logits'], gamma_schedule=gamma_schedule)
    else:
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
        profiles = []
        for p in M3_DEFAULTS['profiles']:
            prof = dict(p)
            prof['gamma'] = params['gamma']
            prof['xi_logits'] = (np.array(p['xi_logits'], float) * params['xi_scale']).tolist()
            profiles.append(prof)
        value_fn = make_value_fn('M3', profiles=profiles, Z=np.array(M3_DEFAULTS['Z']), policies=policies, num_actions_per_factor=num_actions_per_factor)

    # Evaluate
    total = 0.0
    for ref in ref_logs_list:
        ll, _ = evaluate_ll_with_valuefn(value_fn, A, B, D, ref)
        total += ll
    return total


def kfold_cv(generators=['M1','M2','M3','egreedy','softmax'], runs_per_generator=10, num_trials=80, seed=1, reversal_interval=40, K=5):
    A, B, D, refs = generate_all_runs(generators, runs_per_generator, num_trials, seed, reversal_interval)
    N = len(refs)
    idx = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)

    candidate_models = ['M1','M2','M3']
    per_fold_test_ll = {m: [] for m in candidate_models}

    for k in range(K):
        test_idx = folds[k]
        train_idx = np.hstack([folds[i] for i in range(K) if i != k])
        train_refs = [refs[i]['ref_logs'] for i in train_idx]
        test_refs = [refs[i]['ref_logs'] for i in test_idx]

        # Fit each model on training refs
        fitted_params = {}
        for m in candidate_models:
            fitted_params[m] = fit_model_on_runs(m, A, B, D, train_refs)

        # Evaluate on test
        for m in candidate_models:
            test_ll = evaluate_on_test(m, A, B, D, fitted_params[m], test_refs)
            per_fold_test_ll[m].append(test_ll)

    # Aggregate and compute ΔELPD-like stats
    results = {}
    for m in candidate_models:
        arr = np.array(per_fold_test_ll[m])
        results[m] = {'fold_mean': arr.mean(), 'fold_se': arr.std(ddof=1)/np.sqrt(K), 'folds': arr}

    # Paired differences across folds
    diffs = {}
    for i in range(len(candidate_models)):
        for j in range(i+1, len(candidate_models)):
            mi = candidate_models[i]
            mj = candidate_models[j]
            d = np.array(per_fold_test_ll[mi]) - np.array(per_fold_test_ll[mj])
            diffs[f"{mi}_vs_{mj}"] = {'mean': d.mean(), 'se': d.std(ddof=1)/np.sqrt(K), 'folds': d}

    return results, diffs


if __name__ == '__main__':
    res, diffs = kfold_cv()
    print("Per-model CV results:")
    for m, v in res.items():
        print(m, v['fold_mean'], v['fold_se'])
    print("\nPaired diffs:")
    for k, v in diffs.items():
        print(k, v['mean'], v['se'])
