import os
import csv
import logging
import numpy as np
from src.utils.ll_eval import compute_sequence_ll_for_model, evaluate_ll_with_valuefn
from src.utils.model_utils import create_model
from src.utils.simulate import simulate_baseline_run
from src.models import build_A, build_B, build_D, make_value_fn
from src.environment import TwoArmedBandit
from src.utils.recovery_helpers import fit_model_on_runs, evaluate_on_test
from config.experiment_config import *


def run_model_recovery(generators=None, runs_per_generator=20, num_trials=80, seed=123, reversal_interval=None):
    if generators is None:
        generators = ['M1', 'M2', 'M3', 'egreedy', 'softmax']

    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)

    candidate_models = ['M1', 'M2', 'M3']
    per_run_rows = []
    confusion = {g: {m: 0 for m in candidate_models} for g in generators}

    for gen in generators:
        print(f"\nSimulating generator: {gen}")
        for r in range(runs_per_generator):
            run_seed = int(seed + r + (abs(hash(gen)) % 1000))

            if reversal_interval is not None:
                reversal_schedule = [i for i in range(reversal_interval, num_trials, reversal_interval)]
            else:
                reversal_schedule = DEFAULT_REVERSAL_SCHEDULE

            env = TwoArmedBandit(probability_hint=PROBABILITY_HINT,
                                 probability_reward=PROBABILITY_REWARD,
                                 reversal_schedule=reversal_schedule)

            if gen in ('M1', 'M2', 'M3'):
                value_fn = create_model(gen, A, B, D)
                from src.models import AgentRunnerWithLL, run_episode_with_ll
                runner = AgentRunnerWithLL(A, B, D, value_fn,
                                           OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                           OBSERVATION_CHOICES, ACTION_CHOICES,
                                           reward_mod_idx=1)
                try:
                    runner.agent.gamma = 8.0
                except Exception as e:
                    logging.warning("Could not set runner.agent.gamma to 8.0: %s", e)
                ref_logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)

            elif gen in ('egreedy', 'softmax'):
                ref_logs = simulate_baseline_run(env, policy_type=gen, T=num_trials, seed=run_seed,
                                                 epsilon=0.1, temp=1.0, alpha=0.1)
            else:
                value_fn = create_model('M3', A, B, D)
                from src.models import AgentRunnerWithLL, run_episode_with_ll
                runner = AgentRunnerWithLL(A, B, D, value_fn,
                                           OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                           OBSERVATION_CHOICES, ACTION_CHOICES,
                                           reward_mod_idx=1)
                ref_logs = run_episode_with_ll(runner, env, T=num_trials, verbose=False)

            ll_by_model = {}
            for model_name in candidate_models:
                ll_seq = compute_sequence_ll_for_model(model_name, A, B, D, ref_logs)
                total_ll = float(np.sum(ll_seq))
                ll_by_model[model_name] = total_ll

            best_model = max(ll_by_model, key=ll_by_model.get)
            confusion[gen][best_model] += 1

            per_run_rows.append({
                'generator': gen,
                'run_idx': r,
                'seed': run_seed,
                **{f'll_{m}': ll_by_model[m] for m in candidate_models},
                'winner': best_model
            })

    csv_dir = os.path.join('results', 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    rows_path = os.path.join(csv_dir, f'model_recovery_per_run_seed{seed}.csv')
    with open(rows_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['generator', 'run_idx', 'seed'] + [f'll_{m}' for m in candidate_models] + ['winner']
        writer.writerow(header)
        for row in per_run_rows:
            writer.writerow([row['generator'], row['run_idx'], row['seed']] + [row[f'll_{m}'] for m in candidate_models] + [row['winner']])

    conf_path = os.path.join(csv_dir, f'model_recovery_confusion_seed{seed}.csv')
    with open(conf_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['generator'] + candidate_models
        writer.writerow(header)
        for gen in generators:
            writer.writerow([gen] + [confusion[gen][m] for m in candidate_models])

    print(f"\nModel recovery per-run CSV saved: {rows_path}")
    print(f"Confusion matrix saved: {conf_path}")

    return confusion, per_run_rows


def run_model_recovery_with_fitting(generators=None, runs_per_generator=10, num_trials=80, seed=123, reversal_interval=None):
    if generators is None:
        generators = ['M1', 'M2', 'M3', 'egreedy', 'softmax']

    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)

    candidate_models = ['M1', 'M2', 'M3']
    confusion = {g: {m: 0 for m in candidate_models} for g in generators}
    per_run_rows = []

    for gen in generators:
        print(f"\nSimulating generator: {gen}")
        for r in range(runs_per_generator):
            run_seed = int(seed + r + (abs(hash(gen)) % 1000))

            if reversal_interval is not None:
                reversal_schedule = [i for i in range(reversal_interval, num_trials, reversal_interval)]
            else:
                reversal_schedule = DEFAULT_REVERSAL_SCHEDULE

            env = TwoArmedBandit(probability_hint=PROBABILITY_HINT,
                                 probability_reward=PROBABILITY_REWARD,
                                 reversal_schedule=reversal_schedule)

            if gen in ('M1', 'M2', 'M3'):
                value_fn_gen = create_model(gen, A, B, D)
                from src.models import AgentRunnerWithLL, run_episode_with_ll
                runner_gen = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                               OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                               OBSERVATION_CHOICES, ACTION_CHOICES,
                                               reward_mod_idx=1)
                try:
                    runner_gen.agent.gamma = 8.0
                except Exception as e:
                    logging.warning("Could not set runner_gen.agent.gamma to 8.0: %s", e)
                ref_logs = run_episode_with_ll(runner_gen, env, T=num_trials, verbose=False)
            elif gen in ('egreedy', 'softmax'):
                ref_logs = simulate_baseline_run(env, policy_type=gen, T=num_trials, seed=run_seed,
                                                 epsilon=0.1, temp=1.0, alpha=0.1)
            else:
                value_fn_gen = create_model('M3', A, B, D)
                from src.models import AgentRunnerWithLL, run_episode_with_ll
                runner_gen = AgentRunnerWithLL(A, B, D, value_fn_gen,
                                               OBSERVATION_HINTS, OBSERVATION_REWARDS,
                                               OBSERVATION_CHOICES, ACTION_CHOICES,
                                               reward_mod_idx=1)
                ref_logs = run_episode_with_ll(runner_gen, env, T=num_trials, verbose=False)

            fitted_ll = {}
            fitted_params = {}

            # Use shared helpers to fit each candidate model on the single ref_logs
            fitted_ll = {}
            fitted_params = {}
            for m in candidate_models:
                best_params = fit_model_on_runs(m, A, B, D, [ref_logs])
                fitted_params[m] = best_params
                fitted_ll[m] = evaluate_on_test(m, A, B, D, best_params, [ref_logs])

            winner = max(fitted_ll, key=fitted_ll.get)
            confusion[gen][winner] += 1

            per_run_rows.append({
                'generator': gen,
                'run_idx': r,
                'seed': run_seed,
                **{f'fitted_ll_{m}': fitted_ll[m] for m in candidate_models},
                'winner': winner,
                'fitted_params': fitted_params
            })

    csv_dir = os.path.join('results', 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    rows_path = os.path.join(csv_dir, f'model_recovery_fitted_per_run_seed{seed}.csv')
    with open(rows_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['generator', 'run_idx', 'seed'] + [f'fitted_ll_{m}' for m in candidate_models] + ['winner']
        writer.writerow(header)
        for row in per_run_rows:
            writer.writerow([row['generator'], row['run_idx'], row['seed']] + [row[f'fitted_ll_{m}'] for m in candidate_models] + [row['winner']])

    conf_path = os.path.join(csv_dir, f'model_recovery_fitted_confusion_seed{seed}.csv')
    with open(conf_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['generator'] + candidate_models
        writer.writerow(header)
        for gen in generators:
            writer.writerow([gen] + [confusion[gen][m] for m in candidate_models])

    print(f"\nFitted model recovery per-run CSV saved: {rows_path}")
    print(f"Fitted confusion matrix saved: {conf_path}")
    return confusion, per_run_rows
