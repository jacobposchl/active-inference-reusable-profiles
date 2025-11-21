
"""K-fold cross-validation for model recovery.

Re-simulates the same generators/runs used in the recovery tests, fits each
model on training folds (grid search over the same small parameter sets),
evaluates held-out runs, and reports per-model test LL and paired ΔELPD
with standard errors across folds.
"""
import os
import logging
import numpy as np
from tqdm import tqdm
from src.utils.recovery_helpers import generate_all_runs, fit_model_on_runs, evaluate_on_test

# Using simple print statements for progress feedback (no logger)
def _print_info(*args, **kwargs):
    print("[cv_recovery]", *args, **kwargs)

def _print_debug(*args, **kwargs):
    print("[cv_recovery DEBUG]", *args, **kwargs)

from config.experiment_config import *


def kfold_cv(generators=['M1','M2','M3','egreedy','softmax'], runs_per_generator=10, num_trials=80, seed=1, reversal_interval=40, K=5):
    _print_info(f"Generating all reference runs: generators={generators} runs_per_generator={runs_per_generator} num_trials={num_trials} K={K}")
    A, B, D, refs = generate_all_runs(generators, runs_per_generator, num_trials, seed, reversal_interval)
    N = len(refs)
    idx = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)

    candidate_models = ['M1','M2','M3']
    per_fold_test_ll = {m: [] for m in candidate_models}

    for k in range(K):
        _print_info(f"Starting fold {k+1}/{K}: test_size={len(folds[k])}")
        test_idx = folds[k]
        train_idx = np.hstack([folds[i] for i in range(K) if i != k])
        train_refs = [refs[i]['ref_logs'] for i in train_idx]
        test_refs = [refs[i]['ref_logs'] for i in test_idx]

        # Fit each model on training refs
        fitted_params = {}
        for m in candidate_models:
            _print_debug(f"Fitting model {m} on {len(train_refs)} training runs")
            fitted_params[m] = fit_model_on_runs(m, A, B, D, train_refs)
            _print_debug(f"Fitted params for {m}: {fitted_params[m]}")

        # Evaluate on test
        for m in candidate_models:
            test_ll = evaluate_on_test(m, A, B, D, fitted_params[m], test_refs)
            per_fold_test_ll[m].append(test_ll)
            _print_info(f"Fold {k+1}: model {m} test LL = {test_ll:.3f}")

        # Save fold checkpoint
        csv_dir = os.path.join('results', 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        fold_path = os.path.join(csv_dir, f'cv_recovery_fold_{k+1}.csv')
        with open(fold_path, 'w', newline='') as f:
            import csv as _csv
            writer = _csv.writer(f)
            header = ['model', 'test_ll']
            writer.writerow(header)
            for m in candidate_models:
                writer.writerow([m, per_fold_test_ll[m][-1]])
        _print_info(f"Saved fold {k+1} checkpoint: {fold_path}")

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
