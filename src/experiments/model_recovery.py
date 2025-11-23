
"""
Experiment entry point: K-fold cross-validation for model recovery.

Overview
--------
The script coordinates four stages:
1. Generate reference trajectories with multiple generators (M1/M2/M3/baselines).
2. Fit each candidate model on K-1 folds via grid searches (see recovery_helpers).
3. Evaluate the fitted models on held-out runs via teacher forcing.
4. Aggregate per-fold log-likelihoods and paired deltas, writing CSV artifacts.

Progress feedback uses nested `tqdm` progress bars: the outer bar tracks folds,
and inner bars display status for per-model fitting/evaluation.
"""
import argparse
import os
import logging
import shutil
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.recovery_helpers import (
    generate_all_runs,
    fit_model_on_runs,
    cv_fit_single_run,
    write_per_run_metrics,
    write_confusion_tables,
    write_experiment_metadata,
    _make_temp_agent_and_policies,
    MODEL_RECOVERY_BASE,
)
from src.utils.ll_eval import evaluate_ll_with_valuefn
from src.models import make_value_fn

from config.experiment_config import *


def kfold_cv(
    generators=('M1', 'M2', 'M3', 'egreedy', 'softmax'),
    runs_per_generator=10,
    num_trials=80,
    seed=1,
    reversal_interval=40,
    K=5,
):
    """
    Perform K-fold cross-validated model recovery.

    Parameters
    ----------
    generators : sequence of str
        Behavioral generators used to produce reference trajectories.
    runs_per_generator : int
        Number of independent runs simulated for each generator.
    num_trials : int
        Number of bandit trials per run.
    seed : int
        Random seed controlling generation and fold shuffling.
    reversal_interval : int
        Interval at which latent context reverses; if None, uses default schedule.
    K : int
        Number of folds for cross-validation.

    Returns
    -------
    results : dict
        Mapping model name -> {'fold_mean', 'fold_se', 'folds'}
    diffs : dict
        Pairwise log-likelihood differences between models with SEs.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "Generating reference runs (generators=%s runs_per_generator=%s num_trials=%s K=%s)",
        generators,
        runs_per_generator,
        num_trials,
        K,
    )
    A, B, D, refs = generate_all_runs(generators, runs_per_generator, num_trials, seed, reversal_interval)
    logger.info(f"Generated {len(refs)} reference runs")
    N = len(refs)
    idx = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)

    candidate_models = ['M1', 'M2', 'M3']
    per_fold_test_ll = {m: [] for m in candidate_models}

    fold_bar = tqdm(range(K), desc="Folds", unit="fold")
    for k in fold_bar:
        test_idx = folds[k]
        train_idx = np.hstack([folds[i] for i in range(K) if i != k])
        train_refs = [refs[i]['ref_logs'] for i in train_idx]
        test_refs = [refs[i]['ref_logs'] for i in test_idx]
        fold_bar.set_postfix(test_size=len(test_idx))

        fitted_params = {}
        fit_bar = tqdm(candidate_models, desc=f"Fold {k+1} fit", leave=False)
        for m in fit_bar:
            fit_bar.set_postfix(model=m)
            fitted_params[m] = fit_model_on_runs(m, A, B, D, train_refs)

        test_refs_full = [refs[i] for i in test_idx]
        eval_bar = tqdm(candidate_models, desc=f"Fold {k+1} eval", leave=False)
        for m in eval_bar:
            eval_bar.set_postfix(model=m)

            # build value function for this model using fitted params
            params = fitted_params[m]
            # Helper: construct value_fn similar to recovery_helpers.evaluate_on_test
            if m == 'M1':
                value_fn = make_value_fn('M1', C_reward_logits=M1_DEFAULTS['C_reward_logits'], gamma=params['gamma'])
            elif m == 'M2':
                def gamma_schedule(q, t, g_base=params['gamma_base'], k=params['entropy_k']):
                    p = np.clip(np.asarray(q, float), 1e-12, 1.0)
                    H = -(p * np.log(p)).sum()
                    return g_base / (1.0 + k * H)
                value_fn = make_value_fn('M2', C_reward_logits=M2_DEFAULTS['C_reward_logits'], gamma_schedule=gamma_schedule)
            else:
                # M3: need policies and per-profile param wiring
                policies, num_actions_per_factor = _make_temp_agent_and_policies(A, B, D)
                profiles = []
                if params is None:
                    params = {}

                if 'gamma' in params and 'xi_scale' in params:
                    # Legacy format: apply same gamma/scale to all profiles.
                    for p in M3_DEFAULTS['profiles']:
                        prof = dict(p)
                        prof['gamma'] = params['gamma']
                        prof['xi_logits'] = (np.array(p['xi_logits'], float) * params['xi_scale']).tolist()
                        profiles.append(prof)
                elif 'gamma_profile' in params and 'xi_scales_profile' in params:
                    # Modern format: per-profile gammas and action-bias scaling.
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
                    # Fallback to defaults if params missing.
                    for p in M3_DEFAULTS['profiles']:
                        profiles.append(dict(p))

                value_fn = make_value_fn('M3', profiles=profiles, Z=np.array(M3_DEFAULTS['Z']), policies=policies, num_actions_per_factor=num_actions_per_factor)

            # Evaluate each test run, collect per-run LLs and per-trial sequences
            per_run_records = []
            from collections import defaultdict
            gen_totals = defaultdict(float)
            gen_counts = defaultdict(int)
            # store per-run ll by run_idx for paired diffs
            per_run_by_idx = {}

            for ref in test_refs_full:
                total_ll, ll_seq = evaluate_ll_with_valuefn(value_fn, A, B, D, ref['ref_logs'])
                per_run_records.append({'model': m, 'gen': ref['gen'], 'run_idx': ref['run_idx'], 'total_ll': float(total_ll), 'n_trials': len(ll_seq)})
                gen_totals[ref['gen']] += float(total_ll)
                gen_counts[ref['gen']] += 1
                per_run_by_idx[ref['run_idx']] = float(total_ll)

            # save per-run CSV for this fold and model
            csv_dir = os.path.join('results', 'csv')
            os.makedirs(csv_dir, exist_ok=True)
            per_run_path = os.path.join(csv_dir, f'cv_recovery_fold_{k+1}_{m}_per_run.csv')
            with open(per_run_path, 'w', newline='') as f:
                import csv as _csv
                writer = _csv.writer(f)
                writer.writerow(['model', 'gen', 'run_idx', 'total_ll', 'n_trials'])
                for r in per_run_records:
                    writer.writerow([r['model'], r['gen'], r['run_idx'], f"{r['total_ll']:.6f}", r['n_trials']])

            # compute per-generator aggregates and save
            per_gen_path = os.path.join(csv_dir, f'cv_recovery_fold_{k+1}_{m}_per_gen.csv')
            with open(per_gen_path, 'w', newline='') as f:
                import csv as _csv
                writer = _csv.writer(f)
                writer.writerow(['model', 'gen', 'total_ll', 'n_runs', 'mean_ll_per_run', 'mean_ll_per_trial'])
                for g in gen_totals:
                    total = gen_totals[g]
                    n = gen_counts[g]
                    mean_run = total / n if n > 0 else float('nan')
                    # estimate trials per run from first matching ref
                    trials_per_run = next((r['n_trials'] for r in per_run_records if r['gen'] == g), 0)
                    mean_trial = mean_run / trials_per_run if trials_per_run else float('nan')
                    writer.writerow([m, g, f"{total:.6f}", n, f"{mean_run:.6f}", f"{mean_trial:.6f}"])

            # append aggregated total LL for compatibility with downstream code
            total_all = sum([r['total_ll'] for r in per_run_records])
            per_fold_test_ll[m].append(total_all)
            eval_bar.write(f"[Fold {k+1}] {m} test LL = {total_all:.3f}")

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
        fold_bar.write(f"Saved fold {k+1} checkpoint: {fold_path}")

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


def per_run_cv(
    generators=('M1', 'M2', 'M3', 'egreedy', 'softmax'),
    runs_per_generator=1,
    num_trials=20,
    seed=1,
    reversal_interval=40,
    K=2,
    artifact_base=None,
    record_grid=False,
    clean_artifacts=False,
    save_artifacts=True,
):
    """
    Run `cv_fit_single_run` for every generator/run pair and emit the same
    artifact hierarchy as `smoke_cv_demo.py`.

    Parameters mirror `kfold_cv`, with the addition of artifact controls.
    Returns the list of per-run statistics rows written to disk.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "Generating reference runs for per-run CV (generators=%s runs_per_generator=%s num_trials=%s K=%s)",
        generators,
        runs_per_generator,
        num_trials,
        K,
    )
    A, B, D, refs = generate_all_runs(generators, runs_per_generator, num_trials, seed, reversal_interval)
    artifact_dir = artifact_base or os.path.join(MODEL_RECOVERY_BASE, 'full_cv')
    if clean_artifacts:
        shutil.rmtree(artifact_dir, ignore_errors=True)
    os.makedirs(artifact_dir, exist_ok=True)

    candidate_models = ['M1', 'M2', 'M3']
    per_run_stats = []

    run_bar = tqdm(refs, desc="Reference runs", unit="run")
    for ref in run_bar:
        model_bar = tqdm(candidate_models, desc=f"Run {ref['run_idx']} models", leave=False)
        for model_name in model_bar:
            model_bar.set_postfix(model=model_name, gen=ref['gen'])
            result = cv_fit_single_run(
                model_name,
                A,
                B,
                D,
                ref['ref_logs'],
                K=K,
                run_id=ref['run_idx'],
                generator=ref['gen'],
                seed=ref['seed'],
                artifact_base_dir=artifact_dir,
                save_artifacts=save_artifacts,
                record_grid=record_grid,
            )
            summary = result['summary']
            entry = {
                'generator': ref['gen'],
                'model': model_name,
                'run_idx': ref['run_idx'],
                'seed': ref['seed'],
                'mean_test_ll': summary['mean_test_ll'],
                'std_test_ll': summary['std_test_ll'],
                'mean_test_acc': summary['mean_test_acc'],
                'std_test_acc': summary['std_test_acc'],
                'aic': summary['aic'],
                'bic': summary['bic'],
                'runtime_sec': summary['runtime_sec'],
                'grid_evals': summary['grid_evals'],
            }
            per_run_stats.append(entry)

    if not per_run_stats:
        logger.warning("No per-run stats were generated.")
        return []

    write_per_run_metrics(per_run_stats, base_dir=artifact_dir)
    write_confusion_tables(per_run_stats, candidate_models, base_dir=artifact_dir)
    metadata = {
        'mode': 'per_run',
        'generators': generators,
        'runs_per_generator': runs_per_generator,
        'num_trials': num_trials,
        'seed': seed,
        'reversal_interval': reversal_interval,
        'folds': K,
        'candidate_models': candidate_models,
        'artifact_base': artifact_dir,
    }
    write_experiment_metadata(metadata, base_dir=artifact_dir)
    logger.info("Per-run CV artifacts stored under %s", artifact_dir)
    return per_run_stats


def _parse_args():
    """CLI helper so the experiment can be launched directly."""
    parser = argparse.ArgumentParser(
        description="Run K-fold model recovery with configurable generators and settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generators",
        type=str,
        default="M1,M2,M3,egreedy,softmax",
        help="Comma-separated list of behavioral generators to simulate.",
    )
    parser.add_argument("--runs-per-generator", type=int, default=10, help="Number of runs per generator.")
    parser.add_argument("--num-trials", type=int, default=80, help="Trials per run.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--reversal-interval",
        type=int,
        default=40,
        help="Context reversal interval (set <=0 to use default schedule).",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (K).")
    parser.add_argument(
        "--mode",
        choices=('kfold', 'per_run'),
        default='kfold',
        help="Choose 'kfold' for dataset-level CV or 'per_run' to emit smoke-level artifacts.",
    )
    parser.add_argument(
        "--artifact-base",
        type=str,
        default=None,
        help="Destination directory for detailed artifacts when mode='per_run'.",
    )
    parser.add_argument(
        "--record-grid",
        action='store_true',
        help="Record grid-search evaluations inside artifact folders (per_run mode).",
    )
    parser.add_argument(
        "--clean-artifacts",
        action='store_true',
        help="Delete the artifact directory before writing new outputs (per_run mode).",
    )
    return parser.parse_args()


if __name__ == '__main__':
    _cpu_total = os.cpu_count() or 1
    _half_cpu = max(1, (_cpu_total + 1) // 2)
    os.environ['MODEL_COMP_MAX_WORKERS'] = str(_half_cpu)
    print(f"[model_recovery] MODEL_COMP_MAX_WORKERS set to {_half_cpu} (half of {_cpu_total} CPUs)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    gens = tuple(g.strip() for g in args.generators.split(",") if g.strip())
    reversal_interval = args.reversal_interval if args.reversal_interval > 0 else None
    if args.mode == 'kfold':
        res, diffs = kfold_cv(
            generators=gens,
            runs_per_generator=args.runs_per_generator,
            num_trials=args.num_trials,
            seed=args.seed,
            reversal_interval=reversal_interval,
            K=args.folds,
        )
        print("Per-model CV results:")
        for m, v in res.items():
            print(m, v['fold_mean'], v['fold_se'])
        print("\nPaired diffs:")
        for k, v in diffs.items():
            print(k, v['mean'], v['se'])
    else:
        stats = per_run_cv(
            generators=gens,
            runs_per_generator=args.runs_per_generator,
            num_trials=args.num_trials,
            seed=args.seed,
            reversal_interval=reversal_interval,
            K=args.folds,
            artifact_base=args.artifact_base,
            record_grid=args.record_grid,
            clean_artifacts=args.clean_artifacts,
        )
        artifact_dir = args.artifact_base or os.path.join(MODEL_RECOVERY_BASE, 'full_cv')
        print(f"Per-run CV complete. {len(stats)} rows written to {os.path.join(artifact_dir, 'per_run_metrics.csv')}")
