"""
Smoke test for the full model recovery pipeline.

Runs a very small configuration (single generator/run, short trial count)
through `cv_fit_single_run`, writes the structured artifacts under
`results/model_recovery/smoke_demo`, and generates confusion matrices.
"""
import os
import shutil
import sys

# Ensure project root is on sys.path when invoked directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import recovery_helpers as rh


def main():
    os.environ.setdefault('MODEL_COMP_MAX_WORKERS', '1')
    artifact_base = os.path.join('results', 'model_recovery', 'smoke_demo')
    shutil.rmtree(artifact_base, ignore_errors=True)

    generators = ['M1']
    runs_per_generator = 1
    num_trials = 20
    seed = 42
    reversal_interval = 5
    folds = 2

    print("Generating reference runs...")
    A, B, D, refs = rh.generate_all_runs(generators, runs_per_generator, num_trials, seed, reversal_interval)

    candidate_models = ['M1', 'M2', 'M3']
    per_run_stats = []

    for ref in refs:
        for model in candidate_models:
            print(f"[smoke] fitting {model} on generator {ref['gen']} run {ref['run_idx']}")
            result = rh.cv_fit_single_run(
                model,
                A,
                B,
                D,
                ref['ref_logs'],
                K=folds,
                run_id=ref['run_idx'],
                generator=ref['gen'],
                seed=ref['seed'],
                artifact_base_dir=artifact_base,
                save_artifacts=True,
                record_grid=False,
            )
            summary = result['summary']
            entry = {
                'generator': ref['gen'],
                'model': model,
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

    print("Writing aggregate summaries...")
    rh.write_per_run_metrics(per_run_stats, base_dir=artifact_base)
    rh.write_confusion_tables(per_run_stats, candidate_models, base_dir=artifact_base)
    metadata = {
        'generators': generators,
        'runs_per_generator': runs_per_generator,
        'num_trials': num_trials,
        'seed': seed,
        'reversal_interval': reversal_interval,
        'folds': folds,
        'candidate_models': candidate_models,
    }
    rh.write_experiment_metadata(metadata, base_dir=artifact_base)
    print(f"Smoke test complete. Artifacts written to {artifact_base}")


if __name__ == '__main__':
    main()
