"""
Smoke test for the full model recovery pipeline.

Runs a very small configuration (single generator/run, short trial count)
through `cv_fit_single_run`, writes the structured artifacts under
`results/model_recovery/smoke_demo`, and generates confusion matrices.
"""
import os
import sys

# Ensure project root is on sys.path when invoked directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experiments.model_recovery import per_run_cv


def main():
    os.environ.setdefault('MODEL_COMP_MAX_WORKERS', '1')
    artifact_base = os.path.join('results', 'model_recovery', 'smoke_demo')

    generators = ['M1']
    runs_per_generator = 1
    num_trials = 20
    seed = 42
    reversal_interval = 5
    folds = 2

    print("Running smoke per-run CV...")
    per_run_cv(
        generators=generators,
        runs_per_generator=runs_per_generator,
        num_trials=num_trials,
        seed=seed,
        reversal_interval=reversal_interval,
        K=folds,
        artifact_base=artifact_base,
        record_grid=False,
        clean_artifacts=True,
    )
    print(f"Smoke test complete. Artifacts written to {artifact_base}")


if __name__ == '__main__':
    main()
