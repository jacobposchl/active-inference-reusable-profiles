"""Small smoke demo for within-run K-fold CV.

This script runs a tiny example (2 runs, short trials) and exercises the
cv_fit_single_run function to create trial-level CSV outputs.
"""
import os
import sys

# Ensure project root is on sys.path so `import src...` works when running
# this script directly from the `scripts/` folder.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.utils.recovery_helpers import build_abd, cv_fit_single_run, generate_all_runs


def main():
    out_dir = os.path.join('results', 'csv')
    os.makedirs(out_dir, exist_ok=True)

    # generate a small set of refs
    gens = ['M3']
    runs = 2
    trials = 80
    seed = 7
    A, B, D, refs = generate_all_runs(gens, runs, trials, seed, reversal_interval=40)

    # For each ref, run within-run CV for M1, M2, M3
    models = ['M1', 'M2', 'M3']
    for ref in refs:
        for m in models:
            print(f"Running CV for run {ref['run_idx']} generator {ref['gen']} model {m}")
            res = cv_fit_single_run(
                m,
                A,
                B,
                D,
                ref['ref_logs'],
                K=5,
                run_id=ref['run_idx'],
                generator=ref['gen'],
                seed=ref['seed'],
                artifact_base_dir=os.path.join('results', 'model_recovery', 'smoke_demo'),
                save_artifacts=True,
                record_grid=False,
            )
            print('saved trial csv:', res.get('trial_csv'))


if __name__ == '__main__':
    main()
