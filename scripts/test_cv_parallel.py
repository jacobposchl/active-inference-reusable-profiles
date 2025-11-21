r"""Small test script to validate parallelized fitting in CV recovery.

Run from the project root using your virtualenv python, for example:

Windows (cmd.exe):
C:\> "C:/Users/Jacob Poschl/reusable profiles in active inference/active-inference-reusable-profiles/.venv/Scripts/python.exe" scripts/test_cv_parallel.py

The script runs a tiny 2-fold CV with 2 runs per generator (M1,M2,M3) and prints
per-model fold means and paired differences. Set environment variable
`MODEL_COMP_MAX_WORKERS` to control number of worker processes.
"""
import os
import sys

# Ensure project root is on sys.path so `import src...` works when running
# this script directly from the `scripts/` folder.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import os
import sys

# Allow overriding number of workers for quick testing
os.environ.setdefault('MODEL_COMP_MAX_WORKERS', os.environ.get('MODEL_COMP_MAX_WORKERS', str(max(1, (os.cpu_count() or 1) - 1))))

from src.experiments.cv_recovery import kfold_cv


def main():
    try:
        print("Running small parallel CV smoke test...")
        res, diffs = kfold_cv(generators=['M1','M2','M3'], runs_per_generator=2, num_trials=40, seed=123, reversal_interval=20, K=2)

        print('\n--- CV Results Summary ---')
        for m,v in res.items():
            print(m, 'mean:', v['fold_mean'], 'se:', v['fold_se'])

        print('\n--- Paired diffs ---')
        for k,v in diffs.items():
            print(k, 'mean:', v['mean'], 'se:', v['se'])

        print('\nSmoke test completed successfully.')
    except Exception as e:
        print('Smoke test failed with exception:')
        raise


if __name__ == '__main__':
    main()
