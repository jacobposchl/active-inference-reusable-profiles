"""
Smoke test for the full model recovery pipeline.

Runs a very small configuration (1 generator, 1 run, 20 trials) to verify
the end-to-end pipeline works correctly. Uses identical logic to the main
experiment but with minimal parameters for fast validation.
"""
import os
import sys

# Ensure project root is on sys.path when invoked directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experiments.model_recovery import run_model_recovery


def main():
    # Use 4 workers for reasonable parallelism in smoke test
    # (M3 has thousands of grid candidates, needs parallel eval)
    os.environ.setdefault('MODEL_COMP_MAX_WORKERS', '15')

    # Minimal configuration for fast smoke test
    generators = ['M1']
    runs_per_generator = 1
    num_trials = 20
    seed = 42
    reversal_interval = 5
    folds = 2

    print("="*60)
    print("Running smoke test (minimal configuration)...")
    print("  Generators: M1 only")
    print("  Runs: 1")
    print("  Trials: 20")
    print("  Folds: 2")
    print("="*60)
    
    stats, artifact_dir = run_model_recovery(
        generators=generators,
        runs_per_generator=runs_per_generator,
        num_trials=num_trials,
        seed=seed,
        reversal_interval=reversal_interval,
        K=folds,
        run_id='smoke_test',
        artifact_base=os.path.join('results', 'model_recovery'),
        record_grid=False,  # Skip grid evals for speed
    )
    
    print("\n" + "="*60)
    print("✓ Smoke test PASSED")
    print(f"  Processed {len(stats)} model fits")
    print(f"  Results: {artifact_dir}/")
    print("="*60)
    print("\nIf you see this message, the pipeline is working correctly!")
    print("You can now run the full experiment with:")
    print("  python src/experiments/model_recovery.py")


if __name__ == '__main__':
    main()
