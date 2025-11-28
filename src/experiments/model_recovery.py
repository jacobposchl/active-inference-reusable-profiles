
"""
Experiment entry point: Per-run cross-validation for model recovery.

Overview
--------
The script coordinates four stages:
1. Generate reference trajectories with multiple generators (M1/M2/M3/baselines).
2. For each run, fit each candidate model via within-run K-fold CV (see recovery_helpers).
3. Evaluate fitted models on held-out trials and compute per-run metrics (LL, accuracy, AIC/BIC).
4. Aggregate results into confusion matrices and per-run summaries for publication.

Progress feedback uses nested `tqdm` progress bars: the outer bar tracks runs,
and inner bars display status for per-model fitting.
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
    cv_fit_single_run,
    write_per_run_metrics,
    write_confusion_tables,
    write_experiment_metadata,
    MODEL_RECOVERY_BASE,
    _fold_level_path,
)

from config.experiment_config import *


def _check_run_exists(artifact_dir, generator, model_name, run_idx):
    """Check if results for a specific generator+model+run already exist."""
    fold_path = _fold_level_path(generator, model_name, run_idx, artifact_dir)
    return os.path.exists(fold_path)


def _load_existing_run_stats(artifact_dir, generator, model_name, run_idx):
    """Load existing run statistics from fold_level CSV."""
    import pandas as pd
    fold_path = _fold_level_path(generator, model_name, run_idx, artifact_dir)
    
    try:
        df = pd.read_csv(fold_path)
        if len(df) == 0:
            return None
        
        # Calculate statistics from fold results
        mean_test_ll = df['test_ll'].mean()
        std_test_ll = df['test_ll'].std()
        mean_test_acc = df['test_acc'].mean() if 'test_acc' in df.columns else np.nan
        std_test_acc = df['test_acc'].std() if 'test_acc' in df.columns else np.nan
        
        # Get other metadata from first row
        seed = df['seed'].iloc[0] if 'seed' in df.columns else None
        grid_evals = df['grid_evals'].sum() if 'grid_evals' in df.columns else 0
        
        # Calculate AIC/BIC
        from src.utils.model_utils import get_num_parameters
        k_params = get_num_parameters(model_name)
        num_trials = 80  # default, could be read from metadata
        aic = 2 * k_params - 2 * mean_test_ll
        bic = k_params * np.log(max(1, num_trials)) - 2 * mean_test_ll
        
        return {
            'generator': generator,
            'model': model_name,
            'run_idx': run_idx,
            'seed': seed,
            'mean_test_ll': mean_test_ll,
            'std_test_ll': std_test_ll,
            'mean_test_acc': mean_test_acc,
            'std_test_acc': std_test_acc,
            'aic': aic,
            'bic': bic,
            'runtime_sec': 0,  # Unknown for existing runs
            'grid_evals': grid_evals,
        }
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load existing stats for {generator}/{model_name}/run_{run_idx}: {e}")
        return None


def run_model_recovery(
    generators=('M1', 'M2', 'M3', 'egreedy', 'softmax'),
    runs_per_generator=10,
    num_trials=80,
    seed=1,
    reversal_interval=40,
    K=5,
    run_id=None,
    artifact_base=None,
    record_grid=True,
    save_artifacts=True,
    resume_from_existing=False,
):
    """
    Main model recovery pipeline: per-run cross-validation with full diagnostics.

    Generates reference runs from each behavioral generator, then performs within-run
    K-fold CV for each candidate model. Outputs include per-run metrics (LL, accuracy,
    AIC/BIC), confusion matrices, trial-level predictions, and metadata.

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
        Number of folds for within-run cross-validation.
    run_id : str or None
        Unique identifier for this experiment run. If None, auto-generates timestamp-based ID.
    artifact_base : str or None
        Base output directory; defaults to 'results/model_recovery'. Results are saved
        in artifact_base/run_{run_id}/.
    record_grid : bool
        If True, save grid-search evaluations to CSV (default: True).
    save_artifacts : bool
        If True, write trial-level, fold-level, and summary CSVs (default: True).
    resume_from_existing : bool
        If True, skip generator+model+run combinations that already have results.
        Useful for resuming from crashes (default: False).

    Returns
    -------
    per_run_stats : list of dict
        Per-run summary statistics (mean_test_ll, AIC, BIC, etc.) for all models and runs.
    """
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    # Generate unique run ID if not provided
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("="*60)
    logger.info(f"Model Recovery Experiment: {run_id}")
    logger.info("="*60)
    logger.info(
        "Configuration: generators=%s, runs_per_generator=%s, num_trials=%s, K=%s, seed=%s",
        generators,
        runs_per_generator,
        num_trials,
        K,
        seed,
    )
    
    # Generate reference runs
    A, B, D, refs = generate_all_runs(generators, runs_per_generator, num_trials, seed, reversal_interval)
    logger.info(f"Generated {len(refs)} reference runs")
    
    # Set up output directory with unique run ID
    base_dir = artifact_base or MODEL_RECOVERY_BASE
    artifact_dir = os.path.join(base_dir, f'run_{run_id}')
    os.makedirs(artifact_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {artifact_dir}")

    candidate_models = ['M1', 'M2', 'M3']
    per_run_stats = []
    total_fits = len(refs) * len(candidate_models)
    completed_fits = 0
    
    # Summary statistics for progress tracking
    gen_counts = {g: sum(1 for r in refs if r['gen'] == g) for g in set(r['gen'] for r in refs)}
    gen_completed = {g: 0 for g in gen_counts}

    # Process each reference run with all candidate models
    logger.info(f"Starting model fitting: {len(refs)} runs × {len(candidate_models)} models = {total_fits} total fits")
    
    run_bar = tqdm(
        refs, 
        desc="Overall Progress", 
        unit="run", 
        position=0,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} runs [{elapsed}<{remaining}]'
    )
    
    for run_idx, ref in enumerate(run_bar, 1):
        # Update outer progress bar with current generator and run
        gen_progress = f"{gen_completed[ref['gen']] + 1}/{gen_counts[ref['gen']]}"
        run_bar.set_postfix_str(
            f"Gen={ref['gen']} ({gen_progress}), Fits={completed_fits}/{total_fits}"
        )
        
        gen_completed[ref['gen']] += 1
        
        model_bar = tqdm(
            candidate_models, 
            desc=f"  └─ Models", 
            leave=False, 
            position=1,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]'
        )
        
        for model_name in model_bar:
            # Check if this run already exists (resume mode)
            if resume_from_existing and _check_run_exists(artifact_dir, ref['gen'], model_name, ref['run_idx']):
                completed_fits += 1
                model_bar.set_postfix_str(f"{model_name} ({ref['gen']}-run{ref['run_idx']}) [SKIPPED]")
                logger.info(f"Skipping existing result: gen={ref['gen']}, model={model_name}, run={ref['run_idx']}")
                
                # Load existing stats to include in final aggregation
                existing_stats = _load_existing_run_stats(artifact_dir, ref['gen'], model_name, ref['run_idx'])
                if existing_stats:
                    per_run_stats.append(existing_stats)
                
                continue
            
            completed_fits += 1
            model_bar.set_postfix_str(f"{model_name} ({ref['gen']}-run{ref['run_idx']})")
            
            # Time the model fit
            import time
            fit_start = time.time()
            
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
            
            fit_duration = time.time() - fit_start
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
            
            # Write brief summary to model bar after completion
            model_bar.write(
                f"    ✓ {model_name} on {ref['gen']}-run{ref['run_idx']}: "
                f"LL={summary['mean_test_ll']:.2f}, AIC={summary['aic']:.1f}, "
                f"time={fit_duration:.1f}s"
            )
        
        model_bar.close()

    if not per_run_stats:
        logger.warning("No per-run stats were generated.")
        return []

    # Write aggregate outputs
    logger.info("Writing aggregate metrics and confusion matrices...")
    write_per_run_metrics(per_run_stats, base_dir=artifact_dir)
    write_confusion_tables(per_run_stats, candidate_models, base_dir=artifact_dir)
    
    # Write experiment metadata
    metadata = {
        'run_id': run_id,
        'generators': list(generators),
        'runs_per_generator': runs_per_generator,
        'num_trials': num_trials,
        'seed': seed,
        'reversal_interval': reversal_interval,
        'folds': K,
        'candidate_models': candidate_models,
        'artifact_dir': artifact_dir,
        'total_runs': len(refs),
        'record_grid': record_grid,
    }
    write_experiment_metadata(metadata, base_dir=artifact_dir)
    
    logger.info("="*60)
    logger.info("Model recovery pipeline complete!")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Total runs processed: {len(refs)}")
    logger.info(f"Results written to: {artifact_dir}")
    logger.info(f"  - per_run_metrics.csv: {len(per_run_stats)} rows")
    logger.info(f"  - confusion matrices: confusion/")
    logger.info(f"  - trial-level data: trial_level/")
    logger.info("="*60)
    
    return per_run_stats, artifact_dir


def _parse_args():
    """CLI helper for configuring the model recovery experiment."""
    parser = argparse.ArgumentParser(
        description="Model recovery experiment with per-run cross-validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generators",
        type=str,
        default="M1,M2,M3,egreedy,softmax",
        help="Comma-separated list of behavioral generators to simulate.",
    )
    parser.add_argument("--runs-per-generator", type=int, default=5, help="Number of runs per generator.")
    parser.add_argument("--num-trials", type=int, default=400, help="Trials per run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--reversal-interval",
        type=int,
        default=40,
        help="Context reversal interval (set <=0 to use default schedule).",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of within-run CV folds (K).")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Unique identifier for this experiment run (default: auto-generated timestamp).",
    )
    parser.add_argument(
        "--artifact-base",
        type=str,
        default=None,
        help="Base directory for output artifacts (default: results/model_recovery).",
    )
    parser.add_argument(
        "--no-record-grid",
        action='store_true',
        help="Disable recording grid-search evaluations (saves disk space).",
    )
    parser.add_argument(
        "--resume",
        action='store_true',
        help="Resume from existing results, skipping already completed generator+model+run combinations.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    # Configure worker processes to use most CPU cores for parallel grid search
    _cpu_total = os.cpu_count() or 1
    _sub_cpu = max(1, _cpu_total - 10)  # Leave 10 cores for system
    os.environ['MODEL_COMP_MAX_WORKERS'] = str(_sub_cpu)
    print(f"[model_recovery] Using {_sub_cpu} worker processes (out of {_cpu_total} total)")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Parse command-line arguments
    args = _parse_args()
    gens = tuple(g.strip() for g in args.generators.split(",") if g.strip())
    reversal_interval = args.reversal_interval if args.reversal_interval > 0 else None
    
    # Run the main experiment
    stats, artifact_dir = run_model_recovery(
        generators=gens,
        runs_per_generator=args.runs_per_generator,
        num_trials=args.num_trials,
        seed=args.seed,
        reversal_interval=reversal_interval,
        K=args.folds,
        run_id=args.run_id,
        artifact_base=args.artifact_base,
        record_grid=not args.no_record_grid,  # Default True unless --no-record-grid is passed
        resume_from_existing=args.resume,
    )
    
    # Print summary
    print(f"\n✓ Model recovery complete!")
    print(f"  Total entries: {len(stats)}")
    print(f"  Results location: {artifact_dir}/")
    print(f"    - per_run_metrics.csv")
    print(f"    - confusion/")
    print(f"    - trial_level/")
    print(f"    - fold_level/")
    if not args.no_record_grid:
        print(f"    - grid_evals/")
