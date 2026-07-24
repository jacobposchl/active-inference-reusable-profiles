#!/usr/bin/env python
"""
Reproduce the study end-to-end from a single experiment run.

Runs the model-recovery experiment once, then builds BOTH paper figures
(AIC confusion matrix + M3 mechanistic panels) from that SAME run directory,
so the two figures can never drift onto different experiment runs.

Examples
--------
Full pipeline (experiment + both figures):
    python reproduce.py --run-id paper_final --seed 42 --folds 5

Rebuild figures only, from an existing run (no re-fitting):
    python reproduce.py --figures-only --results-dir results/model_recovery/run_paper_final
"""
import argparse
import logging
import os
import sys

# Headless-safe plotting; must precede importing the figure modules.
import matplotlib
matplotlib.use('Agg')

# Repo root on the path so 'src' and 'figure_scripts' import cleanly.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.experiments.model_recovery import run_model_recovery
from figure_scripts.fig_model_recovery_aic import (
    load_confusion_matrices,
    print_summary_stats,
    create_model_recovery_figure,
)
from figure_scripts.fig_mechanistic_analysis import (
    load_trial_data,
    print_mechanistic_summary,
    create_mechanistic_figure,
)


def build_figures(results_dir, figures_dir):
    """Generate both paper figures from a single results directory.

    Figures are written under a per-run subfolder so each run's outputs are
    self-contained and traceable to the run that produced them.
    """
    run_label = os.path.basename(os.path.normpath(results_dir))
    out_dir = os.path.join(figures_dir, run_label)
    os.makedirs(out_dir, exist_ok=True)

    # Figure 2 -- AIC model-recovery confusion matrix.
    aic_path = os.path.join(out_dir, 'model_recovery_aic.png')
    aic_mean, aic_se = load_confusion_matrices(results_dir)
    print_summary_stats(aic_mean, aic_se)
    create_model_recovery_figure(aic_mean, aic_se, aic_path)

    # Figure 1 -- M3 mechanistic panels.
    mech_path = os.path.join(out_dir, 'mechanistic_analysis.png')
    data = load_trial_data(results_dir, generator='M3')
    if not data:
        raise SystemExit(
            f"[reproduce] no trial-level data under {results_dir}; "
            "cannot build the mechanistic figure."
        )
    print_mechanistic_summary(data, run_idx=0)
    create_mechanistic_figure(data, mech_path, run_idx=0, results_dir=results_dir)

    return out_dir


def main():
    p = argparse.ArgumentParser(
        description="Reproduce the study: one experiment run -> both paper figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--run-id', default='paper_final',
                   help="Run id; the experiment writes to results/model_recovery/run_<id>.")
    p.add_argument('--generators', default='M1,M2,M3,egreedy,softmax',
                   help="Comma-separated behavioral generators.")
    p.add_argument('--seed', type=int, default=42, help="Random seed.")
    p.add_argument('--folds', type=int, default=5, help="Number of within-run CV folds (K).")
    p.add_argument('--runs-per-generator', type=int, default=5,
                   help="Independent runs simulated per generator.")
    p.add_argument('--num-trials', type=int, default=400, help="Trials per run.")
    p.add_argument('--reversal-interval', type=int, default=40,
                   help="Context reversal interval; <=0 uses the default schedule.")
    p.add_argument('--reserve-cores', type=int, default=10,
                   help="CPU cores reserved for the system during grid search.")
    p.add_argument('--artifact-base', default=None,
                   help="Base output directory (default: results/model_recovery).")
    p.add_argument('--figures-dir', default='figures',
                   help="Where figures are written (under a per-run subfolder).")
    p.add_argument('--figures-only', action='store_true',
                   help="Skip the experiment; rebuild figures from --results-dir.")
    p.add_argument('--results-dir', default=None,
                   help="Existing run dir to build figures from (required with --figures-only).")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.figures_only:
        if not args.results_dir:
            p.error("--figures-only requires --results-dir")
        run_dir = args.results_dir
        if not os.path.isdir(run_dir):
            p.error(f"--results-dir does not exist: {run_dir}")
        print(f"[reproduce] figures-only from {run_dir}")
    else:
        cpu_total = os.cpu_count() or 1
        workers = max(1, cpu_total - args.reserve_cores)
        os.environ['MODEL_COMP_MAX_WORKERS'] = str(workers)
        print(f"[reproduce] using {workers} worker processes "
              f"(of {cpu_total}, {args.reserve_cores} reserved)")

        gens = tuple(g.strip() for g in args.generators.split(',') if g.strip())
        reversal_interval = args.reversal_interval if args.reversal_interval > 0 else None

        result = run_model_recovery(
            generators=gens,
            runs_per_generator=args.runs_per_generator,
            num_trials=args.num_trials,
            seed=args.seed,
            reversal_interval=reversal_interval,
            K=args.folds,
            run_id=args.run_id,
            artifact_base=args.artifact_base,
        )
        if not (isinstance(result, tuple) and len(result) == 2):
            raise SystemExit("[reproduce] experiment produced no results; aborting before figures.")
        _stats, run_dir = result
        print(f"[reproduce] experiment complete -> {run_dir}")

    out_dir = build_figures(run_dir, args.figures_dir)
    print(f"\n[reproduce] done.\n  run dir : {run_dir}\n  figures : {out_dir}/")


if __name__ == '__main__':
    main()
