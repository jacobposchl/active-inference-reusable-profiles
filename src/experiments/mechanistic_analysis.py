"""
Mechanistic analysis around reversals.

Runs short experiments (90 trials, reversals at 30 and 60) and visualizes
how M1/M2/M3 behave around reversal points.

Produces figures:
- accuracy_around_reversal.png: accuracy aligned to reversal (pooled across runs and reversals) for all models
- gamma_around_reversal.png: policy precision (gamma) aligned to reversals for all models

Usage:
    python src/experiments/mechanistic_analysis.py

"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path (so imports from src.* work)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.experiment_config import *
from src.models import build_A, build_B, build_D
from src.experiments.model_comparison import run_single_agent
from src.utils import trial_accuracy, bootstrap_ci
from src.utils.helpers import find_reversals
from src.utils.plotting import plot_accuracy_around_reversals, plot_m3_profile_weights_by_direction, plot_gamma_around_reversals
from config.experiment_config import M3_DEFAULTS


def run_mechanistic_experiment(models=('M1', 'M2', 'M3'), num_runs=20, num_trials=90, seed=42,
                               reversal_schedule=[30, 60], pre=10, post=20):
    """Run short experiments and produce mechanistic figures.
    """
    print('Mechanistic analysis: short runs around reversals')
    print(f'models={models}, runs={num_runs}, trials={num_trials}, reversals={reversal_schedule}')
    
    A = build_A(NUM_MODALITIES, STATE_CONTEXTS, STATE_CHOICES,
               OBSERVATION_HINTS, OBSERVATION_REWARDS, OBSERVATION_CHOICES,
               PROBABILITY_HINT, PROBABILITY_REWARD)
    B = build_B(STATE_CONTEXTS, STATE_CHOICES, ACTION_CONTEXTS, ACTION_CHOICES,
               context_volatility=DEFAULT_CONTEXT_VOLATILITY)
    D = build_D(STATE_CONTEXTS, STATE_CHOICES)

    results_by_model = {m: [] for m in models}

    for model_name in models:
        print(f'Running {model_name}...')
        for run in tqdm(range(num_runs), desc=f'  {model_name}'):
            run_seed = seed + run if seed is not None else None
            logs = run_single_agent(model_name, A, B, D, num_trials, seed=run_seed, reversal_schedule=reversal_schedule)
            results_by_model[model_name].append(logs)

    # Plotting
    outdir = 'results/figures'
    ensure_dir(outdir)
    plot_accuracy_around_reversals(results_by_model, pre=pre, post=post, outpath=os.path.join(outdir, f'mechanistic_accuracy_rev_{reversal_schedule[0]}_{reversal_schedule[1]}.png'))
    plot_gamma_around_reversals(results_by_model, pre=pre, post=post, outpath=os.path.join(outdir, f'mechanistic_gamma_rev_{reversal_schedule[0]}_{reversal_schedule[1]}.png'))
    plot_m3_profile_weights(results_by_model['M3'], pre=pre, post=post, outpath=os.path.join(outdir, f'mechanistic_m3_profiles_rev_{reversal_schedule[0]}_{reversal_schedule[1]}.png'))
    # Also produce direction-specific profile-weight plots (switch -> left vs switch -> right)
    plot_m3_profile_weights_by_direction(results_by_model['M3'], pre=pre, post=post, outpath=os.path.join(outdir, f'mechanistic_m3_profiles_rev_{reversal_schedule[0]}_{reversal_schedule[1]}.png'))

    print('Mechanistic analysis complete. Figures saved to results/figures')
    return results_by_model


if __name__ == '__main__':
    run_mechanistic_experiment(num_runs=20, num_trials=90, seed=42, reversal_schedule=[30, 60], pre=10, post=20)
