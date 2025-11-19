"""Command-line runner for fitting models to participants.

Usage (example):
python run_fit.py --data data/raw/hartley_twostep_data.csv --participant 001 --model M1 --out results/fitted_parameters

This script fits one participant and writes a CSV with fitted params and metrics.
"""
import argparse
import os
import pandas as pd
import numpy as np

from two_step_task.models.m1_static import M1_StaticPrecision
from two_step_task.models.m2_entropy import M2_EntropyCoupled
from two_step_task.models.m3_profiles import M3_ProfileBased
from two_step_task.fitting.parameter_search import parameter_search

MODEL_MAP = {
    'M1': M1_StaticPrecision,
    'M2': M2_EntropyCoupled,
    'M3': M3_ProfileBased,
}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def fit_single_participant(data_csv, participant_id, model_name, out_dir, n_points=6):
    df = pd.read_csv(data_csv)
    df = df[df['participant_id'] == participant_id]
    if df.empty:
        raise ValueError(f'No data for participant {participant_id}')

    ModelClass = MODEL_MAP.get(model_name)
    if ModelClass is None:
        raise ValueError(f'Unknown model {model_name}')

    model = ModelClass()

    best_params, best_ll = parameter_search(model, df, n_points=n_points)

    # Save results
    ensure_dir(out_dir)
    fname = os.path.join(out_dir, f'{model_name}_participant_{participant_id}.csv')
    # Map params to names
    param_names = list(model.free_parameters().keys())
    if best_params is None:
        row = { 'participant_id': participant_id, 'model': model_name, 'll': None }
    else:
        row = { 'participant_id': participant_id, 'model': model_name, 'll': float(best_ll) }
        for name, val in zip(param_names, best_params):
            row[name] = float(val)

    pd.DataFrame([row]).to_csv(fname, index=False)
    return fname


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to CSV with raw two-step data')
    p.add_argument('--participant', required=True, help='Participant id to fit')
    p.add_argument('--model', choices=['M1','M2','M3'], default='M1')
    p.add_argument('--out', default='two_step_task/results/fitted_parameters')
    p.add_argument('--n_points', type=int, default=6, help='Grid n points per param (coarse)')
    args = p.parse_args()

    outfile = fit_single_participant(args.data, args.participant, args.model, args.out, n_points=args.n_points)
    print('Saved fit to', outfile)
