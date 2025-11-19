"""Smoke test: run a tiny pymdp-based fit on one processed subject.

This script picks the first CSV in `two_step_task/data/processed/` and
runs a very small grid search to ensure the pymdp adapter and fitting
loop execute without errors.
"""
from pathlib import Path
import pandas as pd
from two_step_task.models.m1_static import M1_StaticPrecision
from two_step_task.fitting.parameter_search import parameter_search


def main():
    proc_dir = Path("two_step_task/data/processed")
    files = sorted(proc_dir.glob("participant_*.csv"))
    if not files:
        print("No processed participant files found in two_step_task/data/processed/")
        return

    subj = files[0]
    print(f"Using subject file: {subj}")
    df = pd.read_csv(subj)

    model = M1_StaticPrecision()
    # tiny grid for smoke test
    best_params, best_ll = parameter_search(model, df, n_points=3, top_k=2)

    print("Smoke test result:")
    print("best_params:", best_params)
    print("best_ll:", best_ll)


if __name__ == "__main__":
    main()
