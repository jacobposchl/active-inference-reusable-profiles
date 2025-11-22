"""
Entry point for the behavioral changepoint experiment validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import data_io, nassar_forward, preprocessing, cross_validation, rl_baselines
from .constants import RESULTS_DIR


def run_pipeline(save: bool = True) -> dict:
    events = data_io.load_dataset()
    events_with_signals = nassar_forward.compute_normative_signals(events)
    clean_trials, qc = preprocessing.prepare_trials(events_with_signals)
    enriched = preprocessing.inject_belief_columns(clean_trials)

    loso_results = cross_validation.loso_cv(enriched)
    temporal_results = cross_validation.temporal_split_cv(enriched)
    rl_results = rl_baselines.run_rl_baselines(trials=clean_trials, save=save)
    loso_combined = pd.concat([loso_results, rl_results], ignore_index=True)

    outputs = {
        "qc": {
            "dropped_subjects": qc.dropped_subjects,
            "invalid_fraction": qc.invalid_fraction,
        },
        "loso": loso_results,
        "temporal": temporal_results,
        "rl_loso": rl_results,
        "loso_all": loso_combined,
    }

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        loso_path = RESULTS_DIR / "loso_results.csv"
        temporal_path = RESULTS_DIR / "temporal_split_results.csv"
        rl_path = RESULTS_DIR / "rl_loso_results.csv"
        loso_all_path = RESULTS_DIR / "loso_results_all.csv"
        qc_path = RESULTS_DIR / "qc_summary.json"

        loso_results.to_csv(loso_path, index=False)
        temporal_results.to_csv(temporal_path, index=False)
        rl_results.to_csv(rl_path, index=False)
        loso_combined.to_csv(loso_all_path, index=False)
        with open(qc_path, "w", encoding="utf-8") as f:
            json.dump(outputs["qc"], f, indent=2)

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Run behavioral validation pipeline.")
    parser.add_argument("--no-save", action="store_true", help="Skip writing outputs to disk.")
    args = parser.parse_args()

    outputs = run_pipeline(save=not args.no_save)
    print("QC summary:")
    print(json.dumps(outputs["qc"], indent=2))
    print("LOSO head:")
    print(outputs["loso"].head())


if __name__ == "__main__":
    main()


