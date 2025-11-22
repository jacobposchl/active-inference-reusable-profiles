"""
Utility to print aggregate summaries of the saved CV results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from itertools import combinations

from . import data_io, preprocessing

PARAM_COUNTS = {
    "M1": 1,
    "M2": 1,
    "M3": 2,
    "RW_fixed": 1,
    "RW_dynamic": 3,
    "QL_eps_greedy": 2,
    "QL_softmax": 2,
}


def _load_results():
    root = Path(__file__).resolve().parents[1] / "derivatives" / "analysis"
    loso_path = root / "loso_results_all.csv"
    if loso_path.exists():
        loso = pd.read_csv(loso_path)
    else:
        loso = pd.read_csv(root / "loso_results.csv")
        rl_path = root / "rl_loso_results.csv"
        if rl_path.exists():
            rl = pd.read_csv(rl_path)
            loso = pd.concat([loso, rl], ignore_index=True)
    temporal = pd.read_csv(root / "temporal_split_results.csv")
    return loso, temporal


def _apply_information_criteria(
    df: pd.DataFrame, fallback_counts: Dict[str, int]
) -> pd.DataFrame:
    enriched = df.copy()
    if "num_trials" not in enriched.columns:
        enriched["num_trials"] = enriched["subject_id"].map(fallback_counts)
    enriched["k"] = enriched["model"].map(PARAM_COUNTS).astype(float)
    enriched["aic"] = -2.0 * enriched["test_ll"] + 2.0 * enriched["k"]
    enriched["bic"] = -2.0 * enriched["test_ll"] + enriched["k"] * np.log(
        enriched["num_trials"].astype(float)
    )
    return enriched


def _summaries(df: pd.DataFrame):
    stats = (
        df.groupby("model")["test_ll"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mean_test_ll", "std": "std_test_ll"})
    )
    winners = (
        df.loc[df.groupby("subject_id")["test_ll"].idxmax(), "model"]
        .value_counts()
        .sort_index()
    )
    return stats, winners


def _trial_counts() -> Tuple[Dict[str, int], Dict[str, int]]:
    events = data_io.load_dataset()
    clean, _ = preprocessing.prepare_trials(events)
    loso_counts = clean.groupby("subject_id").size().to_dict()
    temporal_counts = {}
    for subject, sub_df in clean.groupby("subject_id"):
        run_ids = sorted(sub_df["run_id"].unique())
        midpoint = len(run_ids) // 2
        test_runs = run_ids[midpoint:]
        temporal_counts[subject] = sub_df[sub_df["run_id"].isin(test_runs)].shape[0]
    return loso_counts, temporal_counts


def main():
    loso, temporal = _load_results()
    loso_counts, temporal_counts = _trial_counts()
    loso = _apply_information_criteria(loso, loso_counts)
    temporal = _apply_information_criteria(temporal, temporal_counts)

    loso_stats, loso_wins = _summaries(loso)
    temp_stats, temp_wins = _summaries(temporal)

    pivot = loso.pivot(index="subject_id", columns="model", values="test_ll")
    delta_m3_m1 = (pivot["M3"] - pivot["M1"]).mean()
    delta_m3_m2 = (pivot["M3"] - pivot["M2"]).mean()

    print("=== LOSO (leave-one-subject-out) ===")
    print(loso_stats.to_string(float_format="{:.2f}".format))
    print("\nLOSO winners (count of subjects where each model had highest test LL):")
    print(loso_wins.to_string())
    print(f"\nMean ΔLL (M3 - M1): {delta_m3_m1:.2f}")
    print(f"Mean ΔLL (M3 - M2): {delta_m3_m2:.2f}\n")

    def report_ttests(pivot_df: pd.DataFrame, label: str):
        print(f"--- Paired t-tests ({label}) ---")
        cols = [c for c in pivot_df.columns if pivot_df[c].notna().all()]
        if len(cols) < 2:
            print("Not enough models with complete data for t-tests.\n")
            return
        for a, b in combinations(cols, 2):
            series_a = pivot_df[a]
            series_b = pivot_df[b]
            diff = series_a - series_b
            t_stat, p_val = ttest_rel(series_a, series_b)
            mean_delta = diff.mean()
            print(
                f"{a} vs {b}: ΔLL={mean_delta:.2f}, "
                f"t={t_stat:.2f}, p={p_val:.4g}"
            )
        print()

    report_ttests(pivot, "LOSO (all models)")

    print("=== Within-subject temporal split ===")
    print(temp_stats.to_string(float_format="{:.2f}".format))
    print("\nTemporal split winners:")
    print(temp_wins.to_string())

    temp_pivot = temporal.pivot(index="subject_id", columns="model", values="test_ll")
    report_ttests(temp_pivot, "Temporal split (AI models)")

    for label, df in [("LOSO", loso), ("Temporal", temporal)]:
        ic_summary = (
            df.groupby("model")[["aic", "bic"]]
            .mean()
            .rename(columns={"aic": "mean_aic", "bic": "mean_bic"})
        )
        best_counts = (
            df.loc[df.groupby("subject_id")["aic"].idxmin(), "model"]
            .value_counts()
            .rename("aic_best")
        )
        bic_counts = (
            df.loc[df.groupby("subject_id")["bic"].idxmin(), "model"]
            .value_counts()
            .rename("bic_best")
        )
        print(f"\n=== {label} information criteria ===")
        print(ic_summary.to_string(float_format="{:.2f}".format))
        print("\nAIC winners:")
        print(best_counts.to_string())
        print("\nBIC winners:")
        print(bic_counts.to_string())


if __name__ == "__main__":
    main()


