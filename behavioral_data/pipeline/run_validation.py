"""
Entry point for the behavioral changepoint experiment validation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path when running as script
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
from tqdm import tqdm

from behavioral_data.pipeline import data_io, nassar_forward, preprocessing, cross_validation_aif, rl_baselines
from behavioral_data.pipeline.constants import RESULTS_DIR


def run_pipeline(save: bool = True) -> dict:
    # Stage 1: Load data
    with tqdm(total=1, desc="Loading dataset", unit="step") as pbar:
        events = data_io.load_dataset()
        pbar.update(1)
    
    # Stage 2: Compute normative signals
    with tqdm(total=1, desc="Computing normative signals", unit="step") as pbar:
        events_with_signals = nassar_forward.compute_normative_signals(events)
        pbar.update(1)
    
    # Stage 3: Preprocess trials
    with tqdm(total=1, desc="Preprocessing trials", unit="step") as pbar:
        clean_trials, qc = preprocessing.prepare_trials(events_with_signals)
        pbar.update(1)
        tqdm.write(f"  → Loaded {len(clean_trials)} trials from {clean_trials['subject_id'].nunique()} subjects")
        tqdm.write(f"  → Dropped {len(qc.dropped_subjects)} subjects with >20% invalid trials")
    
    # Note: enriched with belief columns not needed for new AIF models
    # They compute beliefs internally via pymdp

    # Stage 4: LOSO cross-validation for AIF models
    tqdm.write("\n" + "="*60)
    tqdm.write("Running Leave-One-Subject-Out Cross-Validation (AIF Models)")
    tqdm.write("="*60)
    loso_path = RESULTS_DIR / "loso_results.csv" if save else None
    loso_results = cross_validation_aif.loso_cv(clean_trials, save_path=loso_path)
    
    # Stage 5: Temporal split cross-validation for AIF models
    tqdm.write("\n" + "="*60)
    tqdm.write("Running Temporal Split Cross-Validation (AIF Models)")
    tqdm.write("="*60)
    temporal_path = RESULTS_DIR / "temporal_split_results.csv" if save else None
    temporal_results = cross_validation_aif.temporal_split_cv(clean_trials, save_path=temporal_path)
    
    # Stage 6: RL baselines
    tqdm.write("\n" + "="*60)
    tqdm.write("Running Reinforcement Learning Baselines")
    tqdm.write("="*60)
    rl_results = rl_baselines.run_rl_baselines(trials=clean_trials, save=save)
    
    # Stage 7: Combine results
    with tqdm(total=1, desc="Combining results", unit="step") as pbar:
        loso_combined = pd.concat([loso_results, rl_results], ignore_index=True)
        pbar.update(1)

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
        with tqdm(total=4, desc="Saving final results", unit="file") as pbar:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            pbar.update(1)
            
            # LOSO and temporal results already saved incrementally, just confirm
            loso_path = RESULTS_DIR / "loso_results.csv"
            if loso_path.exists():
                tqdm.write(f"  → LOSO results already saved incrementally to {loso_path}")
            else:
                loso_results.to_csv(loso_path, index=False)
                tqdm.write(f"  → Saved LOSO results to {loso_path}")
            pbar.update(1)
            
            temporal_path = RESULTS_DIR / "temporal_split_results.csv"
            if temporal_path.exists():
                tqdm.write(f"  → Temporal split results already saved incrementally to {temporal_path}")
            else:
                temporal_results.to_csv(temporal_path, index=False)
                tqdm.write(f"  → Saved temporal split results to {temporal_path}")
            pbar.update(1)
            
            rl_path = RESULTS_DIR / "rl_loso_results.csv"
            rl_results.to_csv(rl_path, index=False)
            pbar.update(1)
            tqdm.write(f"  → Saved RL baseline results to {rl_path}")
            
            loso_all_path = RESULTS_DIR / "loso_results_all.csv"
            loso_combined.to_csv(loso_all_path, index=False)
            tqdm.write(f"  → Saved combined results to {loso_all_path}")
            
            qc_path = RESULTS_DIR / "qc_summary.json"
            with open(qc_path, "w", encoding="utf-8") as f:
                json.dump(outputs["qc"], f, indent=2)
            tqdm.write(f"  → Saved QC summary to {qc_path}")

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


