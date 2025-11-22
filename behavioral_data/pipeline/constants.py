"""
Shared constants for the behavioral changepoint pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Resolve important paths relative to this file so everything stays inside the
# behavioral_data folder as requested.
PIPELINE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PIPELINE_ROOT.parent
RAW_DATA_ROOT = DATA_ROOT / "raw_data"

# Public dataset metadata
PARTICIPANTS_FILE = RAW_DATA_ROOT / "participants.tsv"
EVENTS_GLOB = "sub-*/func/*task-changepoint_run-*_events.tsv"

# Derived-data folders (kept within behavioral_data)
DERIVATIVES_DIR = DATA_ROOT / "derivatives"
SIGNALS_DIR = DERIVATIVES_DIR / "normative_signals"
RESULTS_DIR = DERIVATIVES_DIR / "analysis"
for folder in (DERIVATIVES_DIR, SIGNALS_DIR, RESULTS_DIR):
    os.makedirs(folder, exist_ok=True)

# Task-specific constants
SCREEN_MIN = 0.0
SCREEN_MAX = 300.0
LEARNING_RATE_CLIP = (0.0, 2.0)
UPDATE_Z_THRESHOLD = 3.0
MAX_INVALID_TRIAL_FRACTION = 0.2

# Normative model defaults
HAZARD_RATE = 0.1
STATE_GRID = np.linspace(SCREEN_MIN, SCREEN_MAX, int(SCREEN_MAX - SCREEN_MIN) + 1)
UNIFORM_PRIOR = np.ones_like(STATE_GRID) / STATE_GRID.size

# Noise labels used in metadata
NOISE_LABELS = {
    10: "low",
    25: "high",
}


