# Reusable Value Profiles in Active Inference

This project implements and compares three models of active inference in a two-armed bandit task with reversals, investigating the role of state-dependent value profiles in adaptive behavior.

## Overview

The project investigates whether agents can improve performance by maintaining multiple reusable value profiles that are dynamically mixed based on current beliefs about hidden states. We compare three models:

- **M1 (Static Global)**: Fixed policy precision and outcome preferences
- **M2 (Dynamic Global)**: Policy precision adapts to belief entropy (global uncertainty)
- **M3 (Profile Model)**: Multiple value profiles mixed based on state-specific beliefs

## Project Structure

```
reusable_profiles/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── config/
│   └── experiment_config.py     # Global configuration parameters
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generative_model.py  # A, B, D matrices for agent beliefs
│   │   ├── value_functions.py   # M1, M2, M3 value function implementations
│   │   └── agent_wrapper.py     # AgentRunner and AgentRunnerWithLL classes
│   ├── environment/
│   │   ├── __init__.py
│   │   └── two_armed_bandit.py  # TwoArmedBandit environment class
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── recovery_helpers.py  # Shared CV/grid-search logic + artifact writers
│   │   └── plotting.py          # Visualization helpers
│   └── experiments/
│       ├── __init__.py
│       └── model_recovery.py    # K-fold CV experiment and CLI
├── test_scripts/
│   └── smoke_cv_demo.py         # Tiny wrapper around the model_recovery CLI
├── results/
│   └── model_recovery/          # Structured outputs (trial/fold/run/confusion)
└── tests/
    └── ...                      # Pytest suites for env/models/experiments
```

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Model Recovery (Main Experiment)

Launch the cross-validated recovery study via the CLI:

```bash
python src/experiments/model_recovery.py \
  --generators M1,M2,M3,egreedy,softmax \
  --runs-per-generator 20 \
  --num-trials 80 \
  --seed 1 \
  --reversal-interval 40 \
  --folds 5
```

All arguments have sensible defaults; run `--help` for details. The script prints
progress with nested `tqdm` bars and writes structured outputs to
`results/model_recovery/`.

### Smoke Test

Before running the full study you can execute a lightweight sanity check:

```bash
python test_scripts/smoke_cv_demo.py
```

This invokes the same CLI with a tiny configuration (two generators, single run)
to confirm dependencies and logging work end-to-end.

### Output Layout

After a run you will find:

- `results/model_recovery/trial_level/…` – per-trial CSVs including fold/train/test
  labels, beliefs, gamma, and action log-likelihoods.
- `results/model_recovery/fold_level/…` – fold summaries with train/test indices,
  best parameters, and grid-eval counts.
- `results/model_recovery/run_summary/…` – per-run aggregates (mean LL/accuracy,
  runtime, parameter snapshots).
- `results/model_recovery/confusion/…` – confusion matrices for log-likelihood,
  accuracy, AIC, and BIC (means + standard errors).
- `results/model_recovery/metadata/experiment_summary.json` – metadata describing
  generators, seeds, fold count, and environment settings used in the run.

## Key Concepts

### Hidden States
- **Context Factor**: Which arm is better (left_better / right_better)
- **Choice Factor**: Current position in task (start / hint / left / right)

### Models

**M1: Static Global**
- Fixed policy precision (γ)
- Fixed outcome preferences (C)
- No adaptation to uncertainty

**M2: Dynamic Global Precision**
- Policy precision decreases with belief entropy
- Global adaptation based on uncertainty
- Formula: γ(t) = γ_base / (1 + k·H(q(context)))

**M3: Profile Model**
- Multiple value profiles with different precision/preference combinations
- Profiles mixed based on current beliefs via assignment matrix Z
- State-dependent adaptive control

### Value Profiles

Each profile contains:
- `phi_logits`: Outcome preference logits (becomes C vector after softmax)
- `xi_logits`: Policy preference logits (becomes E vector)
- `gamma`: Policy precision (decisiveness parameter)

Profiles are combined via:
```
w = q(context) @ Z  # Profile weights from beliefs
C_t = softmax(w @ PHI)  # Mixed outcome preferences
gamma_t = w @ GAMMA  # Mixed precision
```

## Tests

The repository includes pytest coverage for the environment, generative model,
value functions, agent wrapper, log-likelihood utilities, recovery helpers, and
the top-level `model_recovery` experiment. Run the suite with:

```bash
pytest
```

## Citation

This work builds on the pymdp package:

```bibtex
@article{Heins2022,
  doi = {10.21105/joss.04098},
  url = {https://doi.org/10.21105/joss.04098},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {73},
  pages = {4098},
  author = {Conor Heins and Beren Millidge and Daphne Demekas and Brennan Klein and Karl Friston and Iain D. Couzin and Alexander Tschantz},
  title = {pymdp: A Python library for active inference in discrete state spaces},
  journal = {Journal of Open Source Software}
}
```



## Author

Jacob Poschl - jposchl@ucsc.edu
