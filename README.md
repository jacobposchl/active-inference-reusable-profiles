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
├── README.md                     
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


## Usage

### Model Recovery (Main Experiment)

Launch the cross-validated recovery study

```bash
python src/experiments/model_recovery.py \
  --generators M1,M2,M3,egreedy,softmax \
  --seed 42 \
  --folds 5 \
  --reserve-cores 10
```

**Common arguments:**
- `--generators`: Comma-separated list of behavioral generators (default: `M1,M2,M3,egreedy,softmax`)
- `--seed`: Random seed for reproducibility (default: `42`)
- `--folds`: Number of cross-validation folds (default: `5`)
- `--runs-per-generator`: Number of runs per generator (default: `5`)
- `--num-trials`: Number of trials per run (default: `400`)
- `--reserve-cores`: Number of CPU cores to reserve for system (default: `10`)
- `--resume`: Resume from existing results, skipping completed combinations

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


### Models

**M1: Static Global**
- Fixed policy precision (γ)
- Fixed outcome preferences (C)
- No adaptation to uncertainty

**M2: Dynamic Global Precision**
- Policy precision decreases with belief entropy
- Global adaptation based on uncertainty
- Formula: γ(t) = γ_base / (1 + k·H(q(better_arm)))

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

## Pymdp citation

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
