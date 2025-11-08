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
│   │   ├── plotting.py          # All visualization functions
│   │   └── helpers.py           # Helper functions (entropy, rolling mean, etc.)
│   └── experiments/
│       ├── __init__.py
│       ├── run_single_trial.py      # Basic single trial execution
│       ├── model_comparison.py      # Main model comparison experiments
│       ├── aic_bic_analysis.py      # Information criteria analysis
│       ├── noise_robustness.py      # Robustness across environmental noise
│       ├── precision_dynamics.py    # Mechanistic validation plots
│       └── parameter_fitting.py     # Subject-level parameter fitting
├── notebooks/
│   └── profile_demo.ipynb       # Interactive demonstration notebook
├── results/
│   ├── figures/                 # Generated plots
│   └── data/                    # Saved experimental data
└── tests/
    └── __init__.py              # Unit tests (to be implemented)
```

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run a basic model comparison:

```python
from src.experiments.model_comparison import run_full_comparison

# Run comparison with 20 seeds
results, analysis_ll, analysis_acc = run_full_comparison(
    n_seeds=20,
    trials=200,
    reversal_schedule=[30, 60, 90, 120, 150, 180]
)
```

### Run Individual Experiments

```python
# AIC/BIC comparison
from src.experiments.aic_bic_analysis import run_aic_bic_comparison
results, model_params = run_aic_bic_comparison(n_seeds=20, trials=200)

# Noise robustness test
from src.experiments.noise_robustness import run_noise_robustness_test
results_noise = run_noise_robustness_test(n_seeds=15, trials=200)

# Precision dynamics analysis
from src.experiments.precision_dynamics import run_precision_dynamics_comparison
trajectories = run_precision_dynamics_comparison(seed=42, trials=200)
```

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

## License

[Specify your license here]

## Author

[Your name/information]
