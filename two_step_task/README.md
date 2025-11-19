Two-step task experiment scaffolding
===================================

This folder contains initial scaffolding for implementing the two-step
task experiment described in the project outline. Current contents:

- `data_loader.py` : CSV loading and preprocessing helpers.
- `generative.py`  : builders for A and B matrices (as in the outline).
- `models/`        : stubs for `M1_StaticPrecision`, `M2_EntropyCoupled`,
                     and `M3_ProfileBased` model classes.
- `fitting/`       : scaffolding for optimization and likelihood
                     computation (agent logic implemented in `agent.py`).

Next recommended steps (I can implement these for you):

1. Add parameter search utilities (grid + refinement) and functions to
   run fitting over all participants and save results.
2. Implement multi-participant runners and parallelization.

This package now includes a compact, pymdp-style Active Inference adapter
`two_step_task.ai.pymdp_adapter.PymdpAgent` and the fitter uses policy-level
Active Inference by default. The adapter computes policy posteriors by
evaluating expected reward under each deterministic policy and applying a
precision parameter (gamma).

If you'd like, I can proceed to implement a full `pymdp`-based generator
that constructs `A`, `B`, `C` matrices and uses the external `pymdp` package
directly — for now the adapter is dependency-free and suitable for fitting.
 
Organization and small utilities:
- Raw CSVs: `two_step_task/data/raw/`
- Results: `two_step_task/results/fitted_parameters/`
- Scripts: `two_step_task/scripts/` (includes `inspect_csv_format.py`)

To inspect your CSVs quickly, run:

```cmd
.venv\\Scripts\\python.exe two_step_task\\scripts\\inspect_csv_format.py --path two_step_task\\data\\raw
```

If you plan to use the upstream `pymdp` library, install it with pip (the
project `requirements.txt` already lists `inferactively-pymdp` as a
recommended dependency):

```cmd
pip install inferactively-pymdp
```
