# Reusable Value Profiles in Active Inference

Research code for the paper *Active Inference with Reusable State-Dependent Value Profiles*.
Compares three active-inference models on a two-armed bandit with reversals via cross-validated
model recovery: **M1** (static precision), **M2** (entropy-coupled precision), **M3** (belief-weighted
mixing of reusable value profiles).

## Run the main experiment
```bash
python src/experiments/model_recovery.py --generators M1,M2,M3,egreedy,softmax --seed 42 --folds 5
```
Outputs land in `results/model_recovery/run_<id>/` — per-trial / fold / run CSVs and confusion matrices.

## Map
- `config/experiment_config.py` — constants, model defaults, M3 profile templates + Z.
- `src/models/generative_model.py` — builds A / B / D.
- `src/models/value_functions.py` — M1/M2/M3 value functions (return `C_t, E_t, gamma_t`).
- `src/models/agent_wrapper.py` — drives the pymdp Agent; `action_logprob` = teacher-forced action LL.
- `src/utils/ll_eval.py` — per-trial action log-likelihood.
- `src/utils/recovery_helpers.py` — K-fold CV grid-search fitting, confusion tables, artifact writers.
- `src/experiments/model_recovery.py` — orchestration + CLI.
- `figure_scripts/` — publication figures (AIC confusion matrix; M3 mechanistic panels).

## Environment
`pip install -r requirements.txt` (built on `inferactively-pymdp==0.0.7.1`).
Model free-parameter counts used for AIC: M1 = 1, M2 = 2, M3 = 6.
