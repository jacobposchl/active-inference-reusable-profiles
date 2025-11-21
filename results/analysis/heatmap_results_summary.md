# Heatmap Analysis: K=1 vs K=2 Reusable Profiles

## Overview
This analysis summarizes the results of systematic experiments comparing single-profile (K=1) and multi-profile (K=2) agents across a grid of environment settings and profile configurations. The experiments were run using the `k_comparison_heatmaps.py` script, which generated heatmaps for log likelihood, accuracy, and BIC differences between K=2 and K=1 agents.

## What We Tested
- **Profile Sets:** Three sets (A, B, C) representing different agent profile configurations:
  - **A:** Baseline (neutral, no strong biases)
  - **B:** Moderate hint-seeking
  - **C:** Strong left/right bias, no hint
- **Environment Grid:**
  - **prob_reward:** [0.65, 0.75, 0.85, 0.95] (probability of reward for correct action)
  - **prob_hint:** [0.65, 0.75, 0.85, 0.95] (probability of receiving a correct hint)
- **Metrics:**
  - **Δ Log Likelihood (K=2 - K=1):** Improvement in model fit
  - **Δ Accuracy (K=2 - K=1):** Improvement in correct choices
  - **Δ BIC (K=2 - K=1):** Improvement in model selection criterion (lower is better)

## Results Summary

## How to Read the Heatmap Tables

Each heatmap table below shows the difference in a metric (log likelihood, accuracy, or BIC) between K=2 and K=1 profile agents, for a grid of environment settings. The grid is defined as follows:

- **Rows:** Different values of `prob_hint` (probability of receiving a correct hint), in order: `[0.65, 0.75, 0.85, 0.95]` (top to bottom)
- **Columns:** Different values of `prob_reward` (probability of reward for correct action), in order: `[0.65, 0.75, 0.85, 0.95]` (left to right)

For example, the value at row 2, column 3 corresponds to `prob_hint=0.75` and `prob_reward=0.85`.

**Interpretation:**
- **Δ Log Likelihood:** Positive values mean the K=2 agent fits the data better than K=1 for that environment setting.
- **Δ Accuracy:** Positive values mean the K=2 agent made more correct choices than K=1; negative means K=1 was more accurate.
- **Δ BIC:** Negative values mean the K=2 agent is preferred by the Bayesian Information Criterion (lower is better, so more negative = stronger evidence for K=2).

All values are computed as (K=2 result) minus (K=1 result), so positive means K=2 is better for that metric, negative means K=1 is better.


## Results Summary

### Profile Set A (Baseline)
**Profiles:**
- K=1: `[{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 0.0, 0.0, 0.0]}]`
- K=2: `[{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 4.0, 6.0, -6.0]}, {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 4.0, -6.0, 6.0]}]`

**Δ Log Likelihood:**
```
[[0.33584065 0.40535062 0.47211308 0.59964172]
 [0.18830506 0.24609571 0.28683956 0.34697309]
 [0.13520845 0.13470996 0.13504471 0.12715756]
 [0.04437263 0.06166657 0.08086051 0.10456701]]
```
**Δ Accuracy:**
```
[[-0.168625 -0.062125 -0.00725   0.05125 ]
 [-0.1395   -0.07525  -0.0445   -0.007125]
 [-0.018875 -0.028375 -0.025    -0.014625]
 [-0.023375 -0.033375 -0.03175  -0.02575 ]]
```
**Δ BIC:**
```
[[-5337.50158395 -6449.66113436 -7517.86041942 -9558.31873654]
 [-2976.93211934 -3901.58257661 -4553.48415473 -5515.62059165]
 [-2127.3863764  -2119.41059611 -2124.76653214 -1998.5722392 ]
 [ -674.01327632  -950.71637029 -1257.81943339 -1637.12329978]]
```

### Profile Set B (Moderate Hint-Seeking)
**Profiles:**
- K=1: `[{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 1.0, 0.0, 0.0]}]`
- K=2: `[{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 1.0, 4.0, -4.0]}, {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 1.0, -4.0, 4.0]}]`

**Δ Log Likelihood:**
```
[[0.52828197 0.55162353 0.58773704 0.58765256]
 [0.35983719 0.37572551 0.40978338 0.42155917]
 [0.21917226 0.22781946 0.24624892 0.29290221]
 [0.10067486 0.10373849 0.10979583 0.12408537]]
```
**Δ Accuracy:**
```
[[-0.000625 -0.000625 -0.01775   0.049125]
 [-0.0015   -0.0015   -0.0015   -0.00425 ]
 [ 0.00025   0.00025   0.00025  -0.00275 ]
 [-0.004    -0.004    -0.004    -0.004   ]]
```
**Δ BIC:**
```
[[-8416.56277051 -8790.02773906 -9367.84378806 -9366.49221983]
 [-5721.44618434 -5975.6594338  -6520.58523435 -6708.99786582]
 [-3470.80733578 -3609.16251314 -3904.03386359 -4650.48651254]
 [-1574.84894843 -1623.86709912 -1720.78450884 -1949.41709175]]
```

### Profile Set C (Strong Left/Right Bias)
**Profiles:**
- K=1: `[{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 3.0, 0.0, 0.0]}]`
- K=2: `[{'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 3.0, 6.0, -6.0]}, {'gamma': 4.0, 'phi_logits': [0.0, -7.0, 7.0], 'xi_logits': [0.0, 3.0, -6.0, 6.0]}]`

**Δ Log Likelihood:**
```
[[0.36505042 0.35826568 0.33333039 0.2956014 ]
 [0.64722676 0.5661036  0.50168722 0.45287885]
 [0.58417499 0.6348502  0.62007315 0.63514825]
 [0.32786766 0.34597671 0.3784212  0.44435931]]
```
**Δ Accuracy:**
```
[[ 0.659625  0.722     0.763125  0.80175 ]
 [ 0.77275   0.773375  0.812     0.85475 ]
 [ 0.010125  0.10475   0.211     0.56    ]
 [-0.003875 -0.003875  0.012375  0.005875]]
```
**Δ BIC:**
```
[[ -5804.85799474  -5696.30208889  -5297.33745565  -4693.67367223]
 [-10319.67941091  -9021.70874707  -7991.04679695  -7210.11283056]
 [ -9310.85103785 -10121.65444997  -9885.22167714 -10126.42315143]
 [ -5209.93377504  -5499.67853088  -6018.79036475  -7073.80015483]]
```
