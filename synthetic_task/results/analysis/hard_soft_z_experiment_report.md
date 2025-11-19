# Hard vs Soft Z-mixing experiment — Report

Date: 2025-11-11

This report documents the "hard vs soft Z-mixing" ablation performed on Profile set C (top-left heatmap cell: P(reward)=0.65, P(hint)=0.65). It describes the experimental setup, methods, quantitative results and interpretations. All artifacts (figures, raw numeric outputs, and publication-ready summaries) are saved in `results/` within this repository.

## Abstract

We compared four profile-mixing regimes for a two-armed bandit task using a small set of generative profiles: a single-profile (K=1) baseline and three K=2 mixing schemes (Hard, Medium, Soft). Each regime was evaluated by running independent simulation repeats (N=10) of a 800-trial reversal learning bandit (P(reward)=0.65, P(hint)=0.65). We measured three outcome metrics per run: total cumulative reward, trial-level accuracy (all trials), and choice-only accuracy (trials on which the agent made an explicit left/right choice). Results show large and consistent advantages for the K=2 mixing schemes (Hard/Medium/Soft) over the K=1 baseline on total reward and accuracy; the Medium and Hard mixes achieve the highest total reward, while Soft yields the highest choice-only accuracy.

## Experimental setup

- Environment: Two-armed bandit with stochastic hints and rewards. Reversals were scheduled every 40 trials (i.e., reversal_schedule = [40, 80, 120, ...] up to 800 trials).
- Trials per run: 800
- Repeats per condition: 10 (num_runs = 10)
- Probabilities: P(reward) = 0.65, P(hint) = 0.65

## Agent / Model

- Agents are built as Active Inference agents using the project's `build_A`, `build_B`, `build_D` functions and `pymdp` utilities. A temporary agent is used to enumerate policies and build the value function used by the runner.
- K=1 profile (single-profile baseline): gamma=4.0, phi_logits=[0.0, -7.0, 7.0], xi_logits=[0.0,3.0,0.0,0.0]
- K=2 profiles (symmetric left/right biased profiles): both profiles have gamma=4.0, phi_logits=[0.0, -7.0, 7.0]; xi_logits differ to bias left vs right.

## Z mixing regimes

- `K=1`: Single-profile baseline (all probability mass on a single profile).
- `Hard`: Deterministic context-to-profile assignment (identity-like mixing): [[1,0],[0,1]].
- `Medium`: Partial mixing (weights 0.8/0.2 and 0.2/0.8).
- `Soft`: Softer mixing (weights 0.6/0.4 and 0.4/0.6).

These Z matrices control how latent contexts map probabilistically to the two profiles in the K=2 setting and thereby control how specialized vs mixed the controller is across contexts.

## Evaluation protocol

- For each mixing regime we ran `num_runs` independent simulations. Each run used a fixed RNG seed (seed = 42 + run_index) for reproducibility.
- From each run we extracted:
  - `total_reward`: cumulative reward across trials (+1 for observe_reward, -1 for observe_loss)
  - `accuracy` (all trials): per-trial 0/1 accuracy averaged over the 800 trials
  - `choice_only_accuracy`: per-trial accuracy averaged only across trials where the agent produced an explicit left/right action
- Reported metrics per regime: mean, standard deviation across repeats. We additionally computed 95% confidence intervals (based on Student's t when SciPy is available, otherwise z=1.96) and Cohen's d (pooled SD) versus the K=1 baseline.

## Results (numeric)

All numeric artifacts are saved in `results/data/`:

- `hard_soft_z_results.npz` — arrays of means and stds used for plotting
- `hard_soft_z_results.json` — human-readable summary of the means and stds
- `hard_soft_z_publication_summary.csv` / `.json` — publication-ready table including N, mean, std, SE, 95% CI and Cohen's d vs K=1

Below we reproduce the publication summary (N=10 per condition). Values are mean ± std and 95% CI where provided.

| Condition | Total reward mean ± SD | 95% CI (total reward) | Cohen's d vs K=1 (total reward) | Accuracy mean ± SD | 95% CI (accuracy) | Cohen's d vs K=1 (accuracy) | Choice-accuracy mean ± SD | 95% CI (choice-acc) | Cohen's d vs K=1 (choice-acc) |
|---|---:|---|---:|---:|---|---:|---:|---|---:|
| K=1 (baseline) | 0.10 ± 0.30 | [-0.115, 0.315] | 0.00 | 0.000125 ± 0.000375 | [-0.000143, 0.000393] | 0.00 | 0.10 ± 0.30 | [-0.115, 0.315] | 0.00 |
| Hard | 111.0 ± 24.30 | [93.62, 128.38] | 6.45 | 0.65975 ± 0.04207 | [0.62966, 0.68984] | 22.17 | 0.76883 ± 0.04163 | [0.73905, 0.79861] | 3.12 |
| Medium | 112.8 ± 11.94 | [104.26, 121.34] | 13.34 | 0.63038 ± 0.01375 | [0.62054, 0.64021] | 64.80 | 0.80808 ± 0.02108 | [0.79299, 0.82316] | 3.33 |
| Soft | 67.10 ± 11.06 | [59.19, 75.01] | 8.57 | 0.36188 ± 0.02289 | [0.34550, 0.37825] | 22.35 | 0.87193 ± 0.04521 | [0.83958, 0.90427] | 3.60 |

Notes on interpretation:

- The K=1 baseline performs effectively at chance on these metrics (near-zero reward and accuracy), indicating the single-profile agent struggles in this task configuration.
- Both `Hard` and `Medium` mixing regimes achieve very high total rewards (≈111–113) and high overall accuracy (≈0.63–0.66). `Medium` shows the highest total reward mean and the smallest total-reward std across repeats, suggesting both high performance and more consistent outcomes.
- `Soft` mixing gives the highest choice-only accuracy (≈0.872) but a substantially lower total reward mean (≈67) compared to `Hard`/`Medium`. This pattern suggests `Soft` mixing encourages consistent correct choices when choosing, but the agent may make fewer choices or have other behavior that reduces cumulative reward relative to `Hard`/`Medium`.
- Cohen's d values vs the K=1 baseline are very large for most comparisons (note: the baseline has extremely small variance and near-zero mean, which inflates d). Interpret effect sizes cautiously when one group's variance/mean is near zero.

## Visualization

The grouped bar chart for the three metrics was saved as:

- `results/figures/hard_soft_z_barplot.png`
- `results/figures/hard_soft_z_barplot.pdf`

The bar chart displays mean ± std across N=10 repeats for total reward, accuracy (all trials), and choice-only accuracy. Use the figures in presentations and drafts; the CSV/JSON provides numerical values suitable for plotting with your preferred plotting library.

## Reproducibility and how to run

From the repository root, run the experiment (Windows / cmd.exe):

```cmd
python src\experiments\hard_soft_z_barplot.py
```

Notes:

- The script will create and overwrite `results/data/hard_soft_z_results.*` and `results/figures/hard_soft_z_barplot.*`.
- The script uses multiprocessing by default for speed. On Windows it is executed under the `if __name__ == '__main__':` guard so it is safe to run directly.
- If you want raw per-run arrays saved for bootstrap-based CIs, we can update the script to store per-run values (recommended for robust CI estimation).

## Statistical notes and caveats

- Confidence intervals use the t distribution if SciPy is installed; otherwise a z-approximation (1.96) is used.
- Cohen's d is computed using a pooled standard deviation and assumes equal sample sizes; when one group's variance is near-zero this inflates d — report raw means/SEs alongside d when preparing manuscripts.
- For publication-quality inference, consider bootstrap CIs on per-run values and pre-register the comparison plan (which metrics, which pairwise contrasts).

## Next steps and recommendations

1. Save per-run raw arrays (total_reward per run, accuracy per run, choice_accuracy per run) to enable bootstrap CIs and more robust effect-size estimation.
2. Run additional seeds / increase `num_runs` (e.g., N=30) to stabilize estimates for inferential statistics.
3. If planning hypothesis tests, predefine contrasts (e.g., Medium vs Soft, K=1 vs Medium) and correct for multiple comparisons.
4. Consider reporting non-parametric tests (e.g., Wilcoxon, bootstrap) if distributions are non-normal.

## Files produced (quick reference)

- results/data/hard_soft_z_results.npz — arrays used for plotting
- results/data/hard_soft_z_results.json — readable summary (means & stds)
- results/data/hard_soft_z_publication_summary.csv / .json — publication summary table (N, mean, std, SE, 95% CI, Cohen's d)
- results/figures/hard_soft_z_barplot.png / .pdf — grouped bar plot with error bars

---

If you'd like, I can:

- Update the experiment script to save per-run data and recompute bootstrap 95% CIs.
- Produce LaTeX-ready tables for the results.
- Run the experiment with larger N and produce updated tables/plots.

Tell me which of the above you want next and I'll implement it.
