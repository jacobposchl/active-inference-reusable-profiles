**Mechanistic Experiment — Analysis & Synthesis**

**Overview**:
- **Purpose**: Evaluate model behaviour (M1, M2, M3) around context reversals in a two-armed bandit-style task. Particular attention is paid to: (1) transient changes in policy precision (γ) around reversals, and (2) recruitment/weighting of reusable value profiles in model M3 after reversals.
- **Key change made**: The averaged M3 profile-weight figure (pooled across reversal directions) was removed because it pools opposite reversal directions and can obscure direction-specific effects. Direction-specific figures (switch → left_better vs switch → right_better) are produced instead.

**Files of interest**:
- `src/experiments/mechanistic_analysis.py` — runner and plotting functions. Notable functions:
  - `plot_gamma_around_reversals` — aligns γ series to reversals and plots mean ± CI.
  - `plot_m3_profile_weights_by_direction` — aligns M3 profile-weight windows by reversal direction and plots direction-specific means ± CI. (This replaces the removed averaged plot.)
  - `run_mechanistic_experiment` — orchestrates runs and saves figures to `results/figures`.
- `src/utils/helpers.py` — `find_reversals(context_series)` returns indices where context changes.
- `config/experiment_config.py` — contains experiment defaults such as `M3_DEFAULTS['Z']` used to compute profile weights.

**Experimental design**:
- **Environment**: Two-armed bandit with a context that indicates which arm is better: `left_better` or `right_better`.
- **Reversal schedule**: Example used in mechanistic runs: reversals at trials [30, 60] (configurable when calling `run_mechanistic_experiment`).
- **Trials**: Example short-run is 90 trials per run.
- **Runs**: Experiment runner uses multiple runs (default `num_runs=20`) to pool across random seeds and quantify variability.
- **Windowing**: Figures align time windows around each reversal using a `pre` and `post` window (defaults `pre=10`, `post=20`) and pool windows across runs (but, for profiles, now pool separately by post-reversal direction).

**Metrics & processing**:
- **Gamma (γ)**: Policy precision per trial, recorded in agent logs as `logs['gamma']`. Plotted as mean ± bootstrap CI across pooled windows.
- **Beliefs & profile weights (M3)**:
  - `beliefs` per trial are stacked to shape (T, n_contexts) from `logs['belief']`.
  - `Z` (profiles matrix) is read from `M3_DEFAULTS['Z']`.
  - Profile weights are computed per trial as: `w_t = beliefs_t @ Z` → time series shape (T, n_profiles).
  - Windows around reversals are extracted per-profile: `w[start:end, p]`.
- **Reversal detection**: `find_reversals(context_series)` returns indices r where `context[t] != context[t-1]` (i.e., the index of the trial after the change). The code uses `ctx[r]` as the post-reversal context label (either `left_better` or `right_better`).
- **Pooling**: For γ and accuracy, windows are pooled across all reversals and runs (because these metrics are direction-agnostic transient signals). For M3 profile weights, windows are pooled separately by the post-reversal context (direction) to avoid mixing opposing effects.
- **Confidence intervals**: Per-timepoint 95% CI computed via `bootstrap_ci` (default bootstrap draws = 2000 in `compute_mean_and_ci`) and plotted as shaded bands around the mean.

**Why not average profile weights across directions?**
- Averaging windows from switches → `left_better` with switches → `right_better` mixes cases where a profile may be recruited specifically for one direction. This can:
  - Attenuate selective signals (profile that increases only after switch→left will be averaged with switch→right where it doesn't change),
  - Create misleading symmetry that hides true directional recruitment,
  - Lead to incorrect conclusions about which profile is "on top" post-reversal.
- Direction-specific pooling preserves potential asymmetric recruitment and yields interpretable curves for each post-reversal direction.

**Results summary (what to look for in saved figures)**
- `mechanistic_accuracy_rev_<a>_<b>.png` — accuracy aligned to reversals (pooled); useful to see adaptation speed after reversals.
- `mechanistic_gamma_rev_<a>_<b>.png` — policy precision (γ) aligned to reversals (pooled); typically M2 shows a dip around reversal points, indicating transient reduction in policy confidence.
- `mechanistic_m3_profiles_rev_<a>_<b>_m3_profiles_to_left.png` — M3 profile weights aligned to reversals that switch to `left_better`.
- `mechanistic_m3_profiles_rev_<a>_<b>_m3_profiles_to_right.png` — M3 profile weights aligned to reversals that switch to `right_better`.

Interpretation guidance:
- If a particular M3 profile (e.g., Profile 1) increases substantially and consistently after `switch -> left_better` but not after `switch -> right_better`, this implies that the profile encodes a strategy or value representation specific to that post-reversal context.
- A transient dip in γ (policy precision) around reversals (seen in M2 or others) suggests reduced policy confidence while evidence accumulates post-change; that dip often precedes re-adaptation (improved accuracy) over subsequent trials.
- Compare the timing of γ dips with profile recruitment:
  - Does profile recruitment occur before γ recovers? That suggests profiles help re-establish confident policies.
  - If profile recruitment lags γ recovery, profiles may reflect slower, later-stage adaptation.

**Statistical and methodological notes**:
- `find_reversals` marks the trial index where context has already changed (the first trial in the new context). Windows are taken with the reversal index r being the first trial in the new context; thus `t=0` in the aligned window corresponds to the first trial after the change.
- Edge handling: reversals too close to start/end where [r-pre, r+post) exceeds the time series are skipped.
- Bootstrapped CI is computed per-timepoint independently (i.e., CI for each t is based on bootstrap samples of pooled values at that t). This is appropriate for visualizing uncertainty but does not correct for multiple comparisons across time.

**Recommendations & next steps**:
- Normalization: subtracting a pre-reversal baseline (e.g., mean of `t` in `[-pre, -1]`) per-window can make post-reversal changes easier to compare across runs and profiles.
- Difference plots: plot `profiles_to_left - profiles_to_right` to directly highlight asymmetric recruitment.
- Per-run traces: overlay semi-transparent per-run windows behind the mean+CI to assess variability and detect outliers.
- Direction-specific hypothesis tests: if you want to claim a profile is preferentially used for one direction, run a paired or independent test across pooled windows (e.g., compare area under curve post-reversal across directions) with permutation or bootstrap testing.
- Check whether the number of pooled windows per direction is balanced; if not, consider subsampling or weighting to avoid bias.

**Reproducibility / how to run**:
- From repository root (Windows `cmd.exe`):

```cmd
python src\experiments\mechanistic_analysis.py
```

- Generated figures will be saved to `results/figures` as described above.
- If you already have run logs and want only to re-generate figures from saved logs, we can add a small loader function to read logs from `results/data` and call the plotting functions; let me know and I will add it.

**Concluding synthesis**:
- The averaged pooled M3 profile-weight plot was removed because it can mask directional asymmetries important for interpreting M3's mechanism.
- Direction-specific alignment reveals whether certain profiles are recruited preferentially after switches to `left_better` or `right_better` and supports the hypothesis that M3 can represent reusable strategies associated with particular contexts.
- Gamma (γ) dynamics remain a robust, direction-agnostic signal of transient belief/policy uncertainty at reversals and provide complementary evidence about the timing of adaptation.

**If you want** I can:
- Run the experiment now and attach the updated figures.
- Add baseline-normalization and difference plots to the analysis.
- Implement a loader to regenerate plots from saved run logs only (no need to re-run simulations).


---
Generated on: (automatically created) 

