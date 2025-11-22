# Experimental Validation Plan: Profile-Based Active Inference on Human Changepoint Data

## Objective
Validate whether the profile-based model (M3) provides superior out-of-sample prediction of human learning behavior compared to static (M1) and entropy-coupled (M2) baselines using the McGuire et al. (2014) helicopter-bucket changepoint task dataset.

---

## Dataset Overview

**Source**: McGuire, Nassar, Gold & Kable (2014) Neuron paper  
**Task**: Spatial prediction with hidden changepoints  
**N**: 32 subjects  
**Trials**: ~480 per subject (4 runs × 120 trials)  
**Structure**: Each trial contains:
- `prediction_t`: Bucket position (subject's spatial prediction)
- `outcome_t`: Bag position (actual outcome)
- `delta_t`: Prediction error = outcome_t - prediction_t
- `update_t`: Belief update = prediction_{t+1} - prediction_t
- `reward_value`: Binary (high/neutral) - task-irrelevant
- `noise_condition`: Low (SD=10) or High (SD=25)
- `RT`: Reaction time (optional, for mechanistic validation)

**Ground truth changepoints**: Known from generative process (when helicopter moves)

---

## Data Preprocessing

### 1. **Data Quality Control**
```python
# Per subject, per run:
# - Remove first trial (no prediction error)
# - Remove last trial (no update observable)
# - Flag outlier trials:
#   - |update| > 3*SD of subject's updates
#   - RT < 200ms or RT > 5s (if using RT)
#   - Prediction outside valid range [0, 300]
# - Exclude subjects with >20% invalid trials
```

### 2. **Compute Derived Variables**

**A. Normalize spatial coordinates**
```python
# McGuire et al. used 0-300 screen units
# Keep this scale for compatibility with their model
# OR normalize to [0, 1] for numerical stability
```

**B. Compute trial-wise learning rate**
```python
learning_rate_t = update_t / delta_t  # when delta_t != 0
# Handle edge cases:
# - delta_t ≈ 0: set learning_rate = 0 or exclude trial
# - Clip learning_rate to [0, 2] to handle noise
```

**C. Compute model-predicted CPP and RU**
```python
# Use Nassar et al. (2012) approximate Bayesian model
# For each subject's observed sequence:
# - Run forward model with known parameters:
#   - hazard_rate = 0.1 (from paper)
#   - sigma_noise = 10 (low) or 25 (high) per run
# - Extract trial-wise CPP and RU

# CPP (change-point probability):
CPP_t = p(changepoint | outcome_t, belief_{t-1})
# Spikes when outcome is surprising

# RU (relative uncertainty):  
RU_t = uncertainty_belief / (uncertainty_belief + uncertainty_noise)
# High after changepoints, decays gradually
```

**D. Compute belief entropy** (for M2)
```python
# If implementing entropy-coupled M2:
# Extract belief distribution over helicopter location from forward model
# H_t = -sum(p_i * log(p_i)) over discretized belief states
```

### 3. **Data Splits for Cross-Validation**

**Option A: Leave-One-Subject-Out (LOSO)**
```python
for subject_i in subjects:
    train_data = all subjects except subject_i
    test_data = subject_i
    # Fit models on train, evaluate on test
```

**Option B: Within-Subject Temporal Split**
```python
for subject in subjects:
    train_trials = trials[0:240]  # First 2 runs
    test_trials = trials[240:480]  # Last 2 runs
    # Fit on first half, predict second half
```

**Option C: K-Fold Within-Subject**
```python
for subject in subjects:
    # 5-fold cross-validation within subject
    # Ensures temporal structure preserved within folds
```

**Recommendation**: Use **LOSO** as primary analysis (tests generalization across people), supplement with **within-subject temporal split** to test generalization across time.

---

## Model Specifications

### Adaptation to Changepoint Task

**Key difference from your bandit task**: 
- Your task: Discrete actions (left/right/hint), categorical outcomes (reward/loss)
- This task: Continuous predictions (bucket position), continuous outcomes (bag position)

**Solution**: Models predict **learning rate** rather than action probabilities.

### Model 1 (M1): Static Learning Rate
```python
# Fixed learning rate across all trials
update_t = alpha_fixed * delta_t

# Free parameters: 
# - alpha_fixed: learning rate in [0, 1]

# Grid search:
alpha_candidates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# Fit: maximize log-likelihood of observed updates given deltas
```

### Model 2 (M2): Entropy-Coupled Dynamic Learning Rate  
```python
# Learning rate adapts based on belief uncertainty
# Two sub-variants to test:

# M2a: CPP-driven (from Nassar model)
alpha_t = CPP_t  # or alpha_base * CPP_t

# M2b: RU-driven  
alpha_t = RU_t  # or alpha_base * RU_t

# M2c: Combined (Nassar normative)
alpha_t = CPP_t + RU_t * (1 - CPP_t)

# Free parameters:
# - alpha_base (optional scaling)

# Grid search:
alpha_base_candidates = [0.5, 1.0, 1.5, 2.0]
```

### Model 3 (M3): Profile-Based State-Conditional Learning

**Conceptual mapping**:
- Hidden states in bandit → Helicopter location states in changepoint
- Profiles → Behavioral modes (exploratory vs exploitative)

**Implementation approach** (2 options):

**Option A: Discrete latent context**
```python
# Discretize helicopter position into K regions
# Assign profiles to regions via Z matrix
# Mix profiles based on belief distribution over regions

# Example with K=2 profiles, 10 spatial regions:
Z = np.array([
    [1.0, 0.0],  # Region 0-29: Profile 0 (high learning)
    [0.8, 0.2],  # Region 30-59: Mostly profile 0
    ...
    [0.0, 1.0],  # Region 270-300: Profile 1 (high learning)
])

# Profile weights:
w_t = belief_over_regions_t @ Z

# Effective learning rate:
alpha_t = w_t @ [alpha_profile_0, alpha_profile_1]
```

**Option B: Change-state abstraction (SIMPLER - RECOMMENDED)**
```python
# Two latent contexts:
# - Context 0: "Stable" (no recent changepoint)
# - Context 1: "Volatile" (recent changepoint)

# Belief over contexts derived from CPP:
belief_stable_t = 1 - CPP_t
belief_volatile_t = CPP_t

# Two profiles:
# - Profile 0: Low learning rate (for stable periods)
# - Profile 1: High learning rate (for volatile periods)

# Hard assignment:
Z = [[1.0, 0.0],   # Stable → Profile 0
     [0.0, 1.0]]   # Volatile → Profile 1

# OR soft assignment:
Z = [[0.8, 0.2],   # Stable → mostly Profile 0
     [0.2, 0.8]]   # Volatile → mostly Profile 1

# Profile weights:
w_t = [belief_stable_t, belief_volatile_t] @ Z

# Effective learning rate:
alpha_t = w_t[0] * alpha_profile_0 + w_t[1] * alpha_profile_1

# Free parameters:
# - alpha_profile_0: learning rate for stable profile
# - alpha_profile_1: learning rate for volatile profile
# - Z (optional: could fix or fit)

# Grid search:
alpha_0_candidates = [0.05, 0.1, 0.2, 0.3]  # Low for stable
alpha_1_candidates = [0.4, 0.6, 0.8, 1.0]   # High for volatile
```

**Why Option B works**: 
- CPP already captures "change-state" belief
- Profiles encode appropriate responses to stable vs volatile contexts
- Mirrors your bandit framework: profiles linked to contexts, mixed by beliefs

---

## Likelihood Computation

For all models, compute log-likelihood of observed updates:

```python
# Assume Gaussian observation model:
# update_t ~ Normal(predicted_update_t, sigma_obs^2)

predicted_update_t = alpha_t * delta_t

log_likelihood_t = -0.5 * log(2*pi*sigma_obs^2) - 
                   0.5 * ((update_t - predicted_update_t)^2 / sigma_obs^2)

total_log_likelihood = sum(log_likelihood_t for all valid trials)

# Treat sigma_obs as:
# Option 1: Free parameter (fit per subject)
# Option 2: Fixed nuisance parameter (use empirical SD of residuals)
# Recommendation: Option 2 to reduce overfitting
```

---

## Cross-Validation Procedure

### LOSO Cross-Validation

```python
# Pseudocode
all_subjects = load_data()
results = {model: [] for model in ['M1', 'M2', 'M3']}

for test_subject in all_subjects:
    train_subjects = [s for s in all_subjects if s != test_subject]
    
    # Fit each model on training subjects
    for model in ['M1', 'M2', 'M3']:
        # Grid search over parameter space
        best_params = None
        best_train_ll = -inf
        
        for params in parameter_grid[model]:
            train_ll = 0
            for train_subject in train_subjects:
                train_ll += compute_likelihood(
                    model, params, train_subject.data
                )
            
            if train_ll > best_train_ll:
                best_train_ll = train_ll
                best_params = params
        
        # Evaluate best params on held-out test subject
        test_ll = compute_likelihood(
            model, best_params, test_subject.data
        )
        
        results[model].append({
            'test_subject': test_subject.id,
            'test_ll': test_ll,
            'best_params': best_params
        })

# Aggregate results
for model in ['M1', 'M2', 'M3']:
    mean_test_ll = np.mean([r['test_ll'] for r in results[model]])
    print(f'{model}: Mean test LL = {mean_test_ll}')
```

### Within-Subject Temporal Split

```python
for subject in all_subjects:
    train_trials = subject.trials[:240]
    test_trials = subject.trials[240:]
    
    for model in ['M1', 'M2', 'M3']:
        # Fit on train_trials
        best_params = grid_search(model, train_trials)
        
        # Evaluate on test_trials
        test_ll = compute_likelihood(model, best_params, test_trials)
        
        results[model][subject.id] = test_ll
```

---

## Statistical Analysis

### 1. **Primary Comparison: Held-Out Log-Likelihood**

```python
# Paired t-tests across subjects (LOSO) or runs (temporal split)
from scipy.stats import ttest_rel

ll_m1 = [results['M1'][i]['test_ll'] for i in range(N)]
ll_m2 = [results['M2'][i]['test_ll'] for i in range(N)]
ll_m3 = [results['M3'][i]['test_ll'] for i in range(N)]

# M3 vs M1
t_stat, p_value = ttest_rel(ll_m3, ll_m1)
delta_ll = np.mean(np.array(ll_m3) - np.array(ll_m1))
print(f'M3 vs M1: ΔLL = {delta_ll:.2f}, p = {p_value:.4f}')

# M3 vs M2
t_stat, p_value = ttest_rel(ll_m3, ll_m2)
delta_ll = np.mean(np.array(ll_m3) - np.array(ll_m2))
print(f'M3 vs M2: ΔLL = {delta_ll:.2f}, p = {p_value:.4f}')

# Effect size (Cohen's d)
def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x)**2 + np.std(y)**2) / 2)
```

### 2. **Information Criteria (BIC/AIC)**

```python
# Per subject, compute:
k_m1 = 1  # alpha_fixed
k_m2 = 1  # alpha_base (if CPP/RU fixed from normative model)
k_m3 = 2  # alpha_profile_0, alpha_profile_1

n_trials = len(test_trials)

AIC = -2 * test_ll + 2 * k
BIC = -2 * test_ll + k * np.log(n_trials)

# Compare mean BIC across subjects
```

### 3. **Bootstrap Confidence Intervals**

```python
# Bootstrap 95% CI for ΔLL
n_bootstrap = 10000
bootstrap_deltas = []

for _ in range(n_bootstrap):
    sample_idx = np.random.choice(N, size=N, replace=True)
    delta = np.mean(ll_m3[sample_idx]) - np.mean(ll_m1[sample_idx])
    bootstrap_deltas.append(delta)

ci_lower = np.percentile(bootstrap_deltas, 2.5)
ci_upper = np.percentile(bootstrap_deltas, 97.5)
```

---

## Mechanistic Validation (Beyond LL)

### 1. **Learning Rate Dynamics Around Changepoints**

```python
# Align trials to changepoint events
# Extract model-predicted learning rates
# Compare temporal profiles

window = [-10, +20]  # trials before/after changepoint

for model in ['M1', 'M2', 'M3']:
    aligned_alphas = []
    for subject in subjects:
        for cp_trial in changepoints:
            alpha_window = model.predict_alpha(
                trials[cp_trial+window[0]:cp_trial+window[1]]
            )
            aligned_alphas.append(alpha_window)
    
    mean_alpha = np.mean(aligned_alphas, axis=0)
    plot(window, mean_alpha, label=model)

# Expected pattern:
# - M1: Flat (no adaptation)
# - M2: Gradual rise then decay (entropy-coupled)
# - M3: Sharp rise then rapid decay (belief-weighted profile switch)
```

### 2. **Reaction Time Predictions** (if RT data available)

```python
# Test prediction: RT should correlate with model uncertainty
# M2 prediction: RT ~ H(belief)
# M3 prediction: RT ~ profile_mixing_entropy

# For M3:
profile_mixing_entropy_t = -sum(w_t * log(w_t))
# High when w_t ≈ [0.5, 0.5] (uncertain which profile)
# Low when w_t ≈ [1, 0] or [0, 1] (confident)

# Regression:
RT ~ profile_mixing_entropy + controls

# Compare to M2:
RT ~ belief_entropy + controls
```

### 3. **Individual Differences**

```python
# Extract subject-level fitted parameters
# Correlate with behavioral flexibility metrics:
# - Mean learning rate around changepoints
# - Adaptation speed (trials to re-stabilize after CP)
# - Total reward earned

# Test: Do M3's profile parameters predict flexibility better than M1/M2?
```

---

## Expected Outputs

### 1. **Main Results Table**

```
Model | k | Mean Test LL | SE | Mean BIC | ΔBIC vs M1
------|---|--------------|-----|----------|------------
M1    | 1 | -XXX.XX     | X.X | XXXX.X   | 0
M2    | 1 | -XXX.XX     | X.X | XXXX.X   | ±XX
M3    | 2 | -XXX.XX     | X.X | XXXX.X   | -XX (better)
```

### 2. **Statistical Tests**

```
Paired comparisons (N=32 subjects):
M3 vs M1: ΔLL = +XX.X (95% CI: [XX, XX]), t(31) = X.XX, p < 0.001
M3 vs M2: ΔLL = +XX.X (95% CI: [XX, XX]), t(31) = X.XX, p < 0.05
```

### 3. **Visualization Figures**

**Figure 1: Cross-Validated Performance**
- Box plots of per-subject test LL for M1, M2, M3
- Individual subject lines showing improvement

**Figure 2: Learning Rate Dynamics**
- Time courses aligned to changepoints
- M1 (flat), M2 (gradual), M3 (sharp) profiles

**Figure 3: Profile Recruitment (M3 only)**
- Heatmap: Time × Profile weight
- Aligned to changepoints
- Shows rapid switch from stable→volatile profile

**Figure 4: Individual Differences**
- Scatter: M3 profile parameters vs behavioral flexibility
- Compare to M1/M2 parameter correlations

---

## Implementation Checklist

### Phase 1: Data Loading & Preprocessing (Week 1)
- [ ] Load McGuire et al. data (request from authors)
- [ ] Parse trial structure (predictions, outcomes, updates)
- [ ] Compute derived variables (CPP, RU via Nassar model)
- [ ] Quality control (outlier removal)
- [ ] Verify data integrity (N subjects, trials per subject)

### Phase 2: Model Implementation (Week 2)
- [ ] Implement M1 likelihood computation
- [ ] Implement M2 variants (CPP-driven, RU-driven, combined)
- [ ] Implement M3 with change-state profiles
- [ ] Test on synthetic data (sanity checks)
- [ ] Verify parameter recovery (fit→generate→refit)

### Phase 3: Cross-Validation (Week 3)
- [ ] Implement LOSO CV loop
- [ ] Implement within-subject temporal split
- [ ] Grid search for each model
- [ ] Parallelize across subjects (if possible)
- [ ] Save all fitted parameters and test LLs

### Phase 4: Analysis & Visualization (Week 4)
- [ ] Statistical tests (paired t-tests, effect sizes)
- [ ] Information criteria (AIC/BIC)
- [ ] Bootstrap CIs
- [ ] Generate all figures
- [ ] Mechanistic validation (CP-aligned dynamics)
- [ ] Write results summary

---

## Code Structure Recommendations

```
experiments/
├── nassar_validation/
│   ├── data/
│   │   ├── raw/                    # Original data from authors
│   │   ├── processed/              # Preprocessed trial data
│   │   └── derivatives/            # CPP, RU computed
│   ├── models/
│   │   ├── base_model.py          # Abstract model class
│   │   ├── m1_static.py
│   │   ├── m2_dynamic.py
│   │   └── m3_profiles.py
│   ├── analysis/
│   │   ├── preprocessing.py
│   │   ├── cross_validation.py
│   │   ├── statistics.py
│   │   └── visualization.py
│   ├── utils/
│   │   ├── nassar_model.py        # Compute CPP/RU
│   │   └── helpers.py
│   └── run_validation.py          # Main script
```

---

## Troubleshooting Notes

**If M3 doesn't outperform M2/M1**:
1. Check CPP/RU computation (verify matches Nassar model)
2. Try different profile parameterizations (more profiles, different Z)
3. Consider hierarchical extension (profile selection as hidden state)
4. Test on subset with strongest changepoint effects

**If results are noisy**:
1. Increase training data (use all subjects in LOSO train set)
2. Add priors/regularization to parameter fitting
3. Use Bayesian model comparison (marginal likelihood) instead of max likelihood
4. Filter subjects by task engagement (exclude low performers)

**Computational efficiency**:
- Grid search is embarrassingly parallel → use `multiprocessing`
- Pre-compute CPP/RU once, reuse across models
- Cache intermediate results (fitted params per fold)

---
