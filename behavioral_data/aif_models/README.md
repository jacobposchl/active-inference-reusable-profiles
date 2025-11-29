# Active Inference Models for Changepoint Task

This module implements proper active inference agents for the changepoint behavioral task. These models:

1. **Track full spatial beliefs** over helicopter position (discretized 0-300 space)
2. **Generate bucket predictions** from beliefs using profile-based strategies
3. **Evaluate on prediction log-likelihood** (matching RL baselines)
4. **Perform actual active inference** (belief updating, prediction generation)

## Architecture

### State Space
- **Position**: Discretized into bins (default: 30 bins covering 0-300)
- **Volatility**: Binary state (stable vs. volatile)

### Generative Model
- **A matrix**: `p(observed_bag | helicopter_position)` - Gaussian likelihood
- **B matrix**: `p(position_t+1 | position_t)` - Changepoint dynamics
- **D vectors**: Prior beliefs over initial states

### Models

#### M1: Static Model
- Fixed precision (`gamma`)
- Predicts belief mean
- No adaptation

#### M2: Entropy-Coupled Model
- Precision adapts to position belief entropy
- Higher uncertainty → lower precision → more exploration
- Dynamic prediction variance

#### M3: Profile-Based Model
- Two profiles (stable vs. volatile)
- Profile mixing via Z matrix based on volatility beliefs
- Each profile encodes different prediction strategies:
  - **Stable profile**: High precision, low exploration, predict mean
  - **Volatile profile**: Low precision, high exploration, recent-bias prediction

### Prediction Generation
- Converts beliefs + profiles → bucket position predictions
- Includes exploration noise (inversely related to precision)
- Returns both mean and uncertainty (std)

### Evaluation
- Computes log-likelihood of human predictions under model's prediction distribution
- Matches RL baseline evaluation metric exactly

## Usage

```python
from behavioral_data.aif_models import (
    ChangepointAgent,
    AgentConfig,
    make_value_fn,
    PredictionProfile,
)

# Create value function
value_fn = make_value_fn("M3", profiles=[...], Z=...)

# Create agent
config = AgentConfig(n_position_bins=30, noise_sd=10.0)
agent = ChangepointAgent(value_fn, config)

# Generate prediction
pred_mean, pred_std = agent.predict()

# Observe outcome and update
agent.observe(observed_bag_position)
```

## Cross-Validation

Use `cross_validation_aif.py` for model evaluation:

```python
from behavioral_data.pipeline.cross_validation_aif import loso_cv

results = loso_cv(trials_df)
```

This evaluates models on prediction log-likelihood, enabling fair comparison with RL baselines.

## Key Differences from Old Implementation

1. **Proper state space**: Full position beliefs, not collapsed binary
2. **Actual inference**: Bayesian belief updating, not just scalar extraction
3. **Prediction generation**: Models predict bucket positions, not learning rates
4. **Correct evaluation**: Log-likelihood of predictions, not update magnitudes
5. **Profile strategies**: Profiles encode behavioral strategies, not just parameters

