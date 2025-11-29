# Tests for Behavioral Data Active Inference Models

This directory contains comprehensive tests for the pymdp-based active inference implementation for the changepoint behavioral task.

## Test Structure

- **`test_state_space.py`**: Tests for state space discretization (position bins, volatility states)
- **`test_generative_model.py`**: Tests for A, B, D matrix construction (pymdp object arrays)
- **`test_value_functions.py`**: Tests for M1, M2, M3 value functions
- **`test_agent.py`**: Tests for `ChangepointAgentPymdp` (prediction, observation, reset)
- **`test_evaluation.py`**: Tests for log-likelihood computation and agent evaluation
- **`test_cross_validation.py`**: Tests for LOSO and temporal split CV
- **`test_integration.py`**: Integration tests for full pipeline

## Running Tests

```bash
# From project root
pytest behavioral_data/tests/ -v

# Run specific test file
pytest behavioral_data/tests/test_agent.py -v

# Run with coverage
pytest behavioral_data/tests/ --cov=behavioral_data.aif_models --cov-report=html
```

## Test Coverage

The tests verify:

1. **State Space**: Correct discretization, bin conversion, volatility states
2. **Generative Model**: Proper pymdp object array format, normalization, hazard rate
3. **Value Functions**: M1 static values, M2 entropy coupling, M3 profile mixing
4. **Agent**: Initialization, prediction generation, belief updates, reset functionality
5. **Evaluation**: Log-likelihood computation, DataFrame evaluation, multi-run handling
6. **Cross-Validation**: LOSO CV, temporal split CV, parameter grid search
7. **Integration**: Full pipeline with synthetic data, all models working together

## Requirements

Tests require `pytest` (added to `requirements.txt`). All other dependencies are the same as the main codebase.

