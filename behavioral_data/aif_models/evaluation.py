"""
Evaluation utilities for changepoint active inference agents.

Computes log-likelihood of predictions matching RL baseline evaluation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

from .agent_pymdp import ChangepointAgentPymdp


def compute_prediction_ll(
    predicted_mean: float,
    predicted_std: float,
    observed_prediction: float,
) -> float:
    """
    Compute log-likelihood of observed prediction under model's prediction distribution.
    
    This matches the evaluation metric used for RL baselines.
    
    Parameters
    ----------
    predicted_mean : float
        Model's predicted bucket position (mean)
    predicted_std : float
        Model's prediction uncertainty (standard deviation)
    observed_prediction : float
        Human's actual bucket position prediction
    
    Returns
    -------
    log_likelihood : float
        Log-probability of observed prediction under model distribution
    """
    # Use Gaussian log-likelihood
    ll = norm.logpdf(observed_prediction, loc=predicted_mean, scale=predicted_std)
    return float(ll)


def evaluate_agent(
    agent: ChangepointAgentPymdp,
    outcomes: np.ndarray,
    predictions: np.ndarray,
    reset_between_runs: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate agent on a sequence of trials.
    
    Parameters
    ----------
    agent : ChangepointAgentPymdp
        Agent to evaluate
    outcomes : np.ndarray
        Observed bag positions (outcomes)
    predictions : np.ndarray
        Human bucket position predictions (what we're evaluating)
    reset_between_runs : bool
        Whether to reset agent between runs (if multiple runs)
    
    Returns
    -------
    total_ll : float
        Total log-likelihood across all trials
    trial_lls : np.ndarray
        Per-trial log-likelihoods
    """
    if reset_between_runs:
        agent.reset()
    
    trial_lls = []
    
    for t in range(len(outcomes)):
        # Agent makes prediction
        pred_mean, pred_std = agent.predict()
        
        # Compute likelihood of human's prediction
        human_pred = predictions[t]
        ll = compute_prediction_ll(pred_mean, pred_std, human_pred)
        trial_lls.append(ll)
        
        # Agent observes outcome and updates
        agent.observe(outcomes[t])
    
    return float(np.sum(trial_lls)), np.array(trial_lls)


def evaluate_agent_on_dataframe(
    agent: ChangepointAgentPymdp,
    df,
    outcome_col: str = "outcome",
    prediction_col: str = "prediction",
    group_by: Optional[str] = None,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate agent on a pandas DataFrame.
    
    Parameters
    ----------
    agent : ChangepointAgentPymdp
        Agent to evaluate
    df : pd.DataFrame
        Trial data with outcome and prediction columns
    outcome_col : str
        Column name for outcomes
    prediction_col : str
        Column name for predictions
    group_by : str, optional
        Column to group by (e.g., "run_id") - resets agent between groups
    
    Returns
    -------
    total_ll : float
        Total log-likelihood
    trial_lls : np.ndarray
        Per-trial log-likelihoods
    """
    if group_by is None:
        # Single sequence
        outcomes = df[outcome_col].to_numpy()
        predictions = df[prediction_col].to_numpy()
        return evaluate_agent(agent, outcomes, predictions, reset_between_runs=False)
    
    else:
        # Multiple groups (e.g., runs)
        all_lls = []
        for group_id, group_df in df.groupby(group_by):
            agent.reset()
            outcomes = group_df[outcome_col].to_numpy()
            predictions = group_df[prediction_col].to_numpy()
            _, group_lls = evaluate_agent(agent, outcomes, predictions, reset_between_runs=False)
            all_lls.extend(group_lls)
        
        return float(np.sum(all_lls)), np.array(all_lls)

