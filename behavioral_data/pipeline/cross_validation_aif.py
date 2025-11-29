"""
Cross-validation utilities for the new active inference models.

Evaluates AIF models on prediction log-likelihood (matching RL baselines).
"""
from __future__ import annotations

import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..aif_models import (
    ChangepointAgentPymdp,
    AgentConfig,
    make_value_fn,
    make_values_M1,
    make_values_M2,
    make_values_M3_with_volatility,
    evaluate_agent_on_dataframe,
)
from ..aif_models.profile_utils import create_dynamic_phi_generator
from ..pipeline.constants import NOISE_LABELS


@dataclass
class FitResult:
    model: str
    subject_id: str
    params: Dict[str, float]
    train_ll: float
    test_ll: float
    num_trials: int


def _expand_grid(grid: Dict[str, Sequence]) -> List[Dict[str, float]]:
    """Expand parameter grid into list of parameter combinations."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for product in itertools.product(*values):
        combos.append(dict(zip(keys, product)))
    return combos


def _create_agent(model: str, params: Dict[str, float], noise_sd: float, seed: Optional[int] = None) -> ChangepointAgentPymdp:
    """Create agent with specified model and parameters."""
    config = AgentConfig(
        n_position_bins=30,
        noise_sd=noise_sd,
        hazard_rate=0.1,
    )
    
    if model == "M1":
        value_fn = make_values_M1(gamma=float(params["gamma"]))
    
    elif model == "M2":
        value_fn = make_values_M2(
            gamma_base=float(params.get("gamma_base", 2.5)),
            entropy_k=float(params.get("entropy_k", 1.0)),
        )
    
    elif model == "M3":
        # Follow the exact pattern from model_utils.py for M3
        # First, build temporary A, B, D to get policies
        from ..aif_models.generative_model import build_generative_model
        from ..aif_models.state_space import create_state_space
        
        position_space, _ = create_state_space(n_position_bins=config.n_position_bins)
        A_temp, B_temp, D_temp = build_generative_model(
            position_space,
            noise_sd=noise_sd,
            hazard_rate=config.hazard_rate,
        )
        
        # Create temporary agent to get policies (same pattern as model_utils.py)
        from pymdp.agent import Agent
        from pymdp import utils as pymdp_utils
        
        C_temp = pymdp_utils.obj_array_zeros([(A_temp[m].shape[0],) for m in range(len(A_temp))])
        temp_agent = Agent(
            A=A_temp,
            B=B_temp,
            C=C_temp,
            D=D_temp,
            policy_len=config.policy_len,
            inference_horizon=config.inference_horizon,
            control_fac_idx=[0],  # Position is controllable
            use_utility=True,
            use_states_info_gain=True,
            action_selection="stochastic",
            gamma=16.0
        )
        
        policies = temp_agent.policies
        n_bins = config.n_position_bins
        num_actions_per_factor = [n_bins]  # One action factor: position predictions
        
        # Create profiles in the format expected by M3 (gamma values)
        # phi_logits will be generated dynamically based on current beliefs
        # Profile 0: Explore (volatile) - flat preferences, low precision
        # Profile 1: Exploit (stable) - peaked preferences, high precision
        profiles = [
            {
                "gamma": float(params["gamma_volatile"]),  # Low precision for exploration
            },
            {
                "gamma": float(params["gamma_stable"]),  # High precision for exploitation
            },
        ]
        
        # Z matrix: stable→profile1, volatile→profile0
        if params.get("soft_assign", False):
            Z = np.array([[0.2, 0.8], [0.8, 0.2]], dtype=float)  # Soft assignment
        else:
            Z = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)  # Hard: stable→profile1, volatile→profile0
        
        # Create dynamic phi_logits generator
        # preference_width controls how peaked the exploit profile is
        preference_width = float(params.get("preference_width", 20.0))
        dynamic_phi_generator = create_dynamic_phi_generator(preference_width=preference_width)
        
        # Create value function with dynamic profiles
        value_fn = make_values_M3_with_volatility(
            profiles=profiles,
            Z=Z,
            policies=policies,
            num_actions_per_factor=num_actions_per_factor,
            dynamic_phi_generator=dynamic_phi_generator,
        )
    
    else:
        raise ValueError(f"Unknown model: {model}")
    
    return ChangepointAgentPymdp(value_fn, config, seed=seed)


def _score(df: pd.DataFrame, model: str, params: Dict[str, float], seed: Optional[int] = None) -> float:
    """
    Compute log-likelihood score for a model on data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Trial data
    model : str
        Model name
    params : dict
        Model parameters
    seed : int, optional
        Random seed for agent initialization
    """
    # Get noise level from data (should be consistent within a run)
    noise_sd = float(df["noise_sd"].iloc[0])
    
    # Create agent
    agent = _create_agent(model, params, noise_sd, seed=seed)
    
    # Evaluate
    total_ll, _ = evaluate_agent_on_dataframe(
        agent,
        df,
        outcome_col="outcome",
        prediction_col="prediction",
        group_by="run_id",  # Reset between runs
    )
    
    return total_ll


def _score_worker_wrapper(args):
    """Worker function wrapper for parallel scoring.
    
    Note: This function must be at module level for multiprocessing.
    DataFrames are pickleable, so we can pass them directly.
    
    Uses a deterministic seed based on params hash to ensure reproducibility.
    """
    train_df, model, params = args
    try:
        # Use deterministic seed based on params to ensure reproducibility
        # Hash params dict to get a seed (convert to sorted tuple for determinism)
        params_tuple = tuple(sorted(params.items()))
        seed = hash(params_tuple) % (2**31)  # Convert to positive int
        ll = _score(train_df, model, params, seed=seed)
        return params, ll
    except Exception:
        return None, -math.inf


def _grid_search(train_df: pd.DataFrame, model: str, parameter_grid: Dict[str, Sequence], max_workers: Optional[int] = None) -> Tuple[Dict[str, float], float]:
    """
    Grid search for best parameters with optional parallel processing.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    model : str
        Model name (M1, M2, M3)
    parameter_grid : dict
        Parameter grid to search
    max_workers : int, optional
        Number of parallel workers. If None, uses half of CPU count.
    """
    param_combos = _expand_grid(parameter_grid)
    
    # Set default workers if not provided
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = max(1, cpu_count // 2)
    
    # Use parallel processing if max_workers > 1 and multiple parameter combinations
    if max_workers > 1 and len(param_combos) > 1:
        best_params = None
        best_ll = -math.inf
        
        # Prepare arguments for workers (DataFrames are pickleable)
        args_list = [(train_df, model, params) for params in param_combos]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {executor.submit(_score_worker_wrapper, args): args[2] for args in args_list}
            
            for future in as_completed(future_to_params):
                params, ll = future.result()
                if params is not None and ll > best_ll:
                    best_ll = ll
                    best_params = params
        
        return best_params or {}, best_ll
    else:
        # Sequential fallback
        best_params = None
        best_ll = -math.inf
        
        for params in param_combos:
            try:
                # Use deterministic seed based on params (same as parallel)
                params_tuple = tuple(sorted(params.items()))
                seed = hash(params_tuple) % (2**31)
                ll = _score(train_df, model, params, seed=seed)
                if ll > best_ll:
                    best_ll = ll
                    best_params = params
            except Exception:
                continue
        
        return best_params or {}, best_ll


# Parameter grids for each model
PARAMETER_GRIDS: Dict[str, Dict[str, Sequence]] = {
    "M1": {
        "gamma": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    },
    "M2": {
        "gamma_base": [2.0, 3.0, 4.0, 5.0, 6.0],
        "entropy_k": [0.5, 1.0, 1.5, 2.0],
    },
    "M3": {
        "gamma_stable": [4.0, 6.0, 8.0, 10.0],
        "gamma_volatile": [1.0, 2.0, 3.0, 4.0],
        "soft_assign": [False, True],
        "preference_width": [15.0, 20.0, 25.0],  # Width of exploit preference peak
    },
}


def loso_cv(trials: pd.DataFrame, max_workers: Optional[int] = None) -> pd.DataFrame:
    """
    Leave-one-subject-out cross-validation.
    
    Parameters
    ----------
    trials : pd.DataFrame
        Trial data
    max_workers : int, optional
        Number of parallel workers for grid search. If None, uses half of CPU count.
        Set via environment variable PARALLEL_WORKERS to override.
    """
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = int(os.getenv("PARALLEL_WORKERS", max(1, cpu_count // 2)))
    
    records = []
    subjects = sorted(trials["subject_id"].unique())
    
    for subject in subjects:
        train_df = trials[trials["subject_id"] != subject].copy()
        test_df = trials[trials["subject_id"] == subject].copy()
        
        for model in ["M1", "M2", "M3"]:
            parameter_grid = PARAMETER_GRIDS[model]
            best_params, train_ll = _grid_search(train_df, model, parameter_grid, max_workers=max_workers)
            test_ll = _score(test_df, model, best_params)
            
            records.append(
                FitResult(
                    model=model,
                    subject_id=subject,
                    params=best_params,
                    train_ll=train_ll,
                    test_ll=test_ll,
                    num_trials=len(test_df),
                ).__dict__
            )
    
    return pd.DataFrame(records)


def temporal_split_cv(trials: pd.DataFrame, max_workers: Optional[int] = None) -> pd.DataFrame:
    """
    Within-subject temporal split: first two runs vs last two runs.
    
    Parameters
    ----------
    trials : pd.DataFrame
        Trial data
    max_workers : int, optional
        Number of parallel workers for grid search. If None, uses half of CPU count.
        Set via environment variable PARALLEL_WORKERS to override.
    """
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = int(os.getenv("PARALLEL_WORKERS", max(1, cpu_count // 2)))
    
    records = []
    
    for subject, subject_df in trials.groupby("subject_id"):
        run_ids = sorted(subject_df["run_id"].unique())
        midpoint = len(run_ids) // 2
        train_runs = run_ids[:midpoint]
        test_runs = run_ids[midpoint:]
        
        train_df = subject_df[subject_df["run_id"].isin(train_runs)].copy()
        test_df = subject_df[subject_df["run_id"].isin(test_runs)].copy()
        
        for model in ["M1", "M2", "M3"]:
            parameter_grid = PARAMETER_GRIDS[model]
            best_params, train_ll = _grid_search(train_df, model, parameter_grid, max_workers=max_workers)
            test_ll = _score(test_df, model, best_params)
            
            records.append(
                FitResult(
                    model=model,
                    subject_id=subject,
                    params=best_params,
                    train_ll=train_ll,
                    test_ll=test_ll,
                    num_trials=len(test_df),
                ).__dict__
            )
    
    return pd.DataFrame(records)

