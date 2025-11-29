"""
Cross-validation utilities for the new active inference models.

Evaluates AIF models on prediction log-likelihood (matching RL baselines).
"""
from __future__ import annotations

import itertools
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

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


def _compute_complexity_penalty(model: str, params: Dict[str, float], n_trials: int) -> float:
    """
    Compute AIC penalty for model complexity.
    
    AIC = -2*log_likelihood + 2*k
    where k = number of free parameters
    
    We return the penalty term (2*k) to subtract from LL.
    """
    # Count free parameters
    if model == "M1":
        k = 1  # gamma
    elif model == "M2":
        k = 2  # gamma_base, entropy_k
    elif model == "M3":
        k = 4  # gamma_stable, gamma_volatile, soft_assign (binary), preference_width
    else:
        k = len(params)
    
    # AIC penalty: 2 * k
    # Using AIC (not BIC) because we have large sample sizes
    penalty = 2.0 * k
    return penalty


def _score_worker_wrapper(args):
    """Worker function wrapper for parallel scoring.
    
    Note: This function must be at module level for multiprocessing.
    DataFrames are pickleable, so we can pass them directly.
    
    Uses a deterministic seed based on params hash to ensure reproducibility.
    Uses validation set to prevent overfitting.
    Applies AIC complexity penalty to favor simpler models.
    """
    import sys
    import traceback
    
    # Handle both old (3 args) and new (4 args) formats for backward compatibility
    if len(args) == 4:
        train_df, val_df, model, params = args
    elif len(args) == 3:
        # Old format: (train_df, model, params) - use train_df as both train and val
        train_df, model, params = args
        val_df = train_df
    else:
        raise ValueError(f"Expected 3 or 4 arguments, got {len(args)}: {args}")
    try:
        # Use deterministic seed based on params to ensure reproducibility
        # Hash params dict to get a seed (convert to sorted tuple for determinism)
        params_tuple = tuple(sorted(params.items()))
        seed = hash(params_tuple) % (2**31)  # Convert to positive int
        
        # Fit on training data, evaluate on validation data
        # This prevents overfitting by selecting params based on validation performance
        agent = _create_agent(model, params, float(train_df["noise_sd"].iloc[0]), seed=seed)
        
        # Fit on training data (run through training trials to update beliefs)
        evaluate_agent_on_dataframe(
            agent,
            train_df,
            outcome_col="outcome",
            prediction_col="prediction",
            group_by="run_id",
        )
        
        # Evaluate on validation data
        val_ll, _ = evaluate_agent_on_dataframe(
            agent,
            val_df,
            outcome_col="outcome",
            prediction_col="prediction",
            group_by="run_id",
        )
        
        # Apply AIC complexity penalty (subtract penalty from LL)
        # This favors simpler models when LL is similar
        n_val_trials = len(val_df)
        penalty = _compute_complexity_penalty(model, params, n_val_trials)
        penalized_ll = val_ll - penalty
        
        return params, penalized_ll
    except Exception as e:
        # Log errors to stderr
        error_msg = f"Error in worker for {model} with params {params}:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr, flush=True)
        return None, -math.inf


def _grid_search(train_df: pd.DataFrame, model: str, parameter_grid: Dict[str, Sequence], max_workers: Optional[int] = None, use_validation: bool = True) -> Tuple[Dict[str, float], float]:
    """
    Grid search for best parameters with optional parallel processing.
    
    For M3, uses smart constrained search with hierarchical refinement.
    
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
    # Set default workers if not provided
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = max(1, cpu_count // 2)
    
    # Split training data into train/validation to prevent overfitting
    # Use 80% for training, 20% for validation
    if use_validation and len(train_df) > 100:  # Only split if we have enough data
        # Split by subjects to maintain independence
        subjects = sorted(train_df["subject_id"].unique())
        n_val_subjects = max(1, int(len(subjects) * 0.2))
        val_subjects = set(subjects[-n_val_subjects:])  # Use last subjects for validation
        val_df = train_df[train_df["subject_id"].isin(val_subjects)].copy()
        train_fit_df = train_df[~train_df["subject_id"].isin(val_subjects)].copy()
    else:
        # Not enough data or validation disabled - use all for training
        val_df = None
        train_fit_df = train_df
    
    # Smart constrained search for M3
    if model == "M3":
        return _grid_search_m3_smart(train_fit_df, val_df, parameter_grid, max_workers)
    
    # Standard grid search for M1 and M2
    param_combos = _expand_grid(parameter_grid)
    
    # Use parallel processing if max_workers > 1 and multiple parameter combinations
    # On Windows, multiprocessing can be problematic, so we'll test with a small timeout first
    if max_workers > 1 and len(param_combos) > 1:
        best_params = None
        best_ll = -math.inf
        
        # Prepare arguments for workers (DataFrames are pickleable)
        if val_df is not None:
            args_list = [(train_fit_df, val_df, model, params) for params in param_combos]
        else:
            # Fallback: use training LL if no validation set
            args_list = [(train_df, train_df, model, params) for params in param_combos]
        
        completed = 0
        start_time = time.time()
        
        # Set multiprocessing start method for Windows compatibility
        import multiprocessing
        if hasattr(multiprocessing, 'set_start_method'):
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_params = {executor.submit(_score_worker_wrapper, args): args[2] for args in args_list}
                
                for future in as_completed(future_to_params):
                    params, ll = future.result()
                    completed += 1
                    
                    # Show progress every 25% or when finding new best
                    if (completed % max(1, len(param_combos) // 4) == 0 or 
                        completed == len(param_combos) or
                        (params is not None and ll > best_ll)):
                        elapsed = time.time() - start_time
                        tqdm.write(f"        [{completed}/{len(param_combos)}] LL={ll:.2f}, best={best_ll:.2f} ({elapsed:.1f}s)")
                    
                    if params is not None and ll > best_ll:
                        best_ll = ll
                        best_params = params
            
            if best_params is None:
                raise RuntimeError("All parameter combinations failed - no valid results")
            
            return best_params or {}, best_ll
        except Exception as e:
            # Parallel processing failed - raise error instead of falling back
            import traceback
            error_msg = f"Parallel processing failed: {e}\n{traceback.format_exc()}"
            tqdm.write(f"        ERROR: {error_msg}")
            raise RuntimeError(f"Grid search failed: {e}") from e
    
    # If we get here, max_workers <= 1 or only one param combo - use sequential
    if len(param_combos) == 1:
        params = param_combos[0]
        params_tuple = tuple(sorted(params.items()))
        seed = hash(params_tuple) % (2**31)
        if val_df is not None:
            # Fit on training, evaluate on validation
            agent = _create_agent(model, params, float(train_fit_df["noise_sd"].iloc[0]), seed=seed)
            evaluate_agent_on_dataframe(agent, train_fit_df, outcome_col="outcome", prediction_col="prediction", group_by="run_id")
            val_ll, _ = evaluate_agent_on_dataframe(agent, val_df, outcome_col="outcome", prediction_col="prediction", group_by="run_id")
            penalty = _compute_complexity_penalty(model, params, len(val_df))
            ll = val_ll - penalty
        else:
            ll = _score(train_df, model, params, seed=seed)
        return params, ll
    
    raise RuntimeError(f"Parallel processing required but max_workers={max_workers}")


def _grid_search_m3_smart(train_df: pd.DataFrame, val_df: Optional[pd.DataFrame], parameter_grid: Dict[str, Sequence], max_workers: int) -> Tuple[Dict[str, float], float]:
    """
    Smart constrained search for M3 that exploits structure:
    - Independent gamma per profile (already doing this)
    - Hierarchical search: coarse grid → fine grid around best
    - Reduces from 96 to ~54-81 evaluations
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    parameter_grid : dict
        Full parameter grid for M3
    max_workers : int
        Number of parallel workers
    """
    # Extract grid values
    gamma_stable_vals = list(parameter_grid["gamma_stable"])
    gamma_volatile_vals = list(parameter_grid["gamma_volatile"])
    soft_assign_vals = list(parameter_grid["soft_assign"])
    preference_width_vals = list(parameter_grid["preference_width"])
    
    # COARSE GRID: Use fewer values for initial search (reduced from 3 to 2 per gamma)
    # This reduces from 54 to 16 combinations (3x faster!)
    gamma_stable_coarse = [gamma_stable_vals[0], gamma_stable_vals[-1]]  # Min and max only
    gamma_volatile_coarse = [gamma_volatile_vals[0], gamma_volatile_vals[-1]]  # Min and max only
    # Test both soft_assign options in coarse (they're conceptually different)
    soft_assign_coarse = soft_assign_vals
    # Test only min and max preference_width (reduces from 3 to 2)
    preference_width_coarse = [preference_width_vals[0], preference_width_vals[-1]]
    
    # Build coarse candidates
    coarse_candidates = []
    for gs in gamma_stable_coarse:
        for gv in gamma_volatile_coarse:
            for sa in soft_assign_coarse:
                for pw in preference_width_coarse:
                    coarse_candidates.append({
                        "gamma_stable": gs,
                        "gamma_volatile": gv,
                        "soft_assign": sa,
                        "preference_width": pw,
                    })
    
    # Evaluate coarse grid
    coarse_start = time.time()
    best_coarse_params = None
    best_coarse_ll = -math.inf
    
    if max_workers > 1 and len(coarse_candidates) > 1:
        if val_df is not None:
            args_list = [(train_df, val_df, "M3", params) for params in coarse_candidates]
        else:
            args_list = [(train_df, train_df, "M3", params) for params in coarse_candidates]
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {executor.submit(_score_worker_wrapper, args): args[2] for args in args_list}
            for future in as_completed(future_to_params):
                params, ll = future.result()
                completed += 1
                
                # Show progress every 25% or when finding new best
                if (completed % max(1, len(coarse_candidates) // 4) == 0 or 
                    completed == len(coarse_candidates) or
                    (params is not None and ll > best_coarse_ll)):
                    elapsed = time.time() - coarse_start
                    tqdm.write(f"        Coarse [{completed}/{len(coarse_candidates)}] LL={ll:.2f}, best={best_coarse_ll:.2f} ({elapsed:.1f}s)")
                
                if params is not None and ll > best_coarse_ll:
                    best_coarse_ll = ll
                    best_coarse_params = params
    else:
        for params in coarse_candidates:
            try:
                params_tuple = tuple(sorted(params.items()))
                seed = hash(params_tuple) % (2**31)
                if val_df is not None:
                    agent = _create_agent("M3", params, float(train_df["noise_sd"].iloc[0]), seed=seed)
                    evaluate_agent_on_dataframe(agent, train_df, outcome_col="outcome", prediction_col="prediction", group_by="run_id")
                    val_ll, _ = evaluate_agent_on_dataframe(agent, val_df, outcome_col="outcome", prediction_col="prediction", group_by="run_id")
                    penalty = _compute_complexity_penalty("M3", params, len(val_df))
                    ll = val_ll - penalty
                else:
                    ll = _score(train_df, "M3", params, seed=seed)
                if ll > best_coarse_ll:
                    best_coarse_ll = ll
                    best_coarse_params = params
            except Exception:
                continue
    
    if best_coarse_params is None:
        return {}, -math.inf
    
    # FINE GRID: Refine around best coarse result
    fine_start = time.time()
    # Find indices of best coarse values
    gs_idx = gamma_stable_vals.index(best_coarse_params["gamma_stable"])
    gv_idx = gamma_volatile_vals.index(best_coarse_params["gamma_volatile"])
    pw_idx = preference_width_vals.index(best_coarse_params["preference_width"])
    
    # Create fine grid around best coarse values
    gs_lo_idx = max(0, gs_idx - 1)
    gs_hi_idx = min(len(gamma_stable_vals) - 1, gs_idx + 1)
    gv_lo_idx = max(0, gv_idx - 1)
    gv_hi_idx = min(len(gamma_volatile_vals) - 1, gv_idx + 1)
    pw_lo_idx = max(0, pw_idx - 1)
    pw_hi_idx = min(len(preference_width_vals) - 1, pw_idx + 1)
    
    gamma_stable_fine = gamma_stable_vals[gs_lo_idx:gs_hi_idx+1]
    gamma_volatile_fine = gamma_volatile_vals[gv_lo_idx:gv_hi_idx+1]
    # Keep best soft_assign from coarse (they're conceptually different, not continuous)
    soft_assign_fine = [best_coarse_params["soft_assign"]]
    preference_width_fine = preference_width_vals[pw_lo_idx:pw_hi_idx+1]
    
    # Build fine candidates
    fine_candidates = []
    for gs in gamma_stable_fine:
        for gv in gamma_volatile_fine:
            for sa in soft_assign_fine:
                for pw in preference_width_fine:
                    fine_candidates.append({
                        "gamma_stable": gs,
                        "gamma_volatile": gv,
                        "soft_assign": sa,
                        "preference_width": pw,
                    })
    
    # Evaluate fine grid
    best_fine_params = best_coarse_params
    best_fine_ll = best_coarse_ll
    
    if max_workers > 1 and len(fine_candidates) > 1:
        if val_df is not None:
            args_list = [(train_df, val_df, "M3", params) for params in fine_candidates]
        else:
            args_list = [(train_df, train_df, "M3", params) for params in fine_candidates]
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {executor.submit(_score_worker_wrapper, args): args[2] for args in args_list}
            for future in as_completed(future_to_params):
                params, ll = future.result()
                completed += 1
                
                # Show progress every 25% or when finding new best
                if (completed % max(1, len(fine_candidates) // 4) == 0 or 
                    completed == len(fine_candidates) or
                    (params is not None and ll > best_fine_ll)):
                    elapsed = time.time() - fine_start
                    tqdm.write(f"        Fine [{completed}/{len(fine_candidates)}] LL={ll:.2f}, best={best_fine_ll:.2f} ({elapsed:.1f}s)")
                
                if params is not None and ll > best_fine_ll:
                    best_fine_ll = ll
                    best_fine_params = params
    else:
        for params in fine_candidates:
            try:
                params_tuple = tuple(sorted(params.items()))
                seed = hash(params_tuple) % (2**31)
                if val_df is not None:
                    agent = _create_agent("M3", params, float(train_df["noise_sd"].iloc[0]), seed=seed)
                    evaluate_agent_on_dataframe(agent, train_df, outcome_col="outcome", prediction_col="prediction", group_by="run_id")
                    val_ll, _ = evaluate_agent_on_dataframe(agent, val_df, outcome_col="outcome", prediction_col="prediction", group_by="run_id")
                    penalty = _compute_complexity_penalty("M3", params, len(val_df))
                    ll = val_ll - penalty
                else:
                    ll = _score(train_df, "M3", params, seed=seed)
                if ll > best_fine_ll:
                    best_fine_ll = ll
                    best_fine_params = params
            except Exception:
                continue
    
    return best_fine_params or {}, best_fine_ll


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


def loso_cv(trials: pd.DataFrame, max_workers: Optional[int] = None, save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Leave-one-subject-out cross-validation.
    
    Parameters
    ----------
    trials : pd.DataFrame
        Trial data
    max_workers : int, optional
        Number of parallel workers for grid search. If None, uses half of CPU count.
        Set via environment variable PARALLEL_WORKERS to override.
    save_path : Path, optional
        Path to save results incrementally. If provided, saves after each subject completes.
    """
    import time
    
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = int(os.getenv("PARALLEL_WORKERS", max(1, cpu_count // 2)))
    
    records = []
    subjects = sorted(trials["subject_id"].unique())
    n_subjects = len(subjects)
    
    tqdm.write(f"\nStarting LOSO CV: {n_subjects} subjects, {max_workers} workers")
    tqdm.write(f"Training on {n_subjects - 1} subjects, testing on 1 subject per fold\n")
    
    # Outer loop: subjects
    subject_pbar = tqdm(subjects, desc="LOSO CV", unit="subject", leave=True)
    for subject_idx, subject in enumerate(subject_pbar, 1):
        subject_start_time = time.time()
        train_df = trials[trials["subject_id"] != subject].copy()
        test_df = trials[trials["subject_id"] == subject].copy()
        
        n_train_trials = len(train_df)
        n_test_trials = len(test_df)
        
        subject_pbar.set_postfix({"subject": subject})
        
        tqdm.write(f"\n  Subject {subject_idx}/{n_subjects}: {subject} (train: {n_train_trials} trials, test: {n_test_trials} trials)")
        
        # Inner loop: models
        for model_idx, model in enumerate(["M1", "M2", "M3"], 1):
            model_start_time = time.time()
            tqdm.write(f"    [{model_idx}/3] Fitting {model}...")
            
            parameter_grid = PARAMETER_GRIDS[model]
            n_grid_points = len(_expand_grid(parameter_grid)) if model != "M3" else "~16-27 (smart search)"
            
            best_params, train_ll = _grid_search(train_df, model, parameter_grid, max_workers=max_workers)
            grid_time = time.time() - model_start_time
            
            if best_params:
                tqdm.write(f"      ✓ Completed in {grid_time:.1f}s ({n_grid_points} grid points)")
                tqdm.write(f"      Best params: {best_params}")
                tqdm.write(f"      Train LL: {train_ll:.2f}")
            else:
                tqdm.write(f"      ✗ Failed ({grid_time:.1f}s)")
                train_ll = -math.inf
            
            # Evaluate on test set
            test_start_time = time.time()
            test_ll = _score(test_df, model, best_params)
            test_time = time.time() - test_start_time
            
            tqdm.write(f"      Test LL: {test_ll:.2f} ({test_time:.1f}s)")
            
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
        
        subject_time = time.time() - subject_start_time
        tqdm.write(f"  ✓ Completed {subject} in {subject_time:.1f}s\n")
        
        # Save incrementally after each subject completes
        if save_path is not None:
            df = pd.DataFrame(records)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            tqdm.write(f"  → Saved incremental results to {save_path} ({len(records)} records)")
    
    tqdm.write(f"\n✓ LOSO CV complete: {len(records)} model-subject combinations evaluated")
    return pd.DataFrame(records)


def temporal_split_cv(trials: pd.DataFrame, max_workers: Optional[int] = None, save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Within-subject temporal split: first two runs vs last two runs.
    
    Parameters
    ----------
    trials : pd.DataFrame
        Trial data
    max_workers : int, optional
        Number of parallel workers for grid search. If None, uses half of CPU count.
        Set via environment variable PARALLEL_WORKERS to override.
    save_path : Path, optional
        Path to save results incrementally. If provided, saves after each subject completes.
    """
    import time
    
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = int(os.getenv("PARALLEL_WORKERS", max(1, cpu_count // 2)))
    
    records = []
    subjects = sorted(trials["subject_id"].unique())
    n_subjects = len(subjects)
    
    tqdm.write(f"\nStarting Temporal Split CV: {n_subjects} subjects, {max_workers} workers")
    tqdm.write(f"Training on first half of runs, testing on second half per subject\n")
    
    # Outer loop: subjects
    subject_pbar = tqdm(subjects, desc="Temporal Split CV", unit="subject", leave=True)
    for subject_idx, subject in enumerate(subject_pbar, 1):
        subject_start_time = time.time()
        subject_df = trials[trials["subject_id"] == subject].copy()
        run_ids = sorted(subject_df["run_id"].unique())
        midpoint = len(run_ids) // 2
        train_runs = run_ids[:midpoint]
        test_runs = run_ids[midpoint:]
        
        train_df = subject_df[subject_df["run_id"].isin(train_runs)].copy()
        test_df = subject_df[subject_df["run_id"].isin(test_runs)].copy()
        
        n_train_trials = len(train_df)
        n_test_trials = len(test_df)
        
        subject_pbar.set_postfix({"subject": subject})
        
        tqdm.write(f"\n  Subject {subject_idx}/{n_subjects}: {subject} (train: {n_train_trials} trials, test: {n_test_trials} trials)")
        
        # Inner loop: models
        for model_idx, model in enumerate(["M1", "M2", "M3"], 1):
            model_start_time = time.time()
            tqdm.write(f"    [{model_idx}/3] Fitting {model}...")
            
            parameter_grid = PARAMETER_GRIDS[model]
            n_grid_points = len(_expand_grid(parameter_grid)) if model != "M3" else "~16-27 (smart search)"
            
            best_params, train_ll = _grid_search(train_df, model, parameter_grid, max_workers=max_workers)
            grid_time = time.time() - model_start_time
            
            if best_params:
                tqdm.write(f"      ✓ Completed in {grid_time:.1f}s ({n_grid_points} grid points)")
                tqdm.write(f"      Best params: {best_params}")
                tqdm.write(f"      Train LL: {train_ll:.2f}")
            else:
                tqdm.write(f"      ✗ Failed ({grid_time:.1f}s)")
                train_ll = -math.inf
            
            # Evaluate on test set
            test_start_time = time.time()
            test_ll = _score(test_df, model, best_params)
            test_time = time.time() - test_start_time
            
            tqdm.write(f"      Test LL: {test_ll:.2f} ({test_time:.1f}s)")
            
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
        
        subject_time = time.time() - subject_start_time
        tqdm.write(f"  ✓ Completed {subject} in {subject_time:.1f}s\n")
        
        # Save incrementally after each subject completes
        if save_path is not None:
            df = pd.DataFrame(records)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            tqdm.write(f"  → Saved incremental results to {save_path} ({len(records)} records)")
    
    tqdm.write(f"\n✓ Temporal Split CV complete: {len(records)} model-subject combinations evaluated")
    return pd.DataFrame(records)

