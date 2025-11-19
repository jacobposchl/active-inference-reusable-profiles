"""Parameter search utilities: coarse grid search + local refinement.

Provides:
- generate_grid: create parameter vectors over bounds (with safeguards)
- optimize_from_init: run L-BFGS-B minimization starting from init
- parameter_search: run grid then refine top candidates
"""
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize

from two_step_task.fitting.fit_models import compute_log_likelihood


def generate_grid(param_bounds: Dict[str, Tuple[float, float]], n_points: int = 10, max_cells: int = 20000) -> List[np.ndarray]:
    """Generate a grid of parameter vectors within bounds.

    If the full Cartesian grid would exceed `max_cells`, randomly sample
    `max_cells` points from the grid instead of returning the full grid.
    """
    keys = list(param_bounds.keys())
    grids = [np.linspace(low, high, n_points) for (low, high) in param_bounds.values()]

    # full grid size
    sizes = [len(g) for g in grids]
    total = int(np.prod(sizes))

    if total <= max_cells:
        mesh = np.meshgrid(*grids, indexing='ij')
        mesh_flat = [m.ravel() for m in mesh]
        params = np.vstack(mesh_flat).T
        return [p for p in params]
    else:
        # sample randomly
        params = []
        for _ in range(max_cells):
            p = np.array([np.random.choice(g) for g in grids])
            params.append(p)
        return params


def optimize_from_init(model, init_params: np.ndarray, participant_df) -> Tuple[np.ndarray, float]:
    """Run local optimization starting from `init_params`. Returns (params, ll)."""
    param_bounds = model.free_parameters()
    bounds = list(param_bounds.values())

    res = minimize(
        fun=lambda x: -compute_log_likelihood(model, x, participant_df),
        x0=init_params,
        method='L-BFGS-B',
        bounds=bounds,
    )
    return res.x, -res.fun


def parameter_search(model, participant_df, n_points: int = 8, top_k: int = 5) -> Tuple[np.ndarray, float]:
    """Two-stage parameter search: coarse grid then refine.

    Returns best_params, best_ll
    """
    param_bounds = model.free_parameters()
    grid = generate_grid(param_bounds, n_points=n_points)

    results = []
    for p in grid:
        try:
            ll = compute_log_likelihood(model, p, participant_df)
        except Exception:
            ll = -np.inf
        results.append((p, ll))

    # get top_k
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    top = results_sorted[:top_k]

    best_ll = -np.inf
    best_params = None
    for init_p, _ in top:
        try:
            params_opt, ll_opt = optimize_from_init(model, init_p, participant_df)
        except Exception:
            continue
        if ll_opt > best_ll:
            best_ll = ll_opt
            best_params = params_opt

    return best_params, best_ll
