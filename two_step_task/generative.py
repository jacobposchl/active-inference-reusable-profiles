"""Shim module: re-export A/B builders from `models.generative`.

The implementations live in `two_step_task.models.generative`. This
top-level module provides backward-compatible imports.
"""
from two_step_task.models.generative import build_A_matrices, build_B_matrices

__all__ = ["build_A_matrices", "build_B_matrices"]
