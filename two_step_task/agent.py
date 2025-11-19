"""Shim module: re-export TwoStepAgent from `models.agent`.

This file keeps a small compatibility layer so existing imports of
`two_step_task.agent` continue to work while implementation lives in
`two_step_task.models.agent`.
"""
from two_step_task.models.agent import TwoStepAgent

__all__ = ["TwoStepAgent"]
