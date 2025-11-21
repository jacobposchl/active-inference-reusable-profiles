"""Utilities module."""

from .helpers import (
    rolling_mean,
    compute_entropy,
    find_reversals,
    trial_accuracy,
    bootstrap_ci,
    print_trial_details
)

__all__ = [
    'rolling_mean',
    'compute_entropy',
    'find_reversals',
    'trial_accuracy',
    'bootstrap_ci',
    'print_trial_details'
]
