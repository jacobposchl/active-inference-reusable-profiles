"""M1: Static global precision model"""
import numpy as np


class M1_StaticPrecision:
    """Fixed policy precision throughout task

    Attributes are left as None until fitted. `free_parameters` returns
    bounds for optimization routines.
    """

    def __init__(self):
        self.gamma = None
        self.C_reward = None

    def free_parameters(self):
        """Return dict of parameter name -> (low, high) bounds."""
        return {
            "gamma": (0.1, 16.0),
            "C_reward": (-5.0, 5.0),
        }

    def compute_gamma(self, *args, **kwargs):
        if self.gamma is None:
            raise ValueError("Parameter gamma not set. Fit the model first.")
        return float(self.gamma)
