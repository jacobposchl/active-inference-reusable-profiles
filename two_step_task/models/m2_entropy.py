"""M2: Entropy-coupled dynamic precision model"""
import numpy as np


class M2_EntropyCoupled:
    """Policy precision inversely proportional to belief entropy.

    compute_gamma expects a belief vector Q_s (1D numpy array).
    """

    def __init__(self):
        self.gamma_base = None
        self.beta = None
        self.C_reward = None

    def free_parameters(self):
        return {
            "gamma_base": (0.1, 16.0),
            "beta": (0.0, 5.0),
            "C_reward": (-5.0, 5.0),
        }

    def compute_gamma(self, Q_s: np.ndarray):
        if self.gamma_base is None or self.beta is None:
            raise ValueError("Model parameters not set. Fit the model first.")
        eps = 1e-16
        H = -np.sum(Q_s * np.log(Q_s + eps))
        return float(self.gamma_base / (1.0 + self.beta * H))
