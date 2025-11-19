"""M3: Reusable value profiles (belief-weighted mixing)"""
import numpy as np


class M3_ProfileBased:
    """Belief-weighted mixing of two value profiles."""

    def __init__(self):
        # profile precisions
        self.gamma_0 = None
        self.gamma_1 = None

        # profile reward preferences
        self.C_reward_0 = None
        self.C_reward_1 = None

        # Z: profile assignment matrix (n_contexts x n_profiles)
        # If None, default to balanced assignments
        self.Z = None

    def free_parameters(self):
        return {
            "gamma_0": (0.1, 16.0),
            "gamma_1": (0.1, 16.0),
            "C_reward_0": (-5.0, 5.0),
            "C_reward_1": (-5.0, 5.0),
        }

    def ensure_Z(self, n_contexts: int = 4):
        # Reinitialize Z if missing or wrong shape. Ensure rows sum to 1.
        if self.Z is None or getattr(self.Z, 'shape', (0, 0))[0] != n_contexts:
            self.Z = np.ones((n_contexts, 2), dtype=float) * 0.5
        # normalize rows to sum to 1
        row_sums = self.Z.sum(axis=1, keepdims=True)
        # avoid divide-by-zero
        row_sums[row_sums == 0] = 1.0
        self.Z = self.Z / row_sums

    def compute_gamma(self, Q_s: np.ndarray) -> float:
        """Compute belief-weighted effective precision.

        Q_s: 1D array of length n_contexts
        """
        if self.gamma_0 is None or self.gamma_1 is None:
            raise ValueError("Profile gammas not set. Fit the model first.")
        self.ensure_Z(len(Q_s))
        gammas = np.array([self.gamma_0, self.gamma_1])
        weights = Q_s @ self.Z  # shape (2,)
        return float(weights @ gammas)
