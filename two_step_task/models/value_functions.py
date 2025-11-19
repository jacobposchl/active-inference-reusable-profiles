"""Value function factory for two-step task models.

This module produces a `value_fn(q_context, t)` that returns a tuple
`(C_eff, E_t, gamma_t)` where `C_eff` is a scalar reward preference
that will be mapped to the reward-modality `C` vector by the runner.
E_t is optional policy prior logits (or None) and gamma_t is policy
precision for the trial.
"""
import numpy as np


def make_values_M1(model):
    def value_fn(q_context, t):
        if getattr(model, "C_reward", None) is None:
            raise ValueError("M1 model parameter C_reward not set")
        if getattr(model, "gamma", None) is None:
            raise ValueError("M1 model parameter gamma not set")
        return float(model.C_reward), None, float(model.gamma)
    return value_fn


def make_values_M2(model):
    def value_fn(q_context, t):
        if getattr(model, "C_reward", None) is None:
            raise ValueError("M2 model parameter C_reward not set")
        if not hasattr(model, "compute_gamma"):
            raise ValueError("M2 model missing compute_gamma(Q_s)")
        gamma_t = float(model.compute_gamma(q_context))
        return float(model.C_reward), None, gamma_t
    return value_fn


def make_values_M3(model):
    def value_fn(q_context, t):
        # ensure Z exists
        if not hasattr(model, "ensure_Z"):
            raise ValueError("M3 model missing ensure_Z")
        model.ensure_Z(len(q_context))
        weights = q_context @ model.Z  # shape (n_profiles,)
        C0 = float(getattr(model, "C_reward_0", 1.0))
        C1 = float(getattr(model, "C_reward_1", 1.0))
        gamma0 = float(getattr(model, "gamma_0", 1.0))
        gamma1 = float(getattr(model, "gamma_1", 1.0))
        C_eff = float(weights[0] * C0 + weights[1] * C1)
        gamma_t = float((weights[0] * gamma0 + weights[1] * gamma1))
        # Note: E_t (policy priors) not implemented in this factory
        return C_eff, None, gamma_t
    return value_fn


def make_value_fn_for_model(model):
    cls = model.__class__.__name__
    if cls == "M1_StaticPrecision":
        return make_values_M1(model)
    elif cls == "M2_EntropyCoupled":
        return make_values_M2(model)
    elif cls == "M3_ProfileBased":
        return make_values_M3(model)
    else:
        # Fallback: attempt to use compute_gamma and C_reward if present
        def value_fn(q_context, t):
            C_eff = float(getattr(model, "C_reward", 1.0))
            if hasattr(model, "compute_gamma"):
                gamma_t = float(model.compute_gamma(q_context))
            else:
                gamma_t = float(getattr(model, "gamma", 1.0))
            return C_eff, None, gamma_t
        return value_fn
