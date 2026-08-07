#!/usr/bin/env python3
"""Generate a synthetic model-recovery run that satisfies the on-disk contract
consumed by figures.py (the AIC + M3 mechanistic figures).

Uses only numpy/pandas/os/json/argparse. No pymdp, no src imports, no datetime.
Deterministic: per-file fixed seeds so bytes are independent of generation order.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


T = 400            # trials per run
N_RUNS = 5         # runs per model
BLOCK = 40         # trials per context block

ACTIONS = ["act_start", "act_hint", "act_left", "act_right"]


# --------------------------------------------------------------------------
# Confusion matrices (fully hard-coded, no RNG).
# --------------------------------------------------------------------------
def write_confusion(run_dir):
    conf_dir = os.path.join(run_dir, "confusion")
    os.makedirs(conf_dir, exist_ok=True)

    index = ["M1", "M2", "M3", "egreedy", "softmax"]
    cols = ["M1", "M2", "M3"]

    mean_data = [
        [8.26, 16.42, 55.31],     # M1
        [11.64, 19.05, 59.06],    # M2
        [180.14, 162.09, 72.30],  # M3
        [227.24, 225.60, 239.70], # egreedy
        [222.98, 225.18, 243.95], # softmax
    ]
    se_data = [
        [0.19, 0.33, 0.97],   # M1
        [0.89, 1.09, 2.07],   # M2
        [5.49, 6.24, 1.67],   # M3
        [0.88, 4.47, 3.63],   # egreedy
        [0.52, 0.60, 1.06],   # softmax
    ]

    df_mean = pd.DataFrame(mean_data, index=index, columns=cols)
    df_se = pd.DataFrame(se_data, index=index, columns=cols)

    df_mean.to_csv(os.path.join(conf_dir, "aic_mean.csv"),
                   index=True, index_label="generator")
    df_se.to_csv(os.path.join(conf_dir, "aic_se.csv"),
                 index=True, index_label="generator")


# --------------------------------------------------------------------------
# Shared global trial structure (identical across all three models).
# --------------------------------------------------------------------------
# 8 balanced context reversals per run (4 per direction) -> pooled over 5 runs
# BOTH reversal directions show n=20 in Panels A/B. Start in the stable regime.
REVERSAL_TRIALS = [BLOCK * i for i in range(1, 9)]   # [40, 80, ..., 320]
REVERSAL_SET = set(REVERSAL_TRIALS)


def _flips(t):
    return sum(1 for rt in REVERSAL_TRIALS if rt <= t)


def _last_reversal(t):
    passed = [rt for rt in REVERSAL_TRIALS if rt <= t]
    return passed[-1] if passed else 0


def true_context_of(t):
    # start stable; each reversal flips the regime
    return "stable" if (_flips(t) % 2 == 0) else "volatile"


def better_arm_of(t):
    if true_context_of(t) == "volatile":
        return "left" if ((t // 10) % 2 == 0) else "right"   # micro-reversals every 10
    return "left" if (_flips(t) % 4 < 2) else "right"        # held within a stable block


def is_reversal_of(t):
    return 1 if t in REVERSAL_SET else 0


def belief_ramp(t, asymptote, amplitude, tau):
    """Active-context belief ramp since the last reversal. Returns (w_vol, w_stab)."""
    s = t - _last_reversal(t)
    w_active = float(np.clip(asymptote - amplitude * np.exp(-s / tau), 0.0, 1.0))
    if true_context_of(t) == "volatile":
        return w_active, 1.0 - w_active
    return 1.0 - w_active, w_active


def action_probs_for(pred, p_chosen=0.7, other=0.1):
    """Length-4 normalized list whose argmax equals pred."""
    probs = np.full(4, other, dtype=float)
    idx = ACTIONS.index(pred)
    probs[idx] = p_chosen
    probs = probs / probs.sum()
    return [round(float(x), 6) for x in probs.tolist()]


def choice_label_of(action):
    return {
        "act_start": "observe_start",
        "act_hint": "observe_hint",
        "act_left": "observe_left",
        "act_right": "observe_right",
    }[action]


def make_trial_df(model, run_n):
    """Build one T-row trial DataFrame for the given model and run index."""
    if model == "M1":
        np.random.seed(1000 + run_n)
    elif model == "M2":
        np.random.seed(2000 + run_n)
    else:
        np.random.seed(3000 + run_n)

    rows = []
    # track when the current better arm last switched (for M2 gamma)
    prev_arm = None
    last_switch_t = 0

    for t in range(T):
        ctx = true_context_of(t)
        arm = better_arm_of(t)
        rev = is_reversal_of(t)

        if prev_arm is None or arm != prev_arm:
            last_switch_t = t
        prev_arm = arm

        # ---- gen_action (unread by figure) ----
        if np.random.rand() < 0.05:
            gen_action = "act_hint"
        else:
            if arm == "left":
                gen_action = "act_left" if np.random.rand() < 0.8 else "act_right"
            else:
                gen_action = "act_right" if np.random.rand() < 0.8 else "act_left"

        # ---- hint_label ----
        if np.random.rand() < 0.5:
            hint_label = "observe_left_hint" if arm == "left" else "observe_right_hint"
        else:
            hint_label = "null"

        # ---- reward_label ----
        if gen_action == "act_start" and np.random.rand() < 0.1:
            reward_label = "null"
        elif np.random.rand() < 0.6:
            reward_label = "observe_reward"
        else:
            reward_label = "observe_loss"

        # ---- choice_label (from gen_action) ----
        choice_label = choice_label_of(gen_action)

        # ---- predicted_action (model signature for M3) ----
        if model == "M3":
            p_hint = 0.45 if ctx == "volatile" else 0.05
        else:
            p_hint = 0.03 if model == "M1" else 0.05
        if np.random.rand() < p_hint:
            predicted_action = "act_hint"
        else:
            if arm == "left":
                predicted_action = "act_left" if np.random.rand() < 0.8 else "act_right"
            else:
                predicted_action = "act_right" if np.random.rand() < 0.8 else "act_left"

        # ---- action_probs ----
        action_probs = json.dumps(action_probs_for(predicted_action))

        # ---- belief_context (ramp; sharper for M3) ----
        if model == "M3":
            w_vol, w_stab = belief_ramp(t, asymptote=0.95, amplitude=0.42, tau=3.0)
        else:
            w_vol, w_stab = belief_ramp(t, asymptote=0.92, amplitude=0.37, tau=4.0)
        belief_context = json.dumps([round(w_vol, 6), round(w_stab, 6)])

        # ---- gamma (model signature) ----
        if model == "M1":
            gamma = 2.5
        elif model == "M2":
            u = t - last_switch_t  # trials since better-arm switch
            g = 1.2 + 3.3 * (1.0 - np.exp(-u / 6.0)) + np.random.normal(0.0, 0.12)
            gamma = float(np.clip(g, 1.0, 5.0))
        else:  # M3 flat ~5.0
            g = 5.0 + np.random.normal(0.0, 0.08)
            if ctx == "stable":
                g += 0.15
            gamma = float(np.clip(g, 1.0, 5.5))

        # ---- better-arm belief + entropy (certainty grows since the arm last switched) ----
        u_arm = t - last_switch_t
        c_arm = 1.0 - np.exp(-u_arm / 4.0)
        p_true = 0.5 + 0.49 * c_arm
        belief_arm = [p_true, 1.0 - p_true] if arm == "left" else [1.0 - p_true, p_true]
        _p = np.clip(np.array(belief_arm), 1e-9, 1.0)
        better_arm_entropy = float(-(_p * np.log(_p)).sum())
        # ---- policy-posterior entropy: rises with arm uncertainty, steeper for M2 ----
        _slope = {"M1": 0.9, "M2": 1.7, "M3": 0.7}[model]
        _base = {"M1": 0.55, "M2": 0.55, "M3": 0.7}[model]
        policy_entropy = float(np.clip(
            _base + _slope * better_arm_entropy + np.random.normal(0.0, 0.05), 0.05, 2.77))

        # ---- ll ----
        ll = -0.3 - 0.5 * np.random.rand()

        # ---- accuracy ----
        chosen_arm = None
        if predicted_action == "act_left":
            chosen_arm = "left"
        elif predicted_action == "act_right":
            chosen_arm = "right"
        accuracy = int(chosen_arm == arm)

        # ---- flags ----
        hint_flag = int(hint_label != "null")
        is_reward = int(reward_label == "observe_reward")
        is_loss = int(reward_label == "observe_loss")

        rows.append({
            "t": t,
            "fold": t % 5,
            "role": "test" if (t % 5 == run_n % 5) else "train",
            "true_context": ctx,
            "current_better_arm": arm,
            "gen_action": gen_action,
            "hint_label": hint_label,
            "reward_label": reward_label,
            "choice_label": choice_label,
            "predicted_action": predicted_action,
            "action_probs": action_probs,
            "belief_context": belief_context,
            "belief_better_arm": json.dumps([round(x, 6) for x in belief_arm]),
            "gamma": round(gamma, 6),
            "policy_entropy": round(policy_entropy, 6),
            "better_arm_entropy": round(better_arm_entropy, 6),
            "ll": round(float(ll), 6),
            "accuracy": accuracy,
            "is_reversal": rev,
            "hint_flag": hint_flag,
            "is_reward": is_reward,
            "is_loss": is_loss,
        })

    cols = ["t", "fold", "role", "true_context", "current_better_arm",
            "gen_action", "hint_label", "reward_label", "choice_label",
            "predicted_action", "action_probs", "belief_context", "belief_better_arm",
            "gamma", "policy_entropy", "better_arm_entropy",
            "ll", "accuracy", "is_reversal", "hint_flag", "is_reward", "is_loss"]
    return pd.DataFrame(rows, columns=cols)


def write_trial_level(run_dir):
    for model in ["M1", "M2", "M3"]:
        mdir = os.path.join(run_dir, "trial_level", "gen_M3", "model_" + model)
        os.makedirs(mdir, exist_ok=True)
        for n in range(N_RUNS):
            df = make_trial_df(model, n)
            fname = "run_%03d.csv" % n
            df.to_csv(os.path.join(mdir, fname), index=False)


# --------------------------------------------------------------------------
# run_summary/gen_M3/model_M3.csv (fully hard-coded best_params_per_fold).
# --------------------------------------------------------------------------
def write_run_summary(run_dir):
    sdir = os.path.join(run_dir, "run_summary", "gen_M3")
    os.makedirs(sdir, exist_ok=True)

    fold_dict = {
        "gamma_profile": [5.0, 5.0],
        "xi_scales_profile": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    }
    best_params_per_fold = json.dumps([fold_dict for _ in range(5)])
    action_distribution = json.dumps(
        {"act_start": 40, "act_hint": 120, "act_left": 120, "act_right": 120})

    columns = [
        "generator", "model", "run_idx", "seed", "runtime_sec", "grid_evals",
        "mean_train_ll", "std_train_ll", "mean_test_ll", "std_test_ll",
        "mean_train_acc", "std_train_acc", "mean_test_acc", "std_test_acc",
        "best_params_per_fold", "reversal_count", "action_distribution",
        "mean_belief_entropy", "best_train_ll", "aic", "bic",
    ]

    rows = []
    for run_idx in range(N_RUNS):
        rows.append({
            "generator": "M3",
            "model": "M3",
            "run_idx": run_idx,
            "seed": 1000 + run_idx,
            "runtime_sec": 123.45,
            "grid_evals": 288,
            "mean_train_ll": -0.33,
            "std_train_ll": 0.04,
            "mean_test_ll": -0.35,
            "std_test_ll": 0.05,
            "mean_train_acc": 0.71,
            "std_train_acc": 0.03,
            "mean_test_acc": 0.70,
            "std_test_acc": 0.04,
            "best_params_per_fold": best_params_per_fold,
            "reversal_count": 9,
            "action_distribution": action_distribution,
            "mean_belief_entropy": 0.30,
            "best_train_ll": -0.32,
            "aic": 72.30,
            "bic": 78.10,
        })

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(os.path.join(sdir, "model_M3.csv"), index=False)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Write a synthetic model-recovery run matching the disk contract.")
    parser.add_argument("--run-id", default="synth",
                        help="Run id; the run dir is <base>/run_<id> (default: run_synth).")
    parser.add_argument("--base", default="results/model_recovery",
                        help="Base directory under which run_<id>/ is written.")
    args = parser.parse_args()

    # Deterministic (spec requests default_rng(0); all values are hard-coded or
    # per-file seeded so this is for form).
    np.random.default_rng(0)

    run_dir = os.path.join(args.base, "run_" + args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    write_confusion(run_dir)
    write_trial_level(run_dir)
    write_run_summary(run_dir)

    print(run_dir)


if __name__ == "__main__":
    main()
