import numpy as np
import os
import concurrent.futures
from src.models import build_A, build_B, build_D, make_value_fn
from src.utils.model_utils import create_model
from config.experiment_config import *


def compute_sequence_ll_for_model(model_name, A, B, D, ref_logs):
    """Compute per-trial log-probabilities of the actions in `ref_logs`
    under `model_name` using teacher-forcing (condition on the same
    observed history). Returns a list of per-trial log-likelihoods.
    """
    value_fn = create_model(model_name, A, B, D)
    from src.models import AgentRunnerWithLL
    runner = AgentRunnerWithLL(A, B, D, value_fn,
                               OBSERVATION_HINTS, OBSERVATION_REWARDS,
                               OBSERVATION_CHOICES, ACTION_CHOICES,
                               reward_mod_idx=1)

    initial_obs_labels = ['null', 'null', 'observe_start']
    obs_ids = runner.obs_labels_to_ids(initial_obs_labels)

    T = len(ref_logs['action'])
    ll_seq = []
    for t in range(T):
        action_label = ref_logs['action'][t]
        ll_t = runner.action_logprob(obs_ids, action_label, t)
        ll_seq.append(ll_t)

        if 'hint_label' in ref_logs and ref_logs['hint_label']:
            next_obs = [ref_logs['hint_label'][t], ref_logs['reward_label'][t], ref_logs['choice_label'][t]]
        else:
            next_obs = ['null', ref_logs['reward_label'][t], ref_logs['choice_label'][t]]

        obs_ids = runner.obs_labels_to_ids(next_obs)

    return ll_seq



def evaluate_ll_with_valuefn(value_fn, A, B, D, ref_logs):
    """Compute total log-likelihood of ref_logs under a model defined by value_fn."""
    from src.models import AgentRunnerWithLL
    runner = AgentRunnerWithLL(A, B, D, value_fn,
                               OBSERVATION_HINTS, OBSERVATION_REWARDS,
                               OBSERVATION_CHOICES, ACTION_CHOICES,
                               reward_mod_idx=1)

    initial_obs_labels = ['null', 'null', 'observe_start']
    obs_ids = runner.obs_labels_to_ids(initial_obs_labels)
    T = len(ref_logs['action'])
    ll_seq = []
    for t in range(T):
        a = ref_logs['action'][t]
        ll_t = runner.action_logprob(obs_ids, a, t)
        ll_seq.append(ll_t)
        if 'hint_label' in ref_logs and ref_logs['hint_label']:
            next_obs = [ref_logs['hint_label'][t], ref_logs['reward_label'][t], ref_logs['choice_label'][t]]
        else:
            next_obs = ['null', ref_logs['reward_label'][t], ref_logs['choice_label'][t]]
        obs_ids = runner.obs_labels_to_ids(next_obs)

    return float(np.sum(ll_seq)), ll_seq


def evaluate_ll_with_valuefn_masked(value_fn, A, B, D, ref_logs, mask_indices):
    """Compute masked total log-likelihood over a subset of trials.

    Runs the model forward over the full trial sequence (so internal
    agent state updates match the original run), returns the full per-trial
    log-likelihood sequence and the sum over `mask_indices`.

    Parameters:
    -----------
    value_fn : callable
        Value function factory for the model (q_context, t) -> (C_t, E_t, gamma_t)
    A, B, D : arrays
        Generative model components (passed to agent)
    ref_logs : dict
        Reference run logs with keys including 'action', 'reward_label',
        'hint_label', 'choice_label'.
    mask_indices : sequence of int
        Trial indices to include in the summed log-likelihood.

    Returns:
    --------
    total_masked : float
        Sum of per-trial log-likelihoods over mask_indices
    ll_seq : list of float
        Full per-trial log-likelihood sequence
    """
    total, ll_seq = evaluate_ll_with_valuefn(value_fn, A, B, D, ref_logs)
    mask_set = set(int(i) for i in (mask_indices or []))
    if not mask_set:
        return 0.0, ll_seq
    masked_sum = 0.0
    for idx, ll in enumerate(ll_seq):
        if idx in mask_set:
            masked_sum += float(ll)
    return float(masked_sum), ll_seq


# Worker helpers for optional parallel evaluation. These reuse globals initialized
# by _worker_init to avoid pickling large A/B/D objects repeatedly.


def _worker_init(A, B, D):
    global A_GLOB, B_GLOB, D_GLOB, M3_POLICIES_GLOB, NUM_ACTIONS_PER_FACTOR_GLOB
    A_GLOB = A
    B_GLOB = B
    D_GLOB = D
    try:
        from pymdp.agent import Agent
        from pymdp import utils as pymdp_utils
        C_temp = pymdp_utils.obj_array_zeros([(A[m].shape[0],) for m in range(len(A))])
        temp_agent = Agent(A=A, B=B, C=C_temp, D=D,
                         policy_len=2, inference_horizon=1,
                         control_fac_idx=[1], use_utility=True,
                         use_states_info_gain=True,
                         action_selection="stochastic", gamma=16)
        M3_POLICIES_GLOB = temp_agent.policies
        NUM_ACTIONS_PER_FACTOR_GLOB = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
    except Exception:
        M3_POLICIES_GLOB = None
        NUM_ACTIONS_PER_FACTOR_GLOB = None


def compute_sequence_ll_for_model_worker(model_name, ref_logs):
    try:
        A_loc = A_GLOB
        B_loc = B_GLOB
        D_loc = D_GLOB
    except NameError:
        raise RuntimeError("Worker globals not initialized")
    return compute_sequence_ll_for_model(model_name, A_loc, B_loc, D_loc, ref_logs)


def _eval_m1_gamma(A, B, D, g_val, ref_logs):
    if A is None:
        try:
            A_loc = A_GLOB
            B_loc = B_GLOB
            D_loc = D_GLOB
        except NameError:
            raise RuntimeError("Worker globals not initialized for _eval_m1_gamma")
    else:
        A_loc, B_loc, D_loc = A, B, D
    value_fn = make_value_fn('M1', C_reward_logits=M1_DEFAULTS['C_reward_logits'], gamma=g_val)
    total_ll, _ = evaluate_ll_with_valuefn(value_fn, A_loc, B_loc, D_loc, ref_logs)
    return total_ll


def _eval_m1_gamma_masked(A, B, D, g_val, ref_logs, mask_indices):
    """Worker helper: evaluate M1 gamma on masked trials (sum over mask_indices)."""
    if A is None:
        try:
            A_loc = A_GLOB
            B_loc = B_GLOB
            D_loc = D_GLOB
        except NameError:
            raise RuntimeError("Worker globals not initialized for _eval_m1_gamma_masked")
    else:
        A_loc, B_loc, D_loc = A, B, D

    value_fn = make_value_fn('M1', C_reward_logits=M1_DEFAULTS['C_reward_logits'], gamma=g_val)
    total_ll, _ = evaluate_ll_with_valuefn_masked(value_fn, A_loc, B_loc, D_loc, ref_logs, mask_indices)
    return total_ll


def _eval_m2_params(A, B, D, g_base, k, ref_logs):
    if A is None:
        try:
            A_loc = A_GLOB
            B_loc = B_GLOB
            D_loc = D_GLOB
        except NameError:
            raise RuntimeError("Worker globals not initialized for _eval_m2_params")
    else:
        A_loc, B_loc, D_loc = A, B, D

    def gamma_schedule(q, t, g_base=g_base, k=k):
        p = np.clip(np.asarray(q, float), 1e-12, 1.0)
        H = -(p * np.log(p)).sum()
        return g_base / (1.0 + k * H)
    value_fn = make_value_fn('M2', C_reward_logits=M2_DEFAULTS['C_reward_logits'], gamma_schedule=gamma_schedule)
    total_ll, _ = evaluate_ll_with_valuefn(value_fn, A_loc, B_loc, D_loc, ref_logs)
    return total_ll


def _eval_m2_params_masked(A, B, D, g_base, k, ref_logs, mask_indices):
    if A is None:
        try:
            A_loc = A_GLOB
            B_loc = B_GLOB
            D_loc = D_GLOB
        except NameError:
            raise RuntimeError("Worker globals not initialized for _eval_m2_params_masked")
    else:
        A_loc, B_loc, D_loc = A, B, D

    def gamma_schedule(q, t, g_base=g_base, k=k):
        p = np.clip(np.asarray(q, float), 1e-12, 1.0)
        H = -(p * np.log(p)).sum()
        return g_base / (1.0 + k * H)

    value_fn = make_value_fn('M2', C_reward_logits=M2_DEFAULTS['C_reward_logits'], gamma_schedule=gamma_schedule)
    total_ll, _ = evaluate_ll_with_valuefn_masked(value_fn, A_loc, B_loc, D_loc, ref_logs, mask_indices)
    return total_ll


def _eval_m3_params(A, B, D, g_val, xi_scale, ref_logs):
    if A is None:
        try:
            policies = M3_POLICIES_GLOB
            num_actions_per_factor = NUM_ACTIONS_PER_FACTOR_GLOB
            A_loc = A_GLOB
            B_loc = B_GLOB
            D_loc = D_GLOB
        except NameError:
            policies = None
            num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
            A_loc, B_loc, D_loc = A, B, D
    else:
        policies = None
        num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
        A_loc, B_loc, D_loc = A, B, D

    profiles = []
    for p in M3_DEFAULTS['profiles']:
        prof = dict(p)
        prof['gamma'] = g_val
        prof['xi_logits'] = (np.array(p['xi_logits'], float) * xi_scale).tolist()
        profiles.append(prof)

    value_fn = make_value_fn('M3', profiles=profiles, Z=np.array(M3_DEFAULTS['Z']), policies=policies, num_actions_per_factor=num_actions_per_factor)
    total_ll, _ = evaluate_ll_with_valuefn(value_fn, A_loc, B_loc, D_loc, ref_logs)
    return total_ll


def _eval_m3_params_masked(A, B, D, g_val, xi_scale, ref_logs, mask_indices):
    if A is None:
        try:
            policies = M3_POLICIES_GLOB
            num_actions_per_factor = NUM_ACTIONS_PER_FACTOR_GLOB
            A_loc = A_GLOB
            B_loc = B_GLOB
            D_loc = D_GLOB
        except NameError:
            policies = None
            num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
            A_loc, B_loc, D_loc = A, B, D
    else:
        policies = None
        num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
        A_loc, B_loc, D_loc = A, B, D

    profiles = []
    for p in M3_DEFAULTS['profiles']:
        prof = dict(p)
        prof['gamma'] = g_val
        prof['xi_logits'] = (np.array(p['xi_logits'], float) * xi_scale).tolist()
        profiles.append(prof)

    value_fn = make_value_fn('M3', profiles=profiles, Z=np.array(M3_DEFAULTS['Z']), policies=policies, num_actions_per_factor=num_actions_per_factor)
    total_ll, _ = evaluate_ll_with_valuefn_masked(value_fn, A_loc, B_loc, D_loc, ref_logs, mask_indices)
    return total_ll


def _eval_m3_params_per_profile(A, B, D, gammas, xi_scales_profile, ref_logs):
    """Worker helper: evaluate M3 with per-profile gammas and per-profile xi scales.

    Parameters:
    - gammas: sequence of per-profile gamma values
    - xi_scales_profile: sequence (length=num_profiles) of 3-tuples (scales for hint,left,right)
    - ref_logs: single run logs to evaluate
    """
    if A is None:
        try:
            policies = M3_POLICIES_GLOB
            num_actions_per_factor = NUM_ACTIONS_PER_FACTOR_GLOB
            A_loc = A_GLOB
            B_loc = B_GLOB
            D_loc = D_GLOB
        except NameError:
            policies = None
            num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
            A_loc, B_loc, D_loc = A, B, D
    else:
        policies = None
        num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
        A_loc, B_loc, D_loc = A, B, D

    profiles = []
    num_profiles = len(M3_DEFAULTS['profiles'])
    for p_idx, p in enumerate(M3_DEFAULTS['profiles']):
        prof = dict(p)
        prof['gamma'] = float(gammas[p_idx])
        orig_xi = np.array(p['xi_logits'], float)
        scales3 = xi_scales_profile[p_idx]
        new_xi = orig_xi.copy()
        # scales3 expected to be length-3 for [hint,left,right]
        new_xi[1] = orig_xi[1] * float(scales3[0])
        new_xi[2] = orig_xi[2] * float(scales3[1])
        new_xi[3] = orig_xi[3] * float(scales3[2])
        prof['xi_logits'] = new_xi.tolist()
        profiles.append(prof)

    value_fn = make_value_fn('M3', profiles=profiles, Z=np.array(M3_DEFAULTS['Z']), policies=policies, num_actions_per_factor=num_actions_per_factor)
    total_ll, _ = evaluate_ll_with_valuefn(value_fn, A_loc, B_loc, D_loc, ref_logs)
    return total_ll


def _eval_m3_params_per_profile_masked(A, B, D, gammas, xi_scales_profile, ref_logs, mask_indices):
    if A is None:
        try:
            policies = M3_POLICIES_GLOB
            num_actions_per_factor = NUM_ACTIONS_PER_FACTOR_GLOB
            A_loc = A_GLOB
            B_loc = B_GLOB
            D_loc = D_GLOB
        except NameError:
            policies = None
            num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
            A_loc, B_loc, D_loc = A, B, D
    else:
        policies = None
        num_actions_per_factor = [len(ACTION_CONTEXTS), len(ACTION_CHOICES)]
        A_loc, B_loc, D_loc = A, B, D

    profiles = []
    for p_idx, p in enumerate(M3_DEFAULTS['profiles']):
        prof = dict(p)
        prof['gamma'] = float(gammas[p_idx])
        orig_xi = np.array(p['xi_logits'], float)
        scales3 = xi_scales_profile[p_idx]
        new_xi = orig_xi.copy()
        new_xi[1] = orig_xi[1] * float(scales3[0])
        new_xi[2] = orig_xi[2] * float(scales3[1])
        new_xi[3] = orig_xi[3] * float(scales3[2])
        prof['xi_logits'] = new_xi.tolist()
        profiles.append(prof)

    value_fn = make_value_fn('M3', profiles=profiles, Z=np.array(M3_DEFAULTS['Z']), policies=policies, num_actions_per_factor=num_actions_per_factor)
    total_ll, _ = evaluate_ll_with_valuefn_masked(value_fn, A_loc, B_loc, D_loc, ref_logs, mask_indices)
    return total_ll
