import os

import numpy as np
import pytest

from config import experiment_config as cfg
from src.utils import recovery_helpers as rh


def test_build_abd_shapes():
    A, B, D = rh.build_abd()
    assert len(A) == cfg.NUM_MODALITIES
    assert len(B) == cfg.NUM_FACTORS
    assert len(D) == cfg.NUM_FACTORS

    # Likelihood shapes match config
    assert A[0].shape == (
        len(cfg.OBSERVATION_HINTS),
        len(cfg.STATE_CONTEXTS),
        len(cfg.STATE_CHOICES),
    )
    assert B[0].shape[0] == len(cfg.STATE_CONTEXTS)
    assert D[0].shape[0] == len(cfg.STATE_CONTEXTS)


def test_generate_all_runs_returns_reference_logs():
    generators = ['M1']
    A, B, D, refs = rh.generate_all_runs(
        generators,
        runs_per_generator=1,
        num_trials=8,
        seed=42,
        reversal_interval=4,
    )
    assert len(refs) == 1
    ref = refs[0]
    assert ref['gen'] == 'M1'
    required_keys = {'action', 'context', 'reward_label', 'choice_label', 'hint_label'}
    assert required_keys.issubset(ref['ref_logs'].keys())
    assert len(ref['ref_logs']['action']) == 8


def test_cv_fit_single_run_creates_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv('MODEL_COMP_MAX_WORKERS', '1')
    _, _, _, refs = rh.generate_all_runs(['M1'], runs_per_generator=1, num_trials=10, seed=7, reversal_interval=5)
    ref = refs[0]
    A, B, D = rh.build_abd()

    result = rh.cv_fit_single_run(
        'M1',
        A,
        B,
        D,
        ref['ref_logs'],
        K=2,
        run_id=ref['run_idx'],
        generator=ref['gen'],
        seed=ref['seed'],
        artifact_base_dir=str(tmp_path),
        save_artifacts=True,
        record_grid=False,
    )

    summary = result['summary']
    for key in ['mean_train_ll', 'mean_test_ll', 'mean_train_acc', 'mean_test_acc', 'runtime_sec']:
        assert key in summary
        assert np.isfinite(summary[key])

    assert result['trial_csv'] is not None
    assert os.path.exists(result['trial_csv'])
    assert os.path.exists(result['fold_csv'])
    assert os.path.exists(result['run_summary_csv'])

