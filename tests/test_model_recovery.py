from pathlib import Path

import pytest

from src.experiments import model_recovery as mr


@pytest.fixture()
def tmp_cwd(tmp_path, monkeypatch):
    """Run experiments inside a temporary directory so they don't pollute repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_kfold_cv_produces_results_and_csvs(tmp_cwd, monkeypatch):
    monkeypatch.setenv("MODEL_COMP_MAX_WORKERS", "1")
    results, diffs = mr.kfold_cv(
        generators=["M1"],
        runs_per_generator=1,
        num_trials=10,
        seed=123,
        reversal_interval=5,
        K=2,
    )

    assert "M1" in results
    assert "fold_mean" in results["M1"]
    assert isinstance(diffs, dict)

    csv_dir = Path("results") / "csv"
    assert csv_dir.exists()
    saved_files = list(csv_dir.iterdir())
    assert saved_files, "Expected fold CSV outputs"
    assert any("cv_recovery_fold" in f.name for f in saved_files)

