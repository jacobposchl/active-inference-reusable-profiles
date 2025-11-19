import numpy as np
import pandas as pd
from pathlib import Path
from two_step_task.models.m1_static import M1_StaticPrecision
from two_step_task.models.m2_entropy import M2_EntropyCoupled
from two_step_task.models.m3_profiles import M3_ProfileBased
from two_step_task.fitting.fit_models import compute_log_likelihood

ROOT = Path(__file__).resolve().parents[2]
PFILE = ROOT / 'two_step_task' / 'data' / 'processed' / 'participant_sub1.csv'


def _params_for_model(model):
    bounds = model.free_parameters()
    return np.array([(low + high) / 2.0 for (low, high) in bounds.values()])


def test_compute_loglikelihood_all_models():
    df = pd.read_csv(PFILE)
    results = {}
    for ModelClass in (M1_StaticPrecision, M2_EntropyCoupled, M3_ProfileBased):
        model = ModelClass()
        params = _params_for_model(model)
        ll = compute_log_likelihood(model, params, df)
        assert isinstance(ll, float)
        assert not (np.isnan(ll) or np.isinf(ll))
