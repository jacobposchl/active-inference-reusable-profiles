import pandas as pd
import numpy as np
from two_step_task.data.data_loader import (
    row_to_obs_ids,
    planet_to_obs_idx,
    reward_to_obs_idx,
)


def test_planet_to_obs_idx_strings():
    assert planet_to_obs_idx('red') == 1
    assert planet_to_obs_idx('Red') == 1
    assert planet_to_obs_idx('purple') == 2
    assert planet_to_obs_idx('P') == 2
    assert planet_to_obs_idx(None) == 0


def test_reward_to_obs_idx():
    assert reward_to_obs_idx(1) == 2
    assert reward_to_obs_idx(0) == 1
    assert reward_to_obs_idx('reward') == 2
    assert reward_to_obs_idx('0') == 1
    assert reward_to_obs_idx(None) == 0


def test_row_to_obs_ids_from_series():
    row = pd.Series({
        'stage1_choice': 0,
        'planet': 'red',
        'stage2_choice': 1,
        'reward': 1,
    })
    obs = row_to_obs_ids(row)
    # expected ordering: [spaceship, planet, alien, reward]
    assert isinstance(obs, list)
    assert obs[0] in (0, 1)
    assert obs[1] == 1
    assert obs[2] in (0, 1)
    assert obs[3] == 2


def test_row_to_obs_ids_numeric_inputs():
    row = {'stage1_choice': 1, 'planet': 2, 'stage2_choice': 0, 'reward': 0}
    obs = row_to_obs_ids(row)
    assert obs == [1, 2, 0, 1]
