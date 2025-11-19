import pytest
from two_step_task.data.data_loader import (
    row_to_obs_ids,
    planet_to_obs_idx,
    reward_to_obs_idx,
)


def test_planet_variety_inputs():
    # common variations
    assert planet_to_obs_idx('Red') == 1
    assert planet_to_obs_idx('r') == 1
    assert planet_to_obs_idx('PURPLE') == 2
    assert planet_to_obs_idx('p') == 2
    # numeric mapping
    assert planet_to_obs_idx(1) == 1
    assert planet_to_obs_idx(2) == 2
    # unknown -> null
    assert planet_to_obs_idx('unknown') == 0


def test_reward_variety_inputs():
    assert reward_to_obs_idx('1') == 2
    assert reward_to_obs_idx('0') == 1
    assert reward_to_obs_idx('yes_reward') == 2
    assert reward_to_obs_idx('') == 1


def test_row_to_obs_ids_missing_fields():
    # missing some keys should fallback to nulls
    row = {'stage1_choice': None}
    obs = row_to_obs_ids(row)
    assert obs == [0, 0, 0, 0]


def test_row_to_obs_ids_weird_types():
    row = {'stage1_choice': 'left', 'planet':  ' R ', 'stage2_choice': 'Right', 'reward': 'True'}
    obs = row_to_obs_ids(row)
    # spaceship mapping falls back to 0/1 depending on conversion; ensure types handled
    assert isinstance(obs, list)
    assert obs[1] == 1
    assert obs[3] == 2
