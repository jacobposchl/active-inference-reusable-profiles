"""Data loading utilities moved into `data` subpackage."""
from pathlib import Path
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(p)


def preprocess_two_step(df: pd.DataFrame, exclude_first_n: int = 9) -> pd.DataFrame:
    cols_needed = [
        "participant_id",
        "trial",
        "stage1_choice",
        "transition",
        "planet",
        "stage2_choice",
        "reward",
    ]
    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.dropna(subset=cols_needed)
    df["trial"] = df["trial"].astype(int)
    df["stage1_choice"] = df["stage1_choice"].astype(int)
    df["stage2_choice"] = df["stage2_choice"].astype(int)
    df["reward"] = df["reward"].astype(int)
    df = df.loc[~df.groupby("participant_id")["trial"].transform(lambda x: x <= exclude_first_n)]
    df = df.sort_values(["participant_id", "trial"]).reset_index(drop=True)
    return df


def load_and_preprocess(path: str, exclude_first_n: int = 9) -> pd.DataFrame:
    df = load_csv(path)
    return preprocess_two_step(df, exclude_first_n=exclude_first_n)


# Observation mapping helpers
OBS_INDEX_MAP = {
    'spaceship': {'left': 0, 'right': 1, 'null': 0},
    'planet': {'null': 0, 'red': 1, 'purple': 2},
    'alien': {'left': 0, 'right': 1},
    'reward': {'null': 0, 'no_reward': 1, 'reward': 2},
}


def planet_to_obs_idx(planet) -> int:
    """Map planet label (string or numeric) to planet-modality observation index.

    Returns index according to `OBS_INDEX_MAP['planet']`.
    """
    if planet is None:
        return OBS_INDEX_MAP['planet']['null']
    if isinstance(planet, str):
        key = planet.strip().lower()
        if key.startswith('r'):
            return OBS_INDEX_MAP['planet']['red']
        if key.startswith('p'):
            return OBS_INDEX_MAP['planet']['purple']
        # fallback: try exact match
        return OBS_INDEX_MAP['planet'].get(key, OBS_INDEX_MAP['planet']['null'])
    # numeric code
    try:
        v = int(planet)
        # if it's 1 or 2 we assume same mapping
        if v in (1, 2):
            return v
    except Exception:
        pass
    return OBS_INDEX_MAP['planet']['null']


def reward_to_obs_idx(reward) -> int:
    """Map reward (0/1 or string) to reward-modality observation index.

    Returns 2 for reward==1, 1 for no_reward (0), 0 for null/unknown.
    """
    if reward is None:
        return OBS_INDEX_MAP['reward']['null']
    if isinstance(reward, str):
        key = reward.strip().lower()
        if key in ('1', 'true', 'yes') or 'reward' in key:
            return OBS_INDEX_MAP['reward']['reward']
        return OBS_INDEX_MAP['reward']['no_reward']
    try:
        v = int(reward)
        return OBS_INDEX_MAP['reward']['reward'] if v == 1 else OBS_INDEX_MAP['reward']['no_reward']
    except Exception:
        return OBS_INDEX_MAP['reward']['no_reward']


def row_to_obs_ids(row) -> list:
    """Convert a processed participant row (pandas Series or dict-like) to obs_ids list.

    The returned list follows the modality order used by `build_A_matrices()`:
      [spaceship, planet, alien, reward]

    Values default to 0 (null) where information is not available.
    """
    obs = [0, 0, 0, 0]

    # spaceship: many fitting loops keep spaceship unobserved at stage1
    s1 = row.get('stage1_choice') if hasattr(row, 'get') else row.stage1_choice
    if s1 is not None:
        try:
            obs[0] = int(s1)
        except Exception:
            key = str(s1).strip().lower()
            obs[0] = OBS_INDEX_MAP['spaceship'].get(key, 0)

    # planet
    planet = row.get('planet') if hasattr(row, 'get') else row.planet
    obs[1] = planet_to_obs_idx(planet)

    # alien (stage2 choice)
    s2 = row.get('stage2_choice') if hasattr(row, 'get') else row.stage2_choice
    if s2 is not None:
        try:
            obs[2] = int(s2)
        except Exception:
            key = str(s2).strip().lower()
            obs[2] = OBS_INDEX_MAP['alien'].get(key, 0)

    # reward
    rew = row.get('reward') if hasattr(row, 'get') else row.reward
    obs[3] = reward_to_obs_idx(rew)

    return obs
