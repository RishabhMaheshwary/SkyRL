import pytest
from omegaconf import OmegaConf

from skyrl_gym.envs.textarena.env import TextArenaEnv


def test_textarena_env_init():
    pytest.importorskip("textarena")
    cfg = OmegaConf.create({})
    env = TextArenaEnv(cfg, {"game": "codenames"})
    prompt, info = env.init([])
    assert isinstance(prompt, list)
    assert isinstance(info, dict)
