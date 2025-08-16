from __future__ import annotations

from typing import Any, Dict, Tuple
import importlib
from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)


class TextArenaEnv(BaseTextEnv):
    """Wrapper around TextArena games.

    This environment pairs the acting LLM with an opponent LLM using the
    [TextArena](https://github.com/LeonGuertler/TextArena) framework. Only a
    subset of games are currently supported.

    Parameters
    ----------
    env_config:
        Configuration for the environment. May include a ``game`` field which
        selects the underlying TextArena game. Supported games are
        ``codenames``, ``colonel_blotto`` and ``three_player_ipd``.
    extras:
        Extra keyword arguments forwarded to the TextArena game constructor.
        The ``game`` can also be specified here on a per-sample basis.
        ``max_turns`` limits the episode length.
    """

    SUPPORTED_GAMES = {"codenames", "colonel_blotto", "three_player_ipd"}

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] | None = None):
        super().__init__()
        extras = extras or {}

        # The game can be supplied either via ``env_config`` or per-sample ``extras``.
        if "game" in env_config:
            game_name = env_config["game"]
        elif "game" in extras:
            game_name = extras.pop("game")
        else:
            raise AssertionError("`game` field is required in env_config or extras")

        game_name = str(game_name).lower()
        if game_name not in self.SUPPORTED_GAMES:
            raise ValueError(f"Unsupported TextArena game: {game_name}")

        # Episode length
        self.max_turns = extras.get("max_turns", 20)

        # Conversation history
        self.chat_history: ConversationType = []

        try:
            textarena = importlib.import_module("textarena")
        except Exception as exc:  # pragma: no cover - handled in tests
            raise ImportError(
                "The `textarena` package is required to use TextArenaEnv."
            ) from exc

        make_fn = None
        for attr in ("make", "make_game", "load_game"):
            if hasattr(textarena, attr):
                make_fn = getattr(textarena, attr)
                break
        if make_fn is None:  # pragma: no cover - depends on external package
            raise AttributeError(
                "Could not locate a game creation function in the `textarena` package."
            )

        # Forward all extra arguments except max_turns to the arena
        arena_kwargs = {k: v for k, v in extras.items() if k != "max_turns"}
        self.env = make_fn(game_name, **arena_kwargs)

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        result = self.env.reset()
        if isinstance(result, tuple):
            observation, info = result
        else:
            observation, info = result, {}

        first_msg = {"role": "user", "content": observation} if observation else None
        self.chat_history = prompt.copy()
        if first_msg:
            self.chat_history.append(first_msg)
            return prompt + [first_msg], info
        return prompt, info

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})
        observation, reward, done, info = self.env.step(action)
        obs_msg = {"role": "user", "content": observation} if observation else None
        if obs_msg:
            self.chat_history.append(obs_msg)
            observations = [obs_msg]
        else:
            observations = []

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=reward,
            done=done,
            metadata=info,
            postprocessed_action=action,
        )
