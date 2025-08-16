"""Dataset generator for the TextArena environment.

Creates a small dataset with conversation prompts for several TextArena
games. The first observation returned from ``env.init`` is included in
each sample so training can begin immediately.
"""

import argparse
import os
from datasets import Dataset
from omegaconf import DictConfig

from skyrl_gym.envs.textarena.env import TextArenaEnv


GAMES = ["codenames", "colonel_blotto", "three_player_ipd"]


def main(output_dir: str, num_examples: int) -> None:
    examples = []
    system_prompt = {
        "role": "system",
        "content": "You are playing a TextArena game. Respond with your move.",
    }
    for i in range(num_examples):
        game = GAMES[i % len(GAMES)]
        env = TextArenaEnv(DictConfig({}), {"game": game})
        prompt, _ = env.init([system_prompt])
        examples.append(
            {
                "prompt": prompt,
                "env_class": "textarena",
                "game": game,
                "max_turns": 20,
            }
        )
    ds = Dataset.from_list(examples)
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")
    ds.to_parquet(train_path)
    ds.to_parquet(val_path)
    print(f"Wrote dataset with {len(ds)} samples to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/textarena")
    parser.add_argument("--num_examples", type=int, default=100)
    args = parser.parse_args()
    main(os.path.expanduser(args.output_dir), args.num_examples)
