"""Entry point. The agent meets whatever world is supplied."""

from __future__ import annotations

import argparse
import importlib
from typing import Any

from agi.agent import discover, explain


def load_env_class(spec: str) -> type:
    if ":" not in spec:
        raise ValueError("env spec must look like 'module.path:ClassName'")

    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_env(env_class: type, seed: int) -> Any:
    try:
        return env_class(seed=seed)
    except TypeError:
        return env_class()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env",
        help="Environment class as module.path:ClassName",
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_class = load_env_class(args.env)
    env = build_env(env_class, seed=args.env_seed)
    report = discover(env, num_episodes=args.episodes, seed=args.seed)
    print(explain(report))


if __name__ == "__main__":
    main()
