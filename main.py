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


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def parse_env_params(items: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--env-param must look like name=value")
        key, value = item.split("=", 1)
        params[key] = parse_value(value)
    return params


def build_env(env_class: type, seed: int, params: dict[str, Any]) -> Any:
    params = {**params, "seed": seed}
    try:
        return env_class(**params)
    except TypeError:
        params.pop("seed", None)
        return env_class(**params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env",
        help="Environment class as module.path:ClassName",
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument(
        "--env-param",
        action="append",
        default=[],
        help="Environment constructor parameter as name=value. Can be repeated.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Render an ASCII view of each step (slow; observation only, "
        "agent behavior is unchanged).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_class = load_env_class(args.env)
    env = build_env(env_class, seed=args.env_seed, params=parse_env_params(args.env_param))
    report = discover(env, num_episodes=args.episodes, seed=args.seed, watch=args.watch)
    print(explain(report))


if __name__ == "__main__":
    main()
