from __future__ import annotations

import argparse

from mythos.config import MythosConfig
from mythos.train import main as train_main
from mythos.train import train_run


def train_main_wrapper() -> None:
    train_main()


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mythos CLI")
    parser.add_argument("--config", default="configs/nano.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--budget-seconds", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def train_from_cli(argv: list[str] | None = None) -> dict:
    args = parse_train_args(argv)
    config = MythosConfig.from_yaml(args.config)
    return train_run(
        config,
        steps=args.steps,
        budget_seconds=args.budget_seconds,
        device=args.device,
    )
