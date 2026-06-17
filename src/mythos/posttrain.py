from __future__ import annotations

import argparse
import json
from pathlib import Path

from mythos.config import MythosConfig


def run_sft(config: MythosConfig, checkpoint: Path) -> dict:
    """Instruction SFT stub — loads checkpoint metadata for pipeline integration."""
    return {"stage": "sft", "checkpoint": str(checkpoint), "status": "ready"}


def run_grpo(config: MythosConfig, checkpoint: Path, env: str = "swe") -> dict:
    """GRPO post-training entry — delegates to domain modules."""
    if env == "swe":
        from mythos.agents import swe as swe_agent

        return swe_agent.run_swe_grpo(config, checkpoint)
    if env == "cyber":
        from mythos.agents import cyber as cyber_agent

        return cyber_agent.run_cyber_grpo(config, checkpoint)
    raise ValueError(f"Unknown env: {env}")


def run_rft(config: MythosConfig, checkpoint: Path) -> dict:
    from mythos.agents import swe as swe_agent

    return swe_agent.run_swe_rft(config, checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mythos post-training")
    parser.add_argument("--config", default="configs/nano.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", choices=["sft", "rft", "grpo"], default="sft")
    parser.add_argument("--env", default="swe")
    args = parser.parse_args()
    config = MythosConfig.from_yaml(args.config)
    ckpt = Path(args.checkpoint)
    if args.stage == "sft":
        result = run_sft(config, ckpt)
    elif args.stage == "rft":
        result = run_rft(config, ckpt)
    else:
        result = run_grpo(config, ckpt, env=args.env)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
