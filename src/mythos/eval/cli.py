from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mythos.config import MythosConfig
from mythos.eval.composite import compute_mythos_score
from mythos.eval.harness import run_full_eval, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Mythos unified eval")
    parser.add_argument("--config", default="configs/nano.yaml")
    parser.add_argument("--mode", choices=["proxy", "full"], default="proxy")
    parser.add_argument("--model", default=None, help="Checkpoint path (.pt)")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", default="eval/results/latest.json")
    args = parser.parse_args()

    config = MythosConfig.from_yaml(args.config)
    model_path = args.model or str(Path("checkpoints") / config.name / "latest.pt")

    raw = run_full_eval(
        model_path=model_path if Path(model_path).exists() else None,
        config=config,
        limit=args.limit,
        mode=args.mode,
    )
    composite = compute_mythos_score(raw)
    payload = {
        "mode": args.mode,
        "checkpoint": model_path if Path(model_path).exists() else None,
        "mythos_score": composite.mythos_score,
        "components": composite.components,
        "unavailable": composite.unavailable,
        "raw": asdict(raw),
    }
    out = Path(args.output)
    save_results(out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
