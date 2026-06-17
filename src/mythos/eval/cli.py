from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mythos.eval.composite import compute_mythos_score
from mythos.eval.harness import run_full_eval, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Mythos unified eval")
    parser.add_argument("--mode", choices=["proxy", "full"], default="proxy")
    parser.add_argument("--model", default=None, help="HF model path or checkpoint")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", default="eval/results/latest.json")
    args = parser.parse_args()

    raw = run_full_eval(model_path=args.model, limit=args.limit, mode=args.mode)
    composite = compute_mythos_score(raw)
    payload = {
        "mode": args.mode,
        "mythos_score": composite.mythos_score,
        "components": composite.components,
        "raw": asdict(raw),
    }
    out = Path(args.output)
    save_results(out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
