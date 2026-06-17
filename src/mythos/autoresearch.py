"""
AutoMythos self-recursive research loop — optimizes real held-out val_bpb.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mythos.config import MythosConfig
from mythos.eval.composite import compute_mythos_score
from mythos.eval.harness import run_full_eval
from mythos.train import train_run


RESULTS_HEADER = (
    "timestamp\texperiment\tval_bpb\tmythos_score\tkept\tcommit\tnotes\n"
)


def append_result(path: Path, row: dict) -> None:
    if not path.exists():
        path.write_text(RESULTS_HEADER)
    score = row["mythos_score"]
    score_str = "" if score is None else f"{score:.6f}"
    line = (
        f"{row['timestamp']}\t{row['experiment']}\t{row['val_bpb']:.6f}\t"
        f"{score_str}\t{row['kept']}\t{row.get('commit', '')}\t{row.get('notes', '')}\n"
    )
    with path.open("a") as f:
        f.write(line)


def git_commit(message: str) -> str | None:
    subprocess.run(["git", "add", "-A"], check=False, capture_output=True)
    result = subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "nothing to commit" in result.stdout + result.stderr:
        return None
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return rev.stdout.strip() if rev.returncode == 0 else None


def git_reset() -> None:
    subprocess.run(["git", "reset", "--hard", "HEAD"], check=False, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], check=False, capture_output=True)


def load_best_bpb(results_path: Path) -> float:
    if not results_path.exists():
        return float("inf")
    best = float("inf")
    for line in results_path.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                best = min(best, float(parts[2]))
            except ValueError:
                continue
    return best


def run_experiment(
    config_path: str,
    budget_minutes: int,
    results_path: Path,
    experiment_id: int,
    eval_limit: int = 5,
) -> dict:
    config = MythosConfig.from_yaml(config_path)
    budget_seconds = budget_minutes * 60

    train_metrics = train_run(config, budget_seconds=budget_seconds)
    checkpoint = train_metrics["checkpoint"]
    raw = run_full_eval(model_path=checkpoint, config=config, limit=eval_limit, mode="proxy")
    raw.val_bpb = train_metrics["val_bpb"]
    composite = compute_mythos_score(raw)

    best_bpb = load_best_bpb(results_path)
    improved = train_metrics["val_bpb"] < best_bpb
    kept = improved and composite.mythos_score is not None

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment_id,
        "val_bpb": train_metrics["val_bpb"],
        "mythos_score": composite.mythos_score,
        "kept": kept,
        "notes": json.dumps(
            {
                "train": train_metrics,
                "components": composite.components,
                "unavailable": composite.unavailable,
            }
        ),
    }

    if kept:
        commit = git_commit(f"autoresearch: exp {experiment_id} val_bpb={train_metrics['val_bpb']:.4f}")
        row["commit"] = commit or ""
    else:
        git_reset()
        row["commit"] = ""

    append_result(results_path, row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoMythos autoresearch loop")
    parser.add_argument("--config", default="configs/nano.yaml")
    parser.add_argument("--budget-minutes", type=int, default=5)
    parser.add_argument("--results", default="results.tsv")
    parser.add_argument("--max-experiments", type=int, default=0, help="0 = NEVER STOP")
    parser.add_argument("--eval-limit", type=int, default=5)
    args = parser.parse_args()

    results_path = Path(args.results)
    experiment = 0
    try:
        while args.max_experiments == 0 or experiment < args.max_experiments:
            experiment += 1
            row = run_experiment(
                args.config,
                args.budget_minutes,
                results_path,
                experiment,
                eval_limit=args.eval_limit,
            )
            print(json.dumps(row, indent=2))
            if args.max_experiments == 0:
                time.sleep(1)
    except KeyboardInterrupt:
        print("AutoMythos stopped by human.", file=sys.stderr)


if __name__ == "__main__":
    main()
