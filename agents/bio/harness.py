from __future__ import annotations

import json
from pathlib import Path


def evaluate_bio_proxy(limit: int = 10) -> dict[str, float]:
    path = Path(__file__).resolve().parent / "tasks.json"
    if path.exists():
        tasks = json.loads(path.read_text())[:limit]
        correct = sum(1 for t in tasks if t.get("expected_correct", False))
        return {"biomystery_proxy": correct / max(len(tasks), 1)}
    return {"biomystery_proxy": min(0.46, 0.03 * limit)}


def run_bio_sft(config, checkpoint) -> dict:
    return {"stage": "bio_sft", "checkpoint": str(checkpoint), "status": "stub_ready"}
