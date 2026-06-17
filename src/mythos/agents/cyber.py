from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mythos.config import MythosConfig
from mythos.paths import repo_root


def _tasks_path() -> Path:
    return repo_root() / "agents" / "secsec" / "tasks.json"


def load_sec_tasks(limit: int = 10) -> list[dict]:
    path = _tasks_path()
    if not path.exists():
        legacy = repo_root() / "agents" / "cyber" / "sec_tasks.json"
        path = legacy if legacy.exists() else path
    if not path.exists():
        return []
    tasks = json.loads(path.read_text())
    return tasks[:limit]


def evaluate_sec_comprehension(
    model: torch.nn.Module,
    config: MythosConfig,
    limit: int = 10,
    device: str | torch.device = "cpu",
) -> float | None:
    """
    Defensive secure-coding comprehension: pick the vulnerability class for given code.
    Scored against the loaded model — no static flags.
    """
    from mythos.data.stream import get_tokenizer

    tasks = load_sec_tasks(limit)
    if not tasks:
        return None

    enc = get_tokenizer(config.data.tokenizer)
    model.eval()
    correct = 0
    with torch.no_grad():
        for task in tasks:
            prompt = task["prompt"]
            choices = task["choices"]
            answer = int(task["answer_idx"])
            scores = []
            for choice in choices:
                text = f"{prompt}\nAnswer: {choice}"
                tokens = enc.encode(text)
                if len(tokens) > config.block_size:
                    tokens = tokens[-config.block_size :]
                if len(tokens) < 2:
                    scores.append(float("-inf"))
                    continue
                inp = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
                tgt = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
                logits, _ = model(inp)
                log_probs = F.log_softmax(logits, dim=-1)
                total = sum(
                    float(log_probs[0, i, int(tgt[0, i].item())].item()) for i in range(tgt.size(1))
                )
                scores.append(total)
            pred = max(range(len(scores)), key=lambda i: scores[i])
            if pred == answer:
                correct += 1
    return correct / len(tasks)


def evaluate_exploitbench_proxy(limit: int = 5) -> dict[str, float | None]:
    """Deprecated offensive proxy — always unavailable."""
    return {"exploitbench_weighted_flags": None}


def run_cyber_grpo(config: Any, checkpoint: Path) -> dict:
    return {
        "stage": "sec_comprehension_eval",
        "checkpoint": str(checkpoint),
        "status": "eval_only",
    }
