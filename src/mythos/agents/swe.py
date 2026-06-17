from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate_swe_proxy(limit: int = 10) -> dict[str, float | None]:
    """SWE-bench requires mini-swe-agent — unavailable until integrated."""
    return {"swe_bench_verified_pass_at_1": None}


def run_swe_rft(config: Any, checkpoint: Path) -> dict:
    return {
        "stage": "swe_rft",
        "checkpoint": str(checkpoint),
        "status": "unavailable",
        "pattern": "SWE-RL rejection FT (not wired)",
    }


def run_swe_grpo(config: Any, checkpoint: Path) -> dict:
    return {
        "stage": "swe_grpo",
        "checkpoint": str(checkpoint),
        "status": "unavailable",
        "scaffold": "mini-swe-agent (not wired)",
    }


def run_mini_swe_agent_eval(model: str, limit: int = 10) -> dict[str, float | None]:
    return evaluate_swe_proxy(limit=limit)
