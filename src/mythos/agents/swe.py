"""SWE agent evaluation.

Two honest layers:
- Real SWE-bench (`evaluate_swe_proxy`) stays `None`/unavailable until mini-swe-agent
  + Docker + the dataset are wired — we never fabricate a pass rate.
- A local, runnable **code-repair** eval (`run_micro_swe_eval`) with a deterministic
  execution oracle: a candidate fix is run against a hidden unit test in a subprocess
  and graded pass/fail. This proves the agentic-eval loop genuinely executes and
  discriminates, without Docker. Candidates are executed in a temp dir with a timeout;
  for untrusted/model-generated code a stronger sandbox should be used.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


def evaluate_swe_proxy(limit: int = 10) -> dict[str, float | None]:
    """SWE-bench requires mini-swe-agent + Docker — unavailable until integrated."""
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


# --- Local runnable code-repair eval (deterministic execution oracle) ---------

MICRO_TASKS: list[dict[str, str]] = [
    {
        "id": "add",
        "buggy": "def add(a, b):\n    return a - b\n",
        "fix": "def add(a, b):\n    return a + b\n",
        "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n",
    },
    {
        "id": "is_even",
        "buggy": "def is_even(n):\n    return n % 2 == 1\n",
        "fix": "def is_even(n):\n    return n % 2 == 0\n",
        "test": "assert is_even(4) is True\nassert is_even(3) is False\n",
    },
    {
        "id": "max_of",
        "buggy": "def max_of(xs):\n    return min(xs)\n",
        "fix": "def max_of(xs):\n    return max(xs)\n",
        "test": "assert max_of([1, 9, 2]) == 9\n",
    },
    {
        "id": "reverse",
        "buggy": "def reverse(s):\n    return s\n",
        "fix": "def reverse(s):\n    return s[::-1]\n",
        "test": "assert reverse('abc') == 'cba'\n",
    },
    {
        "id": "factorial",
        "buggy": "def factorial(n):\n    return n\n",
        "fix": "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
        "test": "assert factorial(5) == 120\nassert factorial(0) == 1\n",
    },
]

Solver = Callable[[dict[str, str]], str]


def oracle_solver(task: dict[str, str]) -> str:
    """Baseline that returns the known fix — proves the harness scores 1.0 when correct."""
    return task["fix"]


def noop_solver(task: dict[str, str]) -> str:
    """Baseline that returns the unchanged buggy code — proves the harness discriminates."""
    return task["buggy"]


def _candidate_passes(code: str, test: str, timeout: int = 10) -> bool:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candidate.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code.rstrip() + "\n\n" + test)
        try:
            result = subprocess.run(
                [sys.executable, path], capture_output=True, timeout=timeout, cwd=d
            )
            return result.returncode == 0
        except Exception:
            return False


def run_micro_swe_eval(
    solver: Solver,
    limit: int | None = None,
    timeout: int = 10,
) -> dict[str, Any]:
    """Run each task through `solver`, execute the candidate against its test, grade."""
    tasks = MICRO_TASKS[:limit] if limit else MICRO_TASKS
    results = []
    passed = 0
    for task in tasks:
        candidate = solver(task)
        ok = _candidate_passes(candidate, task["test"], timeout=timeout)
        passed += int(ok)
        results.append({"id": task["id"], "passed": ok})
    n = len(tasks)
    return {
        "micro_swe_pass_at_1": (passed / n) if n else None,
        "n": n,
        "passed": passed,
        "results": results,
    }
