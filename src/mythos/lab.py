"""
Research memory CLI (praxlab-inspired).

Records hypotheses, outcomes, and failures across AutoMythos sessions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MEMORY_PATH = Path("lab_memory.json")


def _load() -> list[dict[str, Any]]:
    if MEMORY_PATH.exists():
        return json.loads(MEMORY_PATH.read_text())
    return []


def _save(entries: list[dict[str, Any]]) -> None:
    MEMORY_PATH.write_text(json.dumps(entries, indent=2))


def record(hypothesis: str, outcome: str, kept: bool) -> None:
    entries = _load()
    entries.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hypothesis": hypothesis,
            "outcome": outcome,
            "kept": kept,
        }
    )
    _save(entries[-500:])


def query(term: str) -> list[dict[str, Any]]:
    entries = _load()
    term_lower = term.lower()
    return [e for e in entries if term_lower in e.get("hypothesis", "").lower() or term_lower in e.get("outcome", "").lower()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mythos research memory")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record")
    rec.add_argument("--hypothesis", required=True)
    rec.add_argument("--outcome", required=True)
    rec.add_argument("--kept", action="store_true")

    q = sub.add_parser("query")
    q.add_argument("term")

    args = parser.parse_args()
    if args.cmd == "record":
        record(args.hypothesis, args.outcome, args.kept)
        print("Recorded.")
    elif args.cmd == "query":
        results = query(args.term)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
