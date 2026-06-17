# Security & Responsible-Use Posture

Project Mythos is an educational small-LLM research lab. It is **defensive and
evaluation-oriented only.**

## What this project does NOT build

- **No offensive cyber capability.** No exploit/PoC generation, no arbitrary-code-
  execution or sandbox-escape tooling, no automated vulnerability *weaponization*
  against real software. The earlier `agents/cyber` scaffold (a static table of
  pre-labeled flags + stubs) is being replaced by a **read-only defensive
  secure-coding / vulnerability-comprehension** evaluation: given source code or a
  published CVE description, can the model *explain* the issue or *spot* a bug. That
  is understanding and defense — never exploitation.
- **No biological capability uplift.** Replaced by general science knowledge via the
  standard MMLU science subsets.
- **No detection evasion, mass targeting, or destructive tooling.**

## Honesty invariants (enforced by tests)

- No reported metric may be independent of the trained checkpoint. A score that does
  not change when the model changes is a bug. `tests/regression/test_no_fake_wins.py`
  enforces this.
- No fabricated/hardcoded benchmark numbers. Missing capability → report `unavailable`,
  never a constant.

## Deployment safety demo

The dual-tier router (`src/mythos/router/`) is a **responsible-deployment
demonstration**: it classifies prompts and routes flagged categories to a refusal or
weaker fallback. It is illustrative of safe deployment patterns, not a bypass
mechanism, and grants no capability.

## Reporting

This is a personal research/portfolio repo. For concerns, open an issue.
