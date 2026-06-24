# Multi-Agent Implementation Plan — Claude Code · Cursor · Codex

Three coding agents work this repo in parallel. Use **separate git worktrees**
(`bash scripts/setup-worktrees.sh all`) so commits never collide mid-push.

## Worktrees

| Agent | Checkout | Branch prefix |
|-------|----------|---------------|
| Claude Code (A+B core serve) | `mythos-laneB` | `cc/*` |
| Cursor (B polish + tooling) | `mythos-cursor` | `cur/*` |
| Codex (C) | `mythos-codex` | `cdx/*` |

## Golden rules (every agent)

1. **One agent = one lane = one branch prefix.** Never edit files outside your lane
   without flagging it in the PR description.
2. Branch from latest `main`; PR into `main`; **keep `pytest` green**; Conventional Commits.
   **`main` must stay protected** — no direct pushes; see [`BRANCH_PROTECTION.md`](BRANCH_PROTECTION.md).
3. **Honesty invariant:** no metric independent of the trained checkpoint; missing
   capability → `None`/`unavailable`, never a constant. **Defensive/eval-only** (no
   offensive cyber/bio). See `SECURITY.md`.
4. **Contracts are owned by Lane A (Claude).** Other lanes consume them; don't redefine.
5. **Shared files** (`config.py`, `pyproject.toml`) → additive changes only, called out
   in the PR. `PLAN.md`/`SECURITY.md`/`program.md` are Claude-owned; propose via PR.

## Lane assignments

### Lane A — Claude Code · *core: make it learn + honest eval* · branch `cc/*`
**Owns:** `src/mythos/data/**`, `src/mythos/train.py`, `src/mythos/model/gpt.py`,
`src/mythos/eval/harness.py`, `src/mythos/eval/composite.py`,
`tests/regression/test_no_fake_wins.py`, `data/corpus/**`
**Why Claude:** correctness-critical path; runs locally on MPS to *actually train*.
**Tasks (P0/P1):** real tokenized data loader + held-out split; byte-accurate `val_bpb`;
checkpoint-conditioned eval; `None=unavailable` composite with renormalization;
no-fake-wins regression gate; a real tiny training run with committed `samples.txt`.
**Status:** in progress — branch `feat/real-training`, corpus fetched.

**Status:** PR #4 merged (core serve). Cursor polish: streaming, top-k/p, dashboard, CI.

### Lane B polish — Cursor · `cur/*` · `mythos-cursor`
**Owns:** serve extensions (streaming/sampling), `eval/dashboards/**`, `scripts/**`,
`.github/workflows/**`, worktree bootstrap, **`demo/**`**, **`mythos-demo` CLI**.

**Status (2026-06-18):** PR `cur/frontier-demo` — one-command demo (`make demo`),
streaming UI, EOS/repetition sampling fixes. Pickup: [`AGENT_PICKUP.md`](AGENT_PICKUP.md).

### Lane C — Codex · *isolated capability modules* · branch `cdx/*`
**Owns:** `src/mythos/posttrain.py`, `agents/swe/**`, `agents/secsec/**` (new);
removes `agents/cyber/**` and `agents/bio/**`.
**Tasks:** SFT a small chat model on instruction data (consume checkpoint contract);
integrate **mini-swe-agent** as a *real* eval scaffold on SWE-bench Lite (≥5 tasks, real
pass/fail, report honestly); build a read-only **defensive secure-coding comprehension**
eval (given code/CVE text → identify vuln class); fold bio into MMLU science via lm-eval.
Each eval returns a dict matching the RawScores contract below.

## Shared contracts (owned by Lane A, consumed by B & C)

**Checkpoint schema** — `torch.save` to `checkpoints/<name>/latest.pt`:
```python
{
  "model": state_dict,
  "config": config.__dict__,      # includes resolved vocab_size
  "tokenizer": {"kind": "char|gpt2", "vocab": {...}},  # decode without guessing
  "step": int,
  "val_loss": float,              # nats, held-out
  "val_bpb": float,               # bytes, held-out
}
```
Plus `checkpoints/<name>/meta.json` (human-readable mirror) and `samples.txt`.

**Eval RawScores** — every eval returns a `dict[str, float | None]` merged into
`mythos.eval.composite.RawScores`. Keys: `val_bpb`, `humaneval_pass_at_1`, `gsm8k_acc`,
`mmlu_macro`, `secsec_comprehension`, `swe_bench_lite_pass_at_1`. **Rule:** a metric you
cannot really measure on the checkpoint is `None` → composite excludes it and
renormalizes remaining weights. Never emit 0 or a `limit`-derived constant.

**Stable surfaces:** the `[project.scripts]` entry points and config field names must not
break. Add config fields; don't rename/remove.

## Sequencing

```
Lane A P0 (checkpoint + RawScores contract)  ──unblocks──▶  Lane B (serve real ckpt)
                                             └──unblocks──▶  Lane C (posttrain + agent eval)
```
A lands first (in progress). B and C can scaffold against the documented contracts now and
integrate the moment A merges.

## Status board

- [x] PR #4 — core serve + router *(Claude, merged)*
- [ ] **B polish** — streaming, dashboard, CI honesty gate *(Cursor, PR pending)*
- [ ] **C** — posttrain + SWE eval *(Claude — Codex not active yet)*

---

## Copy-paste kickoff — Cursor (Lane B)
```text
You are Lane B (product surface) on /Users/alexisvega/mythos. Read docs/AGENT_LANES.md,
PLAN.md, SECURITY.md. Work ONLY on your lane's files (serve/, router/, cli.py, lab.py,
eval/dashboards/, scripts/, .github/workflows/, docs). Branch cur/<topic> from latest main,
PR into main, keep pytest green. Consume the checkpoint + RawScores contracts in
AGENT_LANES.md — do not redefine them. Tasks: (1) make src/mythos/serve load and serve the
REAL checkpoint via an OpenAI-compatible streaming API; (2) wire the dual-tier router as a
DEFENSIVE safety-classifier demo (real classifier -> refusal/fallback, no capability);
(3) build a dashboard reading only real runs from results.tsv; (4) CI matrix incl. the
no-fake-wins gate. Honesty invariant + defensive-only apply. Never fake a metric.
```

## Copy-paste kickoff — Codex (Lane C)
```text
You are Lane C (capability modules) on /Users/alexisvega/mythos. Read docs/AGENT_LANES.md,
PLAN.md, SECURITY.md. Work ONLY on posttrain.py, agents/swe/, agents/secsec/ (new); delete
agents/cyber and agents/bio. Branch cdx/<topic> from latest main, PR into main, keep pytest
green. Consume the checkpoint + RawScores contracts in AGENT_LANES.md. Tasks: (1) SFT a
small chat model from the pretrained checkpoint on instruction data; (2) integrate
mini-swe-agent as a REAL eval scaffold on SWE-bench Lite (>=5 tasks, real pass/fail, report
honestly even if low/zero); (3) read-only DEFENSIVE secure-coding comprehension eval (given
code or CVE text, identify the vulnerability class) returning secsec_comprehension; (4) fold
bio into MMLU science via lm-eval. No offensive cyber/bio. Never fake a metric.
```
