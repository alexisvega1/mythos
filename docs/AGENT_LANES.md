# Multi-Agent Implementation Plan — Claude Code · Cursor · Codex

Three coding agents work this repo in parallel. **Each agent uses its own git
worktree** so commits never collide mid-push. File lanes are enforced on top of
that isolation.

## Worktree isolation (mandatory)

| Agent | Checkout | Branch prefix | Lane |
|-------|----------|---------------|------|
| **Claude Code** | `/Users/alexisvega/mythos-laneB` (B) · `/Users/alexisvega/mythos` (A) | `cc/*` | A + B — train/eval + serve/router |
| **Cursor** | `/Users/alexisvega/mythos-cursor` | `cur/*` | Tooling — CI, scripts, dashboards |
| **Codex** | `/Users/alexisvega/mythos-codex` | `cdx/*` | C — posttrain, agent evals |

Bootstrap (from any checkout):

```bash
bash scripts/setup-worktrees.sh all   # or: cursor | claude | codex
git worktree list
```

**Rules**

1. Never edit files in another agent's checkout — only push branches and merge via PR.
2. One agent = one lane = one branch prefix. Flag any cross-lane touch in the PR body.
3. Branch from latest `origin/main`; PR into `main`; keep `pytest` green.
4. If two PRs touch the same file, **Lane A wins on contracts** (`checkpoint` schema,
   `RawScores` keys); others rebase and consume.

### Collision note (2026-06-17)

PR #2 crossed lanes (Cursor P0 in Claude's tree). Merged OK. **Claude Code now owns
Lane B serve/router in `mythos-laneB`. Cursor must not touch `serve/` or `router/`.**

---

## Golden rules (every agent)

1. **Honesty invariant:** no metric independent of the trained checkpoint; missing
   capability → `None`/`unavailable`, never a constant. **Defensive/eval-only** (no
   offensive cyber/bio). See `SECURITY.md`.
2. **Contracts are owned by Lane A (Claude).** Other lanes consume them; don't redefine.
3. **Shared files** (`config.py`, `pyproject.toml`) → additive changes only, called out
   in the PR. `PLAN.md`/`SECURITY.md`/`program.md` are Claude-owned; propose via PR.

## Lane assignments

### Lane A — Claude Code · *core: make it learn + honest eval* · `cc/*`
**Checkout:** `/Users/alexisvega/mythos` (or `mythos-claude` worktree)

**Owns:** `src/mythos/data/**`, `src/mythos/train.py`, `src/mythos/model/gpt.py`,
`src/mythos/checkpoint.py`, `src/mythos/eval/harness.py`, `src/mythos/eval/composite.py`,
`tests/regression/test_no_fake_wins.py`, `data/corpus/**`, `data/fixtures/**`

**Status:** PR #2 merged (P0); P1 pretrain continues here.

### Lane B — Claude Code · *serve + router* · `cc/*` · `mythos-laneB`
**Checkout:** `/Users/alexisvega/mythos-laneB`

**Owns:** `src/mythos/serve/**`, `src/mythos/router/**`

**Tasks:** real checkpoint-backed `MythosEngine`, OpenAI-compatible API, dual-tier
defensive router. **Cursor must not touch these paths.**

**Status:** in progress — `cc/serve-real` (`inference.py`, `api.py` rewire).

### Cursor tooling lane · *CI, scripts, dashboards* · `cur/*`
**Checkout:** `/Users/alexisvega/mythos-cursor` **only**

**Owns:** `.github/workflows/**`, `scripts/**`, `eval/dashboards/**`, optional
`lab.py` / `cli.py` polish, README (not `PLAN.md` / `SECURITY.md` / `program.md`)

**Tasks:** CI matrix with no-fake-wins gate; worktree/cloud bootstrap scripts;
dashboard reading real `results.tsv` runs.

### Lane C — Codex · *isolated capability modules* · `cdx/*`
**Checkout:** `/Users/alexisvega/mythos-codex`

**Owns:** `src/mythos/posttrain.py`, `agents/swe/**`, `agents/secsec/**`;
removes legacy `agents/cyber/**` and `agents/bio/**`.

## Shared contracts (Lane A owns, B & C consume)

**Checkpoint schema** — `torch.save` to `checkpoints/<name>/latest.pt`:

```python
{
  "model": state_dict,
  "config": config.__dict__,
  "tokenizer": {"kind": "char|gpt2", "vocab": {...}},
  "step": int,
  "val_loss": float,
  "val_bpb": float,
}
```

Plus `checkpoints/<name>/meta.json` and `samples.txt`.

**Eval RawScores** — keys: `val_bpb`, `humaneval_pass_at_1`, `gsm8k_acc`,
`mmlu_macro`, `secsec_comprehension`, `swe_bench_lite_pass_at_1`. Unmeasurable →
`None`; composite renormalizes. Never emit 0 or limit-derived constants.

**Stable surfaces:** the `[project.scripts]` entry points and config field names must not
break. Add config fields; don't rename/remove.

## Sequencing

```
Lane A P0 (checkpoint + RawScores)  ──unblocks──▶  Lane B (serve real ckpt)
                                    └──unblocks──▶  Lane C (posttrain + eval)
```

## Status board

- [x] PR #1 — honest reset (merged)
- [x] PR #2 — P0 honest lab (merged)
- [ ] **B** — serve + router *(Claude · `mythos-laneB` · `cc/serve-real`)*
- [ ] **Cursor** — CI + dashboards + scripts *( `mythos-cursor` · `cur/*`)*
- [ ] **C** — posttrain + swe/secsec eval *(Codex · `mythos-codex` · `cdx/*`)*

---

## Copy-paste kickoff — Cursor (tooling lane)

```text
You are on /Users/alexisvega/mythos-cursor ONLY. Branch cur/<topic> from latest main.
Edit CI (.github/workflows/), scripts/, eval/dashboards/, optional lab/cli polish.
Do NOT touch src/mythos/serve/, src/mythos/router/ (Claude Code · mythos-laneB),
data/, train.py, eval harness, or agents/. Keep pytest green. Never fake a metric.
```

## Copy-paste kickoff — Claude Code (Lane A + B)

```text
Lane A: /Users/alexisvega/mythos — data/, train, model, checkpoint, eval core.
Lane B: /Users/alexisvega/mythos-laneB — serve/, router/ only (cc/serve-real).
Branch cc/<topic>. Do not edit posttrain/ or agents/. Publish contract changes in
AGENT_LANES before others integrate. Never fake a metric.
```

## Copy-paste kickoff — Codex (Lane C)

```text
You are Lane C on /Users/alexisvega/mythos-codex. Branch cdx/<topic>. Own posttrain.py,
agents/swe/, agents/secsec/. Consume checkpoint contract from AGENT_LANES. Real evals
only; report None when unavailable. Defensive-only. Never fake a metric.
```
