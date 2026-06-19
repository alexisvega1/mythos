# Agent pickup — 2026-06-18

**Branch:** `cur/frontier-demo` → PR into `main`  
**Agent:** Cursor (Lane B polish)  
**Checkout:** `/Users/alexisvega/mythos` (also sync `mythos-cursor` worktree after merge)

## What landed (this PR)

| Area | Change |
|------|--------|
| **One-command demo** | `make demo` / `mythos-demo` / `scripts/run-demo.sh` — bootstraps `.venv`, `pip install -e .` (no lm-eval), skips retrain if `demo/assets/ckpt*/latest.pt` exist, auto port + browser |
| **CLI** | `src/mythos/demo_cli.py` + `mythos-demo` entry in `pyproject.toml` |
| **Demo UX** | SSE streaming, duel mode (base vs SFT), pipeline timeline, health bar (`demo/index.html`) |
| **Sampling fixes** | `serve/inference.py`: stop on EOS, `repetition_penalty`, used by demo with `top_p=0.92` |
| **SFT prompt wrap** | `demo/serve_demo.py` auto-wraps plain text in instruct template for SFT engine |
| **Build flags** | `demo/build_demo.py --quick` (80 steps), `--skip-if-ready` |
| **Tests** | `tests/unit/test_demo_cli.py` |

## What works (verified)

```bash
make demo                    # ~instant serve if ckpts exist; ~1 min rebuild with --rebuild
```

- Metrics chart from committed `demo/assets/run.json` (real run: 21.1M params, val_bpb 1.60 < unigram 1.93)
- Engines: `base` + `sft` load on MPS/CPU
- **SFT chip prompts** (e.g. "What is the capital of France?") → one clean answer after sampling fix
- **Base chip** (`ROMEO:`) → Shakespeare continuation (rough but real)
- Safety router on flagged prompts
- `tests/integration/test_serve.py` green after inference changes

## Honest limitations (do not “fix” with fake metrics)

- **21M params, 500 pretrain steps, 40 SFT examples** — not ChatGPT. Open-ended prompts on SFT will pattern-match training facts (`data/fixtures/sft_instructions.jsonl`).
- Demo sells the **pipeline** (train → post-train → serve → route), not frontier IQ.
- Checkpoints (`*.pt`, ~84MB each) are **gitignored**; `run.json` is committed for the chart only.

## Do not redo

- Re-implement `make demo` bootstrap — use `scripts/run-demo.sh`
- Re-add `pip install -e ".[all]"` as demo prerequisite — core install is enough
- Rebuild demo on every serve — `--skip-if-ready` / `mythos-demo` skips when ckpts exist

## High-value next tasks (pick one lane)

### Cursor `cur/*` — demo & tooling
1. **Prebuilt ckpt download** — GitHub release or `scripts/fetch-demo-ckpt.sh` so fresh clones get live chat without ~2 min train
2. **Dashboard polish** — `eval/dashboards/render.py` link from demo UI; autoresearch ticker from real `results.tsv`
3. **CI job** — `pytest tests/unit/test_demo_cli.py` + smoke `TestClient` on `serve_demo` (no full train in CI)

### In progress (`cur/demo-model-v2`)
- **`configs/demo.yaml`** — 6L/256d (~48M params), 800 pretrain steps on Shakespeare
- **SFT expanded** — 88 instruction examples incl. Shakespeare-style responses (`data/fixtures/sft_instructions.jsonl`)
- **`demo/build_demo.py`** — uses demo config, 280 SFT steps, MythosEngine sampling for snapshots
- After rebuild: commit updated `demo/assets/run.json` with real metrics

### Claude `cc/*` — Lane A/B core
1. **Medium pretrain** — `configs/medium-smoke.yaml` longer run; commit new `samples.txt` + metrics
2. **Serve API parity** — expose `repetition_penalty` on `/v1/chat/completions` (demo already uses it internally)
3. **Regression flakes** — `test_no_fake_wins` failed intermittently on CPU in local run; investigate seed/device

### Codex `cdx/*` — post-train
1. **Richer SFT data** — more diverse instructions (still tiny) to reduce memorization loops
2. **RFT demo chip** — show execution-oracle reward on one micro task in UI

## Commands

```bash
make demo                      # recommended entry
mythos-demo --rebuild          # force retrain
make demo-quick                # 80-step CPU build
python demo/serve_demo.py --device mps

make test                      # full suite
pip install -e ".[all]"        # only when you need lm-eval / wandb
```

## Files touched (ownership)

| File | Lane |
|------|------|
| `demo/**`, `scripts/run-demo.sh`, `Makefile`, `eval/dashboards/**` | Cursor |
| `src/mythos/demo_cli.py` | Cursor (new entry point) |
| `src/mythos/serve/inference.py` | **Shared** — sampling fix; coordinate with Lane B if extending |
| `pyproject.toml` | Shared — additive `mythos-demo` script only |

## Merge note

After merge: `cd mythos-cursor && git fetch && git merge origin/main` so Cursor worktree stays current.
