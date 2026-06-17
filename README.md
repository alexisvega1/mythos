# Project Mythos

A **small-LLM research lab**: train → eval → mutate → keep/discard, starting at nano
scale and scaling only what an honest scoreboard proves. Architecturally inspired by
frontier-lab patterns; **not** a frontier model and makes no such claim.

> ### STATUS: P0 honest lab + Lane B serve (real checkpoint API)
> Training uses **real/fixture text** with held-out **val_bpb**; eval loads checkpoints and
> refuses fake constants. **Serving** loads a real checkpoint via `MythosEngine` — no checkpoint
> → explicit `available: false` (never a stub reply). Fable router demo blocks flagged queries.
> Capability proxies (GSM8K, SWE) still `None` until wired — see [`CURSOR_HANDOFF.md`](CURSOR_HANDOFF.md).

## Architecture

- **Core**: small GPT (RoPE, QK-Norm, ReLU²) + Muon-family optimizer, config-scaled
- **AutoMythos**: Karpathy autoresearch loop optimizing a *real* held-out signal
- **Serving**: OpenAI-compatible API + a defensive safety-classifier router demo

## Quick start

```bash
pip install -e ".[all]"

# Mythos end-to-end demo — train, serve, live dashboard
bash scripts/demo.sh                    # fast (test config, ~2 min CPU)
bash scripts/demo.sh configs/nano.yaml 200 8765   # larger Shakespeare run

# Or step-by-step:

# Proxy eval
mythos-eval --mode proxy --limit 5

# AutoMythos loop (5-min budget per experiment)
mythos-autoresearch --budget-minutes 5

# Serve Mythos/Fable dual-tier API (real checkpoint required for generation)
mythos-train --config configs/test.yaml --steps 100   # produces checkpoints/mythos-test/latest.pt
export MYTHOS_CHECKPOINT=checkpoints/mythos-test/latest.pt
mythos-serve --checkpoint "$MYTHOS_CHECKPOINT"

# Without a checkpoint: clean queries return available=false (no fabricated text).
# Flagged queries (exploit/shellcode/etc.) route to Fable tier regardless.
curl -s localhost:8000/health | jq .
```

## Honest targets (small scale)

These are goals for a *small* model and will be filled in only with numbers from real
runs — no fabricated or model-independent metrics (see [`SECURITY.md`](SECURITY.md)).

| Metric | Goal |
|---|---|
| Held-out val bits/byte (real text) | beat unigram baseline by a documented margin |
| Coherent text generation | grammatical samples committed to `samples.txt` |
| HumanEval / GSM8K (lm-eval on the trained model) | report real pass@1, however low |
| Defensive secure-coding comprehension | identify vuln class in given code |
| SWE-bench Lite (mini-swe-agent, real scaffold) | end-to-end loop on ≥5 tasks, real pass/fail |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Targets unmet stay blank, not faked.

## Layout

```
src/mythos/     Core training, model, optimizers, serving
eval/           Harness wrappers and composite scoring
agents/         SWE, cyber, bio agent integrations
configs/        nano / medium / frontier YAML
program.md      AutoMythos agent constitution
```

## License

MIT
