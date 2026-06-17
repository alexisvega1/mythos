# Project Mythos

A **small-LLM research lab**: train → eval → mutate → keep/discard, starting at nano
scale and scaling only what an honest scoreboard proves. Architecturally inspired by
frontier-lab patterns; **not** a frontier model and makes no such claim.

> ### STATUS: P0 honest lab (real data + checkpoint eval)
> Training uses **real/fixture text** with held-out **val_bpb**; eval loads checkpoints and
> refuses fake constants (`test_no_fake_wins`). Capability proxies (GSM8K, SWE, etc.) still
> return `None` until wired — see [`CURSOR_HANDOFF.md`](CURSOR_HANDOFF.md) for P1–P4.

## Architecture

- **Core**: small GPT (RoPE, QK-Norm, ReLU²) + Muon-family optimizer, config-scaled
- **AutoMythos**: Karpathy autoresearch loop optimizing a *real* held-out signal
- **Serving**: OpenAI-compatible API + a defensive safety-classifier router demo

## Quick start

```bash
pip install -e ".[all]"

# Smoke train (CPU/single GPU)
mythos-train --config configs/nano.yaml --steps 100

# Proxy eval
mythos-eval --mode proxy --limit 5

# AutoMythos loop (5-min budget per experiment)
mythos-autoresearch --budget-minutes 5

# Serve Mythos/Fable dual-tier API
mythos-serve --checkpoint checkpoints/latest
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
