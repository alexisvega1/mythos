# Project Mythos

Self-recursive frontier model inspired by Mythos-class capabilities: train → eval → mutate → keep/discard, starting at nano scale and scaling what the scoreboard proves.

## Architecture

- **Mythos tier**: full-capability weights (restricted access)
- **Fable tier**: same weights with safety routing to fallback model
- **AutoMythos**: Karpathy autoresearch loop optimizing composite `MYTHOS_SCORE`

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

## Benchmark targets

| Benchmark | Phase 3 target | Phase 5 target |
|---|---|---|
| SWE-bench Verified | 45%+ | 65%+ |
| HumanEval | 15%+ | — |
| ExploitBench ACE | 2+ bugs | Tier-3+ |
| BioMysteryBench proxy | 35%+ | — |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full scoreboard.

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
