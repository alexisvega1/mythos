# Project Mythos

A **small-LLM research lab**: train → eval → post-train → serve, with honest metrics at
every stage. Architecturally inspired by frontier-lab patterns; **not** a frontier model
and makes no such claim — but the **full pipeline is real and reproducible**.

> ### STATUS: semi-functional lab (verified 2026-06-18)
> **Pretrain** on real text (Shakespeare corpus) → **val_bpb ~1.6** beats unigram baseline (~1.93).
> **SFT** on instruction data (loss 7.5→0.46). **RFT** with execution-oracle rewards.
> **Serve** real checkpoints via OpenAI-compatible API + Fable safety router.
> **Demo model (`configs/demo.yaml`):** 30.5M params, 6L/256d, val_bpb **1.56** beats unigram **1.94**, 88 SFT examples.
> **43+ tests** green including `test_no_fake_wins` honesty gate.
> Agent pickup: [`docs/AGENT_PICKUP.md`](docs/AGENT_PICKUP.md).

## See it in action (visual demo)

**One command** — creates venv, installs deps, skips retrain if checkpoints exist, opens browser:

```bash
make demo
# or: bash scripts/run-demo.sh
# or: pip install -e . && mythos-demo
```

Rebuild from scratch (~2–3 min on Apple Silicon, 30M params / 800 steps):

```bash
mythos-demo --rebuild
make demo-rebuild
```

CPU-friendly quick train:

```bash
make demo-quick
```

The UI streams tokens live, runs **base vs SFT duels**, shows the real training curve, and routes flagged prompts through the Fable safety tier. `demo/assets/run.json` ships committed metrics so the chart works even before a local build.

## Full pipeline

```bash
make install
make test                    # 41 tests, honesty gate included

make pretrain                # Shakespeare ~21M params, writes samples.txt
make sft                     # instruction fine-tune on pretrained ckpt
make rft                     # rejection fine-tuning with unit-test oracle

make speedrun                # fast test-config pipeline (~2 min CPU)
bash scripts/demo.sh         # train + interactive dashboard + API

mythos-autoresearch --budget-minutes 5   # optimize real val_bpb
```

## Architecture

| Stage | What | Honest? |
|-------|------|---------|
| Pretrain | GPT (RoPE, QK-Norm, ReLU², Flash SDPA) + NorMuon | Real text, real val_bpb |
| Eval | Held-out bpb, sec comprehension, gsm8k (if lm-eval installed) | None when unavailable |
| Post-train | SFT + RFT (execution oracle) + micro-SWE eval | Real loss / pass rates |
| Serve | `MythosEngine` + OpenAI API + Fable router | No fabricated replies |
| AutoResearch | Karpathy loop on val_bpb | `test_no_fake_wins` gate |

## Quick API serve

```bash
mythos-train --config configs/test.yaml --steps 100
export MYTHOS_CHECKPOINT=checkpoints/mythos-test/latest.pt
mythos-serve --checkpoint "$MYTHOS_CHECKPOINT"
curl -s localhost:8000/health | python -m json.tool
```

## Docs

- [`PLAN.md`](PLAN.md) — honest scope and phases
- [`demo/README.md`](demo/README.md) — visual demo details
- [`docs/BRANCH_PROTECTION.md`](docs/BRANCH_PROTECTION.md) — protect `main` for multi-agent work
- [`docs/AGENT_LANES.md`](docs/AGENT_LANES.md) — lane assignments

## License

MIT
