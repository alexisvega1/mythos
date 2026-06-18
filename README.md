# Project Mythos

A **small-LLM research lab**: train → eval → post-train → serve, with honest metrics at
every stage. Architecturally inspired by frontier-lab patterns; **not** a frontier model
and makes no such claim — but the **full pipeline is real and reproducible**.

> ### STATUS: semi-functional lab (verified 2026-06-17)
> **Pretrain** on real text (Shakespeare corpus) → **val_bpb ~1.6** beats unigram baseline (~1.93).
> **SFT** on instruction data (loss 7.5→0.46). **RFT** with execution-oracle rewards.
> **Serve** real checkpoints via OpenAI-compatible API + Fable safety router.
> **41 tests** green including `test_no_fake_wins` honesty gate.

## See it in action (visual demo)

Best for showing employers — training curve, base↔SFT chat toggle, safety routing:

```bash
pip install -e ".[all]"
make demo          # trains ~21M model on Shakespeare + SFT (~1 min on MPS)
# open http://127.0.0.1:8000
```

Or use the committed metrics immediately (chart renders; run `make demo-build` for live chat):

```bash
python demo/serve_demo.py
```

`demo/assets/run.json` has **real** training curve data from a verified run.

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
