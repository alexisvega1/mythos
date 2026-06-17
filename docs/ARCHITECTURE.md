# Mythos Architecture

## Overview

Project Mythos implements a self-recursive frontier model pipeline inspired by Mythos-class capabilities:

1. **Pretrain** (`src/mythos/train.py`) — GPT decoder with modded-nanogpt optimizations
2. **AutoMythos** (`src/mythos/autoresearch.py`) — fixed-budget experiments optimizing MYTHOS_SCORE
3. **Post-train** (`src/mythos/posttrain.py`) — SWE RFT/GRPO, cyber ladder RL
4. **Eval** (`src/mythos/eval/`) — composite benchmark scoreboard
5. **Serve** (`src/mythos/serve/`) — Mythos/Fable dual-tier API with safety routing

## Dual-tier model

Same checkpoint, different routing:

- **Mythos**: full weights for vetted partners (Glasswing cyber, bio-trust)
- **Fable**: classifiers route cyber/bio/distillation queries to fallback model

Router: `src/mythos/router/classifiers.py`

## AutoMythos loop

```
program.md → agent edits train.py → 5-min train → proxy eval → keep/revert
```

Research memory: `mythos-lab record|query`

## Data flow

```
configs/*.yaml → MythosConfig → GPT model → checkpoints/
                                      ↓
                              eval/composite MYTHOS_SCORE
                                      ↓
                              posttrain (SFT/RFT/GRPO)
                                      ↓
                              serve API (Mythos/Fable)
```

## External integrations

| Component | Upstream |
|---|---|
| Base eval | EleutherAI lm-evaluation-harness |
| SWE eval | mini-swe-agent scaffold |
| Cyber eval | ExploitBench-style ladder |
| RL | verifiers / prime-rl (optional) |
| Pretrain reference | karpathy/nanochat |
| Optimizers | KellerJordan/modded-nanogpt |

## Scale gates

Promote nano → medium → frontier when AutoMythos proxy score plateaus and val_bpb gate holds.
