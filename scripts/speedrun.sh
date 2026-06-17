#!/usr/bin/env bash
set -euo pipefail

# Mythos-Nano end-to-end speedrun (nanochat-inspired)
CONFIG="${1:-configs/nano.yaml}"
STEPS="${2:-100}"

echo "==> Phase 1: Pretrain"
python -m mythos.train --config "$CONFIG" --steps "$STEPS"

echo "==> Phase 2: Proxy eval"
mythos-eval --mode proxy --limit 5 --output eval/results/speedrun.json

echo "==> Phase 3: SFT stub"
CKPT="checkpoints/$(basename "$CONFIG" .yaml)/latest.pt"
python -m mythos.posttrain --config "$CONFIG" --checkpoint "$CKPT" --stage sft

echo "==> Done. Results in eval/results/speedrun.json"
