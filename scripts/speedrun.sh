#!/usr/bin/env bash
# Full pipeline speedrun: pretrain → eval → SFT → micro-SWE eval
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/test.yaml}"
STEPS="${2:-60}"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

NAME="$(python -c "from mythos.config import MythosConfig; print(MythosConfig.from_yaml('${CONFIG}').name)")"
CKPT="checkpoints/${NAME}/latest.pt"

echo "==> Pretrain"
python -m mythos.train --config "$CONFIG" --steps "$STEPS"

echo "==> Eval"
mkdir -p eval/results
mythos-eval --mode proxy --checkpoint "$CKPT" --limit 5 --output "eval/results/speedrun.json" || true

echo "==> SFT"
python -m mythos.posttrain --config "$CONFIG" --checkpoint "$CKPT" --stage sft --steps 80

SFT_CKPT="checkpoints/${NAME}-sft/latest.pt"
echo "==> Micro-SWE oracle eval"
python -c "
from mythos.agents.swe import run_micro_swe_eval, oracle_solver, noop_solver
print('oracle:', run_micro_swe_eval(oracle_solver, limit=5))
print('noop:', run_micro_swe_eval(noop_solver, limit=5))
"

echo "==> Done"
echo "  checkpoint: $CKPT"
echo "  sft:        $SFT_CKPT"
echo "  samples:    checkpoints/${NAME}/samples.txt"
