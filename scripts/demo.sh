#!/usr/bin/env bash
# End-to-end Mythos demo: train → eval → dashboard → live API.
# Usage: bash scripts/demo.sh [config] [steps] [port]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/test.yaml}"
STEPS="${2:-100}"
PORT="${3:-8765}"
HTTP_PORT="${MYTHOS_HTTP_PORT:-$((PORT + 1))}"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
elif [[ -f "/Users/alexisvega/mythos/.venv/bin/activate" ]]; then
  source "/Users/alexisvega/mythos/.venv/bin/activate"
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

NAME="$(python -c "from mythos.config import MythosConfig; print(MythosConfig.from_yaml('${CONFIG}').name)")"
CKPT="${ROOT}/checkpoints/${NAME}/latest.pt"
SAMPLES="${ROOT}/checkpoints/${NAME}/samples.txt"
META="${ROOT}/eval/dashboards/demo_meta.json"
DASH="${ROOT}/eval/dashboards/index.html"

cleanup() {
  [[ -n "${SERVE_PID:-}" ]] && kill "$SERVE_PID" 2>/dev/null || true
  [[ -n "${HTTP_PID:-}" ]] && kill "$HTTP_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Mythos demo (config=${CONFIG}, steps=${STEPS})"
echo "==> [1/4] Pretrain on real/fixture text"
TRAIN_LOG="$(mktemp)"
python -m mythos.train --config "$CONFIG" --steps "$STEPS" | tee "$TRAIN_LOG"

echo "==> [2/4] Checkpoint eval + sample generation"
EVAL_JSON="${ROOT}/eval/results/demo.json"
mkdir -p "$(dirname "$EVAL_JSON")"
python -m mythos.eval.cli --mode proxy --checkpoint "$CKPT" --limit 3 --output "$EVAL_JSON" 2>/dev/null || true

mkdir -p "$(dirname "$SAMPLES")"
python -c "
from pathlib import Path
from mythos.serve.inference import MythosEngine
eng = MythosEngine('${CKPT}', device='cpu')
text, _, _ = eng.generate('the king said', max_tokens=32, temperature=0.8)
Path('${SAMPLES}').write_text(text)
print('sample:', repr(text[:80]))
"

echo "==> [3/4] Write demo metadata + render dashboard"
python -c "
import json
from pathlib import Path
from eval.dashboards.demo_meta import write_demo_meta
train = json.loads(Path('${TRAIN_LOG}').read_text())
eval_path = Path('${EVAL_JSON}')
eval_extra = json.loads(eval_path.read_text()) if eval_path.exists() else {}
write_demo_meta(
    '${CKPT}',
    '${META}',
    config_name='${NAME}',
    train_extra=train,
    eval_extra=eval_extra,
    api_port=${PORT},
    samples_path='${SAMPLES}',
)
"
rm -f "$TRAIN_LOG"
python eval/dashboards/render.py --results results.tsv --demo-meta "$META" --api-port "$PORT" --output "$DASH"

echo "==> [4/4] Start API (port ${PORT}) + static dashboard (port ${HTTP_PORT})"
export MYTHOS_CHECKPOINT="$CKPT"
python -m mythos.serve.api --host 127.0.0.1 --port "$PORT" --device cpu &
SERVE_PID=$!
sleep 1.5

python -m http.server "$HTTP_PORT" --directory "${ROOT}/eval/dashboards" &
HTTP_PID=$!

DASH_URL="http://127.0.0.1:${HTTP_PORT}/index.html"
echo ""
echo "=========================================="
echo "  Mythos demo is live"
echo "  Dashboard: ${DASH_URL}"
echo "  API:       http://127.0.0.1:${PORT}/health"
echo "  Checkpoint: ${CKPT}"
echo "=========================================="
echo "Press Ctrl+C to stop."

if command -v open >/dev/null 2>&1; then
  open "$DASH_URL" || true
fi

wait "$SERVE_PID"
