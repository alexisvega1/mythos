#!/usr/bin/env bash
# One-command Mythos demo — venv bootstrap, skip rebuild if checkpoints exist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "→ Creating .venv …"
  python3 -m venv "${ROOT}/.venv"
fi

if ! "$PY" -c "import mythos" 2>/dev/null; then
  echo "→ Installing mythos …"
  "$PY" -m pip install -q -e "${ROOT}"
fi

exec "$PY" -m mythos.demo_cli --no-install "$@"
