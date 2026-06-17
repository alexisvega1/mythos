#!/usr/bin/env bash
# Render eval/dashboards/index.html from autoresearch results.tsv
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python eval/dashboards/render.py "$@"
