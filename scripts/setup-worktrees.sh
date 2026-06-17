#!/usr/bin/env bash
# Create sibling git worktrees so Claude Code, Cursor, and Codex never share one
# working tree. Run from any checkout of alexisvega1/mythos.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
PARENT="$(dirname "$ROOT")"
REPO="$(basename "$ROOT")"

usage() {
  cat <<EOF
Usage: $(basename "$0") [cursor|claude|codex|all]

Creates isolated checkouts next to this repo:
  ${PARENT}/${REPO}-cursor   branch cur/lane-b-polish   (Cursor · tooling + serve polish)
  ${PARENT}/${REPO}-claude   branch cc/lane-a-core      (Claude Code · Lane A)
  ${PARENT}/${REPO}-laneB    branch cc/serve-real       (Claude Code · Lane B serve)

See docs/AGENT_LANES.md for file ownership.
EOF
}

add_worktree() {
  local suffix="$1" branch="$2" base="${3:-origin/main}"
  local path="${PARENT}/${REPO}-${suffix}"

  if [[ -d "$path" ]]; then
    echo "skip (exists): $path"
    return 0
  fi

  git fetch origin
  git worktree add "$path" -b "$branch" "$base" 2>/dev/null || git worktree add "$path" "$base"
  echo "created: $path  ($branch ← $base)"
}

target="${1:-all}"

case "$target" in
  cursor) add_worktree cursor cur/lane-b-polish ;;
  claude) add_worktree claude cc/lane-a-core ;;
  laneB)  add_worktree laneB cc/serve-real origin/main ;;
  codex)  add_worktree codex cdx/lane-c-agents ;;
  all)
    add_worktree claude cc/lane-a-core
    add_worktree cursor cur/lane-b-polish
    add_worktree laneB cc/serve-real origin/main
    ;;
  -h|--help|help) usage; exit 0 ;;
  *) echo "unknown target: $target" >&2; usage; exit 1 ;;
esac

echo
git worktree list
