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
  ${PARENT}/${REPO}-cursor   branch cur/lane-b-serve   (Cursor · Lane B)
  ${PARENT}/${REPO}-claude   branch cc/lane-a-core     (Claude Code · Lane A)
  ${PARENT}/${REPO}-codex    branch cdx/lane-c-agents  (Codex · Lane C)

Each agent edits only its lane's files (see docs/AGENT_LANES.md).
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
  git worktree add "$path" -b "$branch" "$base"
  echo "created: $path  ($branch ← $base)"
}

target="${1:-cursor}"

case "$target" in
  cursor) add_worktree cursor cur/lane-b-serve ;;
  claude) add_worktree claude cc/lane-a-core ;;
  codex)  add_worktree codex cdx/lane-c-agents ;;
  all)
    add_worktree claude cc/lane-a-core
    add_worktree cursor cur/lane-b-serve
    add_worktree codex cdx/lane-c-agents
    ;;
  -h|--help|help) usage; exit 0 ;;
  *) echo "unknown target: $target" >&2; usage; exit 1 ;;
esac

echo
echo "Worktrees:"
git worktree list
