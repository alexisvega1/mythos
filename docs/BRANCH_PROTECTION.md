# Branch protection for `main`

Protect `main` so multiple agents (Claude Code, Cursor, Codex) cannot push or
merge broken or colliding work directly. This repo had a near-miss where a PR merged
while another agent was still pushing — branch protection + PR-only merges prevents that.

## Current state

**`main` is protected** via the `protect-main` ruleset (PR required + CI checks).
Re-apply after cloning: `bash scripts/enable-branch-protection.sh`

## Recommended ruleset (GitHub Settings → Rules → Rulesets)

Create or edit a ruleset (e.g. `protect-main`) with:

| Setting | Value |
|---------|--------|
| **Enforcement** | Active |
| **Target branches** | `main` (or Default branch) |
| **Require a pull request before merging** | On |
| **Require status checks to pass** | On (see below) |
| **Block force pushes** | On |
| **Restrict deletions** | On |

### Required status checks

These jobs run in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Add each
after it has run at least once on a PR (GitHub only lists checks that have executed):

| Check name | What it guards |
|------------|------------------|
| `test` | Unit + integration tests |
| `honesty gate (test_no_fake_wins)` | Anti-fake-win regression (`test_no_fake_wins.py`) |
| `regression` | Full regression suite |
| `smoke-train` | Smoke training integration test |

Optional stricter policy: require only `test` + `honesty gate` for faster merges; keep
`regression` and `smoke-train` as advisory until CI is stable.

## Skip for now

- **Require signed commits** — adds friction for local/agent workflows unless GPG/SSH signing is already set up.
- **Require linear history** — optional; squash merges are fine.
- **Require deployments** — not needed until a real deploy environment exists.
- **Restrict creations / Restrict updates** — overkill for a personal research repo.

## Bypass list

Keep **empty** unless you need emergency direct-push access. Repository admins can
always override in an emergency; agents should never be on the bypass list.

## Agent workflow (all lanes)

1. Branch from latest `origin/main` in your **worktree** (`cur/*`, `cc/*`, `cdx/*`).
2. Keep `pytest` green locally.
3. Open a PR; wait for required checks.
4. Merge via GitHub UI or `gh pr merge` — **never** `git push origin main`.

Worktree bootstrap: `bash scripts/setup-worktrees.sh all` (see [`AGENT_LANES.md`](AGENT_LANES.md)).

## Verify protection

```bash
gh api repos/alexisvega1/mythos/branches/main/protection
# Should return rules JSON, not 404 "Branch not protected"
```

Or: Settings → Branches / Rules → confirm the ruleset targets `main` and shows **Active**.
