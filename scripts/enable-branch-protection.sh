#!/usr/bin/env bash
# Enable GitHub branch protection on main via Rulesets API.
# Requires: gh auth with repo admin scope.
set -euo pipefail

REPO="${1:-alexisvega1/mythos}"

echo "Creating/updating ruleset 'protect-main' on ${REPO}..."

# List existing rulesets named protect-main
EXISTING=$(gh api "repos/${REPO}/rulesets" --jq '.[] | select(.name=="protect-main") | .id' 2>/dev/null | head -1 || true)

PAYLOAD=$(cat <<'EOF'
{
  "name": "protect-main",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": {
      "include": ["~DEFAULT_BRANCH"],
      "exclude": []
    }
  },
  "rules": [
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": false,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": false
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "required_status_checks": [
          {"context": "test"},
          {"context": "honesty gate (test_no_fake_wins)"},
          {"context": "regression"},
          {"context": "smoke-train"}
        ]
      }
    },
    {
      "type": "non_fast_forward"
    },
    {
      "type": "deletion"
    }
  ]
}
EOF
)

if [[ -n "$EXISTING" ]]; then
  gh api "repos/${REPO}/rulesets/${EXISTING}" --method PUT --input - <<< "$PAYLOAD"
  echo "Updated ruleset id=${EXISTING}"
else
  gh api "repos/${REPO}/rulesets" --method POST --input - <<< "$PAYLOAD"
  echo "Created protect-main ruleset"
fi

echo "Verify:"
gh api "repos/${REPO}/rulesets" --jq '.[] | select(.name=="protect-main") | {id, enforcement, target}'
