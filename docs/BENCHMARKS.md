# Benchmarks

## MYTHOS_SCORE composite

| Component | Weight | Normalize ceiling |
|---|---|---|
| SWE-bench Verified pass@1 | 0.25 | 0.80 |
| HumanEval pass@1 | 0.20 | 0.90 |
| ExploitBench weighted flags | 0.15 | 1.00 |
| GSM8K | 0.15 | 0.95 |
| MMLU macro | 0.10 | 0.90 |
| FrontierCode proxy | 0.10 | 0.30 |
| val_bpb efficiency | 0.05 | 1.50 (lower is better) |

## Staged targets

### Phase 1 — Nano ($100 tier)

| Metric | Target |
|---|---|
| CORE | ≥ 0.26 |
| HumanEval post-SFT | ≥ 15% |
| MMLU post-SFT | ≥ 35% |

### Phase 3 — SWE

| Metric | Target |
|---|---|
| SWE-bench Verified (mini-swe-agent) | ≥ 45% |
| SWE-bench Pro | ≥ 35% |
| FrontierCode proxy | ≥ 15% |

### Phase 4 — Cyber / Bio

| Metric | Target |
|---|---|
| ExploitBench ACE | ≥ 2 bugs |
| BioMysteryBench proxy | ≥ 35% |

## Running evals

```bash
# Proxy (fast, CI)
mythos-eval --mode proxy --limit 5

# Full suite (weekly)
mythos-eval --mode full --output eval/results/full.json
```

## Regression policy

CI fails if any primary metric drops >2% absolute vs `tests/regression/golden_scores.json`.

## Scaffold documentation

Always report which agent scaffold was used:

- SWE: mini-swe-agent (primary), OpenHands (ablation)
- Cyber: ExploitBench deterministic oracle
- Base LM: lm-evaluation-harness task list in `eval/harness/config.yaml`
