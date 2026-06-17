# AutoMythos Agent Constitution

## Mission

Optimize **real held-out val_bpb** on a trained checkpoint. Downstream metrics
(gsm8k, defensive sec comprehension) count only when actually measured — never
hardcoded.

## NEVER STOP

1. Read this file + last 20 rows of `results.tsv`
2. Edit **only** `src/mythos/train.py`
3. Fixed-budget train on **real/fixture text** (never default synthetic for experiments)
4. Evaluate checkpoint with `mythos-eval --model checkpoints/.../latest.pt`
5. Keep if **val_bpb improves** on held-out split; else `git reset --hard`

## Allowed mutations

- Optimizer, architecture toggles, LR, batch, schedule
- Data config only via existing config fields (not bypassing held-out split)

## Forbidden

- Synthetic data as default for autoresearch runs
- Hardcoded eval constants or limit-derived scores
- Offensive cyber/bio capability code
- Removing `test_no_fake_wins.py`

## Benchmark weights (available metrics only)

| Component | Weight |
|---|---|
| Held-out val bpb | 0.55 |
| GSM8K (lm-eval on checkpoint) | 0.15 |
| Defensive sec comprehension | 0.15 |
| MMLU science | 0.10 |
| SWE-bench (when wired) | 0.05 |

Unavailable metrics are excluded and weights renormalized.

## Promotion

- Proxy improvement must replicate 3× before full-depth promotion
- Full eval every 50 kept experiments
