# AutoMythos Agent Constitution

## Mission

Optimize **MYTHOS_SCORE** — a composite of *really measured* capability (real
held-out language-modeling efficiency, code, math, knowledge, defensive
secure-coding) — via autonomous edits to `src/mythos/train.py` only.

## NEVER STOP

Run experiments until a human interrupts. Each experiment:

1. Read this file + last 20 rows of `results.tsv` + `mythos-lab query` memory
2. Propose 1–2 focused edits to `src/mythos/train.py`
3. `git commit` the edit
4. Run `mythos-autoresearch --max-experiments 1` OR train + eval manually
5. If MYTHOS_SCORE improved AND val_bpb did not regress >1% → keep
6. Else → `git reset --hard`
7. Log to `results.tsv` and `mythos-lab record`

## Allowed mutations

- Optimizer choice (muon, nor_muon, aurora)
- Architecture toggles (RoPE, QK-Norm, ReLU², untied embeddings)
- Hyperparameters (LR, batch, warmup, weight decay)
- Data mix ratios in config (via train.py reading config)
- Training schedule and grad accumulation

## Forbidden

- **Reporting any metric not conditioned on the trained checkpoint** (a score that
  doesn't change when the model changes is a bug, not a result)
- **Counting training on synthetic/random tokens as a real result** — real runs use
  real text on a held-out split
- **Offensive cyber or bio capability** — defensive comprehension/eval only (see `SECURITY.md`)
- Removing eval oracles or reward functions
- Benchmark contamination (SWE-bench repo IDs in training data)
- Disabling safety router or serve safeguards
- Editing `program.md`, eval harnesses, or safety classifiers to inflate scores

## Promotion rules

- Proxy improvement must replicate 3 times before full `--depth` run
- Full eval every 50 kept experiments
- Scale nano → medium → frontier only when proxy score plateaus (Chinchilla gate)

## Benchmark weights

Weights apply only to metrics that are **really measured on the checkpoint**. A
metric with no real implementation is `unavailable` and excluded (renormalize the
rest) — never scored as a constant.

| Component | Weight |
|---|---|
| Held-out val bits/byte (real text) | 0.30 |
| HumanEval (lm-eval on trained model) | 0.20 |
| GSM8K (lm-eval on trained model) | 0.15 |
| MMLU (incl. science) | 0.15 |
| Defensive secure-coding comprehension | 0.10 |
| SWE-bench Lite (mini-swe-agent, real) | 0.10 |

## Safety

Defensive and evaluation-only (see `SECURITY.md`). No offensive cyber (exploit/PoC/
ACE) or bio-uplift. The dual-tier router is a responsible-deployment demo, not a
capability gate.
