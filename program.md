# AutoMythos Agent Constitution

## Mission

Optimize **MYTHOS_SCORE** — composite capability across SWE, cyber, code, math, and pretrain efficiency — via autonomous edits to `src/mythos/train.py` only.

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

- Removing eval oracles or reward functions
- Benchmark contamination (SWE-bench repo IDs in training data)
- Disabling safety router or serve safeguards
- Editing `program.md`, eval harnesses, or safety classifiers to inflate scores

## Promotion rules

- Proxy improvement must replicate 3 times before full `--depth` run
- Full eval every 50 kept experiments
- Scale nano → medium → frontier only when proxy score plateaus (Chinchilla gate)

## Benchmark weights

| Component | Weight |
|---|---|
| SWE-bench Verified | 0.25 |
| HumanEval | 0.20 |
| ExploitBench ladder | 0.15 |
| GSM8K | 0.15 |
| MMLU | 0.10 |
| FrontierCode proxy | 0.10 |
| val_bpb (efficiency) | 0.05 |

## Safety

All cyber/bio training is defensive framing only. Mythos tier requires access tier; public API serves Fable with routing.
