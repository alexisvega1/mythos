# Project Mythos — Definitive Plan

> Status: **honest reset.** This document replaces the "frontier model in one-shot"
> framing with what is actually buildable, valuable, and true. Read it before
> trusting any number in this repo.

## 0. The reality check (read this first)

You cannot build a frontier model (Opus/Mythos/Fable class) in one shot, in a chat
session, or in this repo. That requires ~$10M–$100M+ of compute, a large team,
proprietary data pipelines, and months of work. No plan changes that. Anyone who
tells you otherwise is selling a fantasy.

What you **can** build — and what is genuinely impressive to an AI or connectomics
lab — is a **complete, reproducible, honest small-LLM laboratory** that actually
learns from real text, is actually measured, and improves itself against a real
signal. That is a real systems-and-eval portfolio piece. This plan builds that.

### What this repo actually is today (ground truth, verified 2026-06-16)

The previous build produced a clean 69-file scaffold with 23 passing tests — but
**nothing in it learns or measures learning:**

| Component | File | Reality |
|---|---|---|
| Training data | `src/mythos/data/stream.py:12` | `SyntheticTokenStream` yields **uniformly random tokens**. The model learns to predict noise. `val_bpb` is meaningless. |
| LM eval | `src/mythos/eval/harness.py:46` | `_proxy_lm_scores` returns `min(0.15, 0.02*limit)` — a function of the CLI flag, **not the model**. The real `lm_eval` branch discards its result (line 43) and returns the proxy anyway, and would have run `gpt2`, not the trained checkpoint. |
| SWE / cyber / bio | `agents/*/` | Static JSON of pre-labeled boolean flags + stub functions returning `{"status": "stub_ready"}`. No execution, no model in the loop. |
| MYTHOS_SCORE | `eval/composite.py` | Composes the above constants. **The autoresearch loop is optimizing the eval limit, not the model.** |

The architecture (GPT with RoPE/QK-Norm/ReLU², Muon-family optimizer, configs,
CLI, serving, CI) is real and well-structured. It is an excellent **skeleton**. The
job is to make the skeleton tell the truth.

## 1. North star (honest)

A **nanochat-grade, fully reproducible, honest small-LLM lab** that:

1. **Actually trains** a small GPT on real text.
2. **Actually evaluates** it with metrics conditioned on the trained checkpoint.
3. **Self-improves** via a Karpathy-style autoresearch loop optimizing a *real*
   held-out signal — with anti-reward-hacking guards.
4. **Serves** the real model behind a clean OpenAI-compatible API with a
   **responsible-deployment / safety-classifier demo** (defensive only).

Success = every number in `results.tsv` and the README traces to a real run on a
real model, and a freshly-initialized model scores near chance while a trained one
beats it.

## 2. Scope guardrails

**In scope.** Real data loading, real tokenization, real (tiny) pretrain + SFT,
real model-conditioned eval, honest autoresearch, serving, defensive-security
*comprehension* eval, safety-router demo, reproducibility/CI.

**Explicitly out of scope, and why.**
- **Literal frontier parity** — compute/data infeasible; pretending otherwise is dishonest.
- **Offensive cyber capability** (exploit/PoC generation, ACE, sandbox escape against
  real software). Dual-use offensive tooling, not appropriate to build here, and
  non-functional at this scale regardless. Replaced by a **read-only defensive
  secure-coding / vulnerability-comprehension** eval (explain a known CVE, spot a bug
  in given code) — understanding, never weaponization.
- **Bio capability uplift.** Replaced by general science knowledge via standard MMLU.
- **Any fabricated or model-independent metric.** A score that doesn't depend on the
  checkpoint is a bug, not a benchmark. See `SECURITY.md` and `program.md`.

## 3. Phases (each ships an honest, tested artifact)

### Phase 0 — Make it real & honest  *(no GPU; the most important phase)*
Convert the Potemkin skeleton into a real lab.
- **Real data:** replace/augment `stream.py` with a tokenized-text loader (TinyStories
  or a FineWeb-Edu sample via HF `datasets` streaming + `tiktoken`). Keep a tiny
  synthetic stream only as an explicit, clearly-labeled unit-test fixture.
- **Real eval:** `harness.py` loads the **actual checkpoint** and computes real
  held-out val loss/bpb; wire `lm-eval` to evaluate *that* model (not `gpt2`, not a
  constant). If `lm-eval` is absent, return `None`/`NaN` and mark the metric
  `unavailable` — never a fake constant.
- **Honest objective:** autoresearch optimizes real held-out bpb (+ optional real
  downstream), not the limit.
- **Anti-fake-win test (gate):** `tests/regression/test_no_fake_wins.py` — a
  random-init model must score ~chance; a trained model must beat it; eval output
  must change when the checkpoint changes.
- **Acceptance:** tests green; deleting the checkpoint changes the scores.

### Phase 1 — Real tiny pretrain  *(MPS or one small GPU, ~$0–50)*
Train ~10–50M params on real text to a real, reported val loss.
- **Acceptance:** model emits grammatical text; held-out bpb beats a unigram baseline
  by a documented margin; `samples.txt` committed; numbers reproducible from seed.

### Phase 2 — Honest autoresearch loop
Karpathy autoresearch over `train.py` optimizing real held-out bpb.
- Held-out shard never trained on; contamination check; 3× replication before a "keep".
- **Acceptance:** ≥1 reproducible improvement over the Phase-1 baseline, logged with
  real numbers and replicated 3×.

### Phase 3 — Post-train + real agent eval  *(optional GPU)*
SFT a small chat model; integrate **mini-swe-agent** as a *real* eval scaffold on
SWE-bench Lite. Low/zero pass rate at this scale is fine — **report it honestly.**
- **Acceptance:** end-to-end agent loop runs on ≥5 real tasks with real pass/fail.

### Phase 4 — Serving + safety demo
vLLM/FastAPI OpenAI-compatible serving of the real checkpoint. Dual-tier router as a
**defensive safety-classifier demo** (flagged prompts → refusal/fallback), framed as
responsible deployment. Defensive secure-coding comprehension eval.
- **Acceptance:** API serves the real model; router demo documented; no offensive
  capability present.

### Scale gate
Rent bigger GPUs only when the honest scoreboard shows the small model plateauing
**and** there is a concrete reason to scale (Chinchilla-style check). Document the
decision in `results.tsv`.

## 4. Cost model (honest)

| Phase | Hardware | Est. cost |
|---|---|---|
| 0 Make it real | laptop CPU/MPS | $0 |
| 1 Tiny pretrain | MPS or 1× small GPU spot | $0–50 |
| 2 Autoresearch | 1× GPU, short runs | $20–100 |
| 3 Post-train + agent | 1–8× GPU, optional | $50–500 |
| 4 Serving demo | 1× GPU or CPU demo | ~$0 ongoing |

## 5. Definition of done

- [ ] No metric in the repo is independent of the trained checkpoint.
- [ ] `test_no_fake_wins` passes and is in CI.
- [ ] A trained small model generates coherent text and beats baseline bpb.
- [ ] Autoresearch produces ≥1 replicated real improvement.
- [ ] README states the honest scope; no "frontier"/offensive claims.
- [ ] No offensive cyber/bio code; `SECURITY.md` documents the posture.
- [ ] Every README/`results.tsv` number is reproducible from a seed + command.

See `CURSOR_HANDOFF.md` for the implementation brief handed to Cursor Composer.
