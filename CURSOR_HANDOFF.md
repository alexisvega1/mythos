# Cursor Composer 2.5 — Implementation Brief: Make Mythos Real

Paste this whole file into Cursor Composer as the task. It is the authoritative
spec. Work in tandem with the plan in `PLAN.md` and the posture in `SECURITY.md`.

---

## Role & mission

You are implementing in `/Users/alexisvega/mythos`. The repo is a clean, well-built
**skeleton** with 23 passing tests — but it is a Potemkin village: **nothing in it
actually learns or measures learning.** Your mission is to make it tell the truth:
turn it into a real, reproducible, honest small-LLM lab. This is NOT a frontier
model and you must never claim it is.

## Ground truth you must internalize (verify, don't trust me)

1. `src/mythos/data/stream.py` — `SyntheticTokenStream` yields **uniformly random
   tokens**. Training learns nothing; `val_bpb` is meaningless noise.
2. `src/mythos/eval/harness.py` — `_proxy_lm_scores` returns `min(0.15, 0.02*limit)`,
   a function of the CLI flag, not the model. The `lm_eval` branch (line ~43)
   **discards** its real result and returns the proxy anyway, and targets `gpt2`
   instead of the trained checkpoint.
3. `agents/{swe,cyber,bio}/` — static JSON flags + `stub_ready` functions. No model
   in the loop.
4. `eval/composite.py` — composes the above constants into `MYTHOS_SCORE`. The
   autoresearch loop therefore optimizes the eval limit, not the model.

So: **every capability number in this repo is currently fake.** Fixing that is the
entire job.

## Non-negotiable constraints

- **Honesty invariant.** No reported metric may be independent of the trained
  checkpoint. If a capability isn't really measured, report `None`/`unavailable` —
  never a hardcoded constant. Add and keep green
  `tests/regression/test_no_fake_wins.py` (spec below).
- **Defensive only.** Do NOT build offensive cyber (exploit/PoC/ACE/sandbox-escape)
  or bio-uplift. Replace `agents/cyber` with a read-only secure-coding /
  vulnerability-**comprehension** eval; replace `agents/bio` with MMLU science. See
  `SECURITY.md`.
- **Keep the suite green** at every commit. Never delete a test to make CI pass;
  update it to assert the new, honest behavior.
- **No fabricated results.** Every number you put in `results.tsv`, README, or docs
  must come from a real run you actually executed, reproducible from a seed + command.
- **Small, reviewable commits.** Conventional Commits. Work on a branch, open a PR.
  End commit messages with the Co-Authored-By trailer the user uses.
- **Don't fight the docs.** `PLAN.md`, `SECURITY.md`, this file, and the README
  STATUS banner were just authored by the user's other assistant. Treat them as the
  spec; update them only to keep them accurate.

## Work items (priority order — do P0 fully before P1)

### P0.1 — Real data loader  *(no GPU)*
- In `src/mythos/data/stream.py`, add a `RealTextStream` that streams a real corpus
  via HF `datasets` (default: `roneneldan/TinyStories`; config-selectable
  `HuggingFaceFW/fineweb-edu` sample), tokenizes with `tiktoken` (`gpt2`/`cl100k`),
  and yields contiguous `(x, y)` next-token blocks of `block_size`.
- Add `data.source: real|synthetic` + `data.dataset` + `data.tokenizer` to config
  (`src/mythos/config.py`, `configs/*.yaml`). Default real for nano/medium; keep
  `synthetic` ONLY as an explicitly-labeled test fixture.
- Build a held-out validation split that is **never** trained on. Compute true
  `val_bpb` in **bytes** (use the tokenizer's byte counts), not the current
  vocab-size hack in `bits_per_byte`.
- **Acceptance:** `mythos-train --config configs/nano.yaml --steps 50` on real data
  shows training loss decreasing meaningfully below `ln(vocab_size)`; val computed on
  held-out split.

### P0.2 — Real, checkpoint-conditioned eval  *(no GPU)*
- Rewrite `src/mythos/eval/harness.py` so `run_full_eval(model_path=...)` **loads the
  checkpoint** and evaluates THAT model: real held-out bpb, and real `lm-eval` tasks
  (gsm8k/hellaswag/arc) against the loaded model when `lm-eval` is installed.
- Delete `_proxy_lm_scores` as a *result source*. If `lm-eval` is missing, return the
  metric as `None` and have `composite.py` treat `None` as `unavailable` (excluded
  from the weighted sum with renormalization), never 0-faked-as-real.
- **Acceptance:** eval output changes when you swap checkpoints; deleting the
  checkpoint makes capability metrics `unavailable`, not a number.

### P0.3 — Anti-fake-win regression test  *(gate)*
- Add `tests/regression/test_no_fake_wins.py`:
  - a random-initialized tiny model scores ~chance on the real eval;
  - a 200-step-trained model scores strictly better on held-out bpb;
  - `run_full_eval` returns different values for two different checkpoints;
  - asserts no capability metric equals a `limit`-derived constant.
- Wire it into `.github/workflows/ci.yml`.

### P0.4 — Honest autoresearch objective
- Point `src/mythos/autoresearch.py` at real held-out bpb (+ optional real downstream)
  as the optimization signal. Add held-out contamination guard and require 3×
  replication before a "keep". Update `program.md` benchmark weights to the real,
  available metrics only.

### P1.1 — Real tiny pretrain run  *(MPS / 1 small GPU)*
- Train ~10–50M params on TinyStories to a real val loss. Commit `samples.txt`
  (generated text) and the real metrics to `results.tsv`. Document exact command+seed.
- **Acceptance:** coherent grammatical samples; held-out bpb beats a unigram baseline
  by a documented margin.

### P1.2 — Optimizer/attention truthfulness
- Verify `src/mythos/optim/muon.py` actually implements Muon/NorMuon (Newton–Schulz
  orthogonalization) and that attention uses `F.scaled_dot_product_attention`
  (Flash) instead of the manual softmax in `model/gpt.py`. Fix or rename honestly.

### P2 — Honest autoresearch loop
- Run the loop; record real keepers in `results.tsv`. Add a results dashboard that
  reads only real runs.

### P3 — Post-train + real agent eval  *(optional GPU)*
- SFT a small chat model. Integrate **mini-swe-agent** as a real eval scaffold on
  SWE-bench Lite (≥5 real tasks, real pass/fail). Low/zero pass is fine — report it.

### P4 — Serving + defensive safety
- Serve the real checkpoint via the FastAPI OpenAI-compatible API. Keep the dual-tier
  router as a documented **defensive** safety-classifier demo. Replace `agents/cyber`
  with the secure-coding comprehension eval; `agents/bio` → MMLU science.

## Reframe the security/bio agents (P0.2 / P4)
- `agents/cyber/` → `agents/secsec/` (or keep name, change semantics): tasks become
  "given this code/CVE text, identify the vulnerability class" — multiple-choice or
  short-answer **comprehension**, scored against the loaded model. No exploit code,
  no execution of attacks.
- `agents/bio/` → fold into MMLU science tasks via lm-eval. Delete bespoke bio scoring.
- Update `eval/composite.py` weights and all tests accordingly; keep them green.

## Commands you'll use
```bash
source .venv/bin/activate
pip install -e ".[all]"
python -m pytest tests/ -q                 # must stay green
mythos-train --config configs/nano.yaml --steps 50
mythos-eval --mode proxy --limit 5
mythos-autoresearch --budget-minutes 5 --max-experiments 1
```

## Definition of done (mirror of PLAN.md §5)
- [ ] No metric independent of the checkpoint; `test_no_fake_wins` green in CI.
- [ ] Trained small model emits coherent text and beats baseline bpb (committed proof).
- [ ] Autoresearch produces ≥1 replicated real improvement.
- [ ] README states honest scope; no "frontier"/offensive claims.
- [ ] No offensive cyber/bio code; `SECURITY.md` accurate.
- [ ] Every README/`results.tsv` number reproducible from seed + command.

## Reporting
When you finish a work item, state plainly what you ran, the real numbers you got,
and what's still stubbed. If something fails, say so with the output. Do not describe
a stub as if it works.
