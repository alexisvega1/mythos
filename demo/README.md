# Mythos — visual demo

See the small model **train, post-train, serve, and safety-route** — live, with real numbers.

## One command

```bash
make demo
# equivalent: bash scripts/run-demo.sh
# equivalent: pip install -e . && mythos-demo
```

This bootstraps `.venv`, installs core deps (no lm-eval), skips retrain when checkpoints
exist, picks a free port, and opens the browser.

```bash
mythos-demo --rebuild    # force retrain (~1 min on Apple Silicon)
make demo-quick          # 80-step CPU-friendly build
```

## What you get

- **Token streaming** — watch completions arrive token-by-token
- **Duel mode** — base vs SFT on the same prompt, side by side
- **Training curve** — real held-out bits/byte; beats unigram baseline
- **Safety router** — flagged prompts hit the Fable tier, not the model
- **Pipeline timeline** — pretrain → SFT → serve → safety, all measured

`build_demo.py` writes `demo/assets/run.json` and checkpoints under `demo/assets/` (gitignored).
`demo/assets/run.json` is also committed so the chart works before a local build.

## Manual

```bash
python demo/build_demo.py --skip-if-ready
python demo/serve_demo.py --device mps
```

Nano-scale by design — the **pipeline** is the product. See `../SECURITY.md`.
