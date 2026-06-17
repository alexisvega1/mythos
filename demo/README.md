# Mythos — visual demo

See the small model **train, post-train, serve, and safety-route** — live, with real numbers.

```bash
pip install -e ".[all]"
python demo/build_demo.py     # trains + SFTs a ~21M model on the Shakespeare corpus (~1 min on MPS)
python demo/serve_demo.py     # then open http://127.0.0.1:8000
```

`build_demo.py` writes `demo/assets/run.json` (training curve, val bits/byte vs unigram
baseline, base-vs-SFT samples) and saves the base + SFT checkpoints under
`demo/assets/` (gitignored). `serve_demo.py` loads them and serves:

- **Live chat** with a base↔SFT toggle — watch instruction fine-tuning change the model
  (base rambles Shakespeare; SFT answers `"The capital of France is Paris."`).
- **Training curve** — real held-out bits/byte over steps; beats the unigram baseline.
- **Safety router** — a flagged prompt is routed to the Fable tier, not run through the model.

It's a nano-scale model, so output is rough by design — the point is that the **entire
pipeline is real and reproducible** (no fabricated metrics; see `../SECURITY.md`). The
demo server is standalone and reuses `mythos.serve.inference` + `mythos.router`.
