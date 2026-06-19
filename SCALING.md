# Logos scaling-law demo — predict-then-verify

The headline of Project Logos: fit the Chinchilla law on a sweep of small models,
then **predict the held-out loss of a larger model *before* training it**, and
report predicted-vs-observed. The flex isn't parameter count — it's a numerical
commitment about an unseen model that comes true.

## Reproduce ($0, Apple Silicon / MPS, ~10 min)

```bash
mythos-sweep --config configs/shakespeare.yaml --device mps --seed 42
# -> eval/scaling/scaling_results.tsv  +  eval/scaling/scorecard.json
```

Everything is seeded; the same command reproduces the numbers below. The engine
itself is independently proven by a synthetic round-trip
(`tests/unit/test_scaling.py`): it recovers a *known* law and extrapolates via
leave-one-out within tolerance.

## The sweep (12 runs, seed 42)

`N` = non-embedding params (the Chinchilla quantity), `D` = tokens trained on,
loss = **minimum** held-out `val_bpb` over training.

| n_embd | depth | N | D | min val_bpb |
|---:|---:|---:|---:|---:|
| 32 | 2 | 24,800 | 200,448 | 3.1245 |
| 32 | 2 | 24,800 | 999,936 | 2.1090 |
| 48 | 3 | 83,424 | 200,448 | 2.9222 |
| 48 | 3 | 83,424 | 999,936 | 1.9608 |
| 64 | 3 | 148,000 | 999,936 | 1.8711 |
| 96 | 4 | 443,424 | 200,448 | 2.4912 |
| 96 | 4 | 443,424 | 999,936 | 1.7533 |

(full 12-row grid in `eval/scaling/scaling_results.tsv`)

Loss falls monotonically with both `N` and `D`. Fitted law
`L(N,D) = E + A/N^α + B/D^β`:

```
E = 0.027   A = 15.01   B = 124.0   α = 0.272   β = 0.332   (Huber residual 1.7e-4)
```

The exponents land near Chinchilla's published `α≈0.34, β≈0.37` — notable for a
toy run, though they will (and should) differ at this scale / single dataset /
NorMuon optimizer.

## The prediction (committed before training)

Held-out model: `n_embd 128, depth 5` → **984,768 non-embedding params (2.2× the
largest swept model)**, `D = 1,499,904` tokens (1.5× the largest swept budget) —
a genuine extrapolation, not interpolation.

```
PREDICTED  val_bpb 1.4752   (95% CI 1.4219 – 1.5827)     [from the fit, before training]
OBSERVED   val_bpb 1.5797                                 [trained once, seed 42]
relative error 6.6%   |   leave-one-out max error 7.0%   |   inside 95% CI ✓
```

**A model 2.2× larger than anything in the sweep was predicted within 6.6%, before it was trained.**

## Honest caveats (this is a toy)

- **Toy scale on a small corpus.** Tiny-Shakespeare is ~270k training tokens, so
  the larger budgets repeat data (1M tokens ≈ 3.7 epochs; the held-out 1.5M ≈ 5.5
  epochs). Repetition breaks the single-epoch scaling assumption, so the absolute
  numbers are illustrative, not publication-grade.
- **Observed sits at the top edge of the CI.** The extrapolation slightly
  *under*-predicted the loss — expected, because the held-out point has the most
  repetition. It's inside the band, but barely; don't over-read the CI.
- **`E ≈ 0.03`** (the irreducible-loss floor) is not physically meaningful at this
  scale; it's what the fit returns on repeated toy data.

The **methodology and pipeline are validated end-to-end**; the clean, honest
version is the same `mythos-sweep` run with a wider grid on FineWeb-Edu on a
rented GPU (Project Logos P2/P3), where tokens are not repeated.
