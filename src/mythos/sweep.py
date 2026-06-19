"""Scaling-law sweep runner — generate real (N, D, loss) points and predict-then-verify.

The P2/P3 headline of Project Logos. Train a grid of small models at varying
width/depth (N = non-embedding params) and token budgets (D), record each run's
MINIMUM held-out val_bpb, fit the Chinchilla law (``mythos.scaling``), then
PREDICT a held-out larger model's loss *before* training it and report
predicted-vs-observed.

Everything is reproducible from a seed. Toy-scale runs work on Apple Silicon
(MPS) at $0; the same code scales to a rented GPU by widening the grid.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from mythos.config import MythosConfig
from mythos.model import GPT
from mythos.scaling import (
    ScalePoint,
    fit_scaling_law,
    max_loo_rel_error,
    predict_with_ci,
)
from mythos.train import train_run


@dataclass(frozen=True)
class SizeSpec:
    n_embd: int
    depth: int
    n_head: int

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd {self.n_embd} not divisible by n_head {self.n_head}")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head_dim must be even (RoPE rotates adjacent pairs)")


def tokens_per_step(cfg: MythosConfig) -> int:
    return cfg.batch_size * cfg.block_size * cfg.grad_accum


def steps_for_tokens(cfg: MythosConfig, tokens: int) -> int:
    return max(1, round(tokens / tokens_per_step(cfg)))


def config_for(base: MythosConfig, size: SizeSpec, name: str) -> MythosConfig:
    cfg = copy.deepcopy(base)
    cfg.n_embd, cfg.depth, cfg.n_head, cfg.name = size.n_embd, size.depth, size.n_head, name
    return cfg


def _non_embedding_N(cfg: MythosConfig) -> int:
    """Non-embedding param count for a config, without training (for prediction)."""
    synced = copy.deepcopy(cfg)
    if synced.data.source != "synthetic":
        from mythos.data.stream import get_tokenizer
        synced.sync_vocab_from_tokenizer(get_tokenizer(synced.data.tokenizer).n_vocab)
    return GPT.from_config(synced).count_non_embedding_parameters()


def run_point(
    base: MythosConfig,
    size: SizeSpec,
    tokens: int,
    seed: int,
    device: str | None,
    ckpt_root: Path,
    label: str,
) -> tuple[ScalePoint, dict]:
    cfg = config_for(base, size, label)
    steps = steps_for_tokens(cfg, tokens)
    metrics = train_run(cfg, steps=steps, seed=seed, device=device, checkpoint_dir=ckpt_root / label)
    D = steps * tokens_per_step(cfg)
    point = ScalePoint(N=metrics["non_embedding_params"], D=D, loss=metrics["min_val_bpb"])
    row = {
        "label": label, "n_embd": size.n_embd, "depth": size.depth, "n_head": size.n_head,
        "N": point.N, "D": point.D, "min_val_bpb": round(point.loss, 6),
        "steps": steps, "seed": seed,
    }
    return point, row


def run_sweep(
    base: MythosConfig,
    sizes: Sequence[SizeSpec],
    token_budgets: Sequence[int],
    seed: int = 42,
    device: str | None = None,
    ckpt_root: Path | None = None,
    tsv_path: Path | None = None,
    log: Callable[[str], None] = print,
) -> tuple[list[ScalePoint], list[dict]]:
    ckpt_root = ckpt_root or Path("checkpoints/sweep")
    points, rows = [], []
    for size in sizes:
        for tokens in token_budgets:
            label = f"sweep-e{size.n_embd}-d{size.depth}-t{int(tokens)}"
            point, row = run_point(base, size, tokens, seed, device, ckpt_root, label)
            points.append(point)
            rows.append(row)
            log(f"  {label:<26} N={point.N:>9,}  D={point.D:>10,}  min_val_bpb={point.loss:.4f}")
    if tsv_path:
        write_tsv(tsv_path, rows)
    return points, rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["label", "n_embd", "depth", "n_head", "N", "D", "min_val_bpb", "steps", "seed"]
    lines = ["\t".join(header)]
    lines += ["\t".join(str(r[h]) for h in header) for r in rows]
    path.write_text("\n".join(lines) + "\n")


@dataclass
class PredictThenVerify:
    heldout_N: int
    heldout_D: int
    predicted: float
    ci_lo: float
    ci_hi: float
    observed: float
    rel_error: float
    loo_max_rel_error: float
    inside_ci: bool
    law: dict
    n_points: int

    def scorecard(self) -> str:
        ok = "PASS" if self.inside_ci else "OUTSIDE CI"
        return (
            f"Logos predicts the unseen {self.heldout_N:,}-param model (D={self.heldout_D:,}) "
            f"-> val_bpb {self.predicted:.4f}  (95% CI {self.ci_lo:.4f}-{self.ci_hi:.4f})\n"
            f"OBSERVED {self.observed:.4f}  |  relative error {self.rel_error*100:.1f}%  "
            f"|  leave-one-out max error {self.loo_max_rel_error*100:.1f}%  |  CI: {ok}"
        )


def predict_then_verify(
    base: MythosConfig,
    sizes: Sequence[SizeSpec],
    token_budgets: Sequence[int],
    heldout_size: SizeSpec,
    heldout_tokens: int,
    seed: int = 42,
    device: str | None = None,
    ckpt_root: Path | None = None,
    tsv_path: Path | None = None,
    n_boot: int = 100,
    log: Callable[[str], None] = print,
) -> PredictThenVerify:
    ckpt_root = ckpt_root or Path("checkpoints/sweep")

    log("Sweep (fitting points):")
    points, _ = run_sweep(base, sizes, token_budgets, seed, device, ckpt_root, tsv_path, log)

    law = fit_scaling_law(points)
    loo = max_loo_rel_error(points)

    # PREDICT before training: held-out N comes from param count (no training needed).
    held_cfg = config_for(base, heldout_size, "heldout")
    N_held = _non_embedding_N(held_cfg)
    D_held = steps_for_tokens(held_cfg, heldout_tokens) * tokens_per_step(held_cfg)
    predicted, lo, hi = predict_with_ci(points, N_held, D_held, n_boot=n_boot, seed=seed)
    log("")
    log(f"Fitted law: E={law.E:.3f} A={law.A:.2f} B={law.B:.2f} alpha={law.alpha:.3f} beta={law.beta:.3f}")
    log(f"PREDICTION (committed before training): val_bpb {predicted:.4f} (95% CI {lo:.4f}-{hi:.4f})")

    # VERIFY: train the held-out model once and compare.
    metrics = train_run(
        held_cfg, steps=steps_for_tokens(held_cfg, heldout_tokens),
        seed=seed, device=device, checkpoint_dir=ckpt_root / "heldout",
    )
    observed = metrics["min_val_bpb"]
    result = PredictThenVerify(
        heldout_N=N_held, heldout_D=D_held,
        predicted=predicted, ci_lo=lo, ci_hi=hi, observed=observed,
        rel_error=abs(predicted - observed) / observed,
        loo_max_rel_error=loo, inside_ci=(lo <= observed <= hi),
        law=law.as_dict(), n_points=len(points),
    )
    log("")
    log(result.scorecard())
    return result


# Toy defaults: a real end-to-end predict-then-verify on Apple Silicon at $0.
_DEMO_SIZES = (
    SizeSpec(32, 2, 2),
    SizeSpec(48, 3, 2),
    SizeSpec(64, 3, 4),
    SizeSpec(96, 4, 4),
)
_DEMO_BUDGETS = (200_000, 500_000, 1_000_000)
_DEMO_HELDOUT = SizeSpec(128, 5, 4)
_DEMO_HELDOUT_TOKENS = 1_500_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Logos scaling sweep + predict-then-verify")
    parser.add_argument("--config", default="configs/shakespeare.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval/scaling")
    args = parser.parse_args()

    base = MythosConfig.from_yaml(args.config)
    out = Path(args.out)
    result = predict_then_verify(
        base, _DEMO_SIZES, _DEMO_BUDGETS, _DEMO_HELDOUT, _DEMO_HELDOUT_TOKENS,
        seed=args.seed, device=args.device,
        ckpt_root=Path("checkpoints/sweep"), tsv_path=out / "scaling_results.tsv",
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "scorecard.json").write_text(json.dumps(result.__dict__, indent=2))
    print(f"\nWrote {out/'scaling_results.tsv'} and {out/'scorecard.json'}")


if __name__ == "__main__":
    main()
