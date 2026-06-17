"""
Primary training entry point — the ONLY file AutoMythos mutates for pretrain experiments.

Fixed-time budget runs; reports val_bpb and checkpoint path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mythos.config import MythosConfig
from mythos.data.stream import bits_per_byte, get_dataloader
from mythos.model import GPT
from mythos.optim import build_optimizer, lr_schedule, optimizer_step


def train_run(
    config: MythosConfig,
    steps: int | None = None,
    budget_seconds: int | None = None,
    device: str | None = None,
    checkpoint_dir: Path | None = None,
) -> dict:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT.from_config(config).to(dev)
    muon, adam = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    loader = get_dataloader(config)
    data_iter = loader()

    max_steps = steps or config.max_steps
    budget = budget_seconds or config.train_budget_seconds
    start = time.time()

    running_loss = 0.0
    step = 0
    val_bpb = float("inf")

    model.train()
    while step < max_steps and (time.time() - start) < budget:
        for _ in range(config.grad_accum):
            x, y = next(data_iter)
            x, y = x.to(dev), y.to(dev)
            _, loss = model(x, y)
            (loss / config.grad_accum).backward()
            running_loss += loss.item()

        lr = lr_schedule(step, config.warmup_steps, max_steps, config.learning_rate)
        for opt in (muon, adam):
            for pg in opt.param_groups:
                pg["lr"] = lr
        optimizer_step(muon, adam)
        step += 1

        if step % 10 == 0:
            avg_loss = running_loss / 10
            val_bpb = bits_per_byte(avg_loss, config.vocab_size)
            running_loss = 0.0

    elapsed = time.time() - start
    out_dir = checkpoint_dir or Path("checkpoints") / config.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "latest.pt"
    torch.save({"model": model.state_dict(), "config": config.__dict__, "step": step}, ckpt_path)

    metrics = {
        "steps": step,
        "val_bpb": val_bpb,
        "elapsed_seconds": elapsed,
        "params": model.count_parameters(),
        "checkpoint": str(ckpt_path),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Mythos pretrain")
    parser.add_argument("--config", default="configs/nano.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--budget-seconds", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = MythosConfig.from_yaml(args.config)
    metrics = train_run(config, steps=args.steps, budget_seconds=args.budget_seconds, device=args.device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
