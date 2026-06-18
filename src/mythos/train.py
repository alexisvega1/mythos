"""
Primary training entry point — the ONLY file AutoMythos mutates for pretrain experiments.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mythos.checkpoint import latest_checkpoint_for, save_checkpoint, write_metrics
from mythos.config import MythosConfig
from mythos.data.stream import bits_per_byte, byte_weighted_bpb, get_batch_iterator, get_tokenizer
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

    if config.data.source != "synthetic":
        enc = get_tokenizer(config.data.tokenizer)
        config.sync_vocab_from_tokenizer(enc.n_vocab)

    model = GPT.from_config(config).to(dev)
    muon, adam = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    train_batches = get_batch_iterator(config, split="train")

    max_steps = steps or config.max_steps
    budget = budget_seconds or config.train_budget_seconds
    start = time.time()

    running_loss = 0.0
    step = 0
    train_bpb = float("inf")

    model.train()
    while step < max_steps and (time.time() - start) < budget:
        for _ in range(config.grad_accum):
            x, y = next(train_batches)
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
            train_bpb = bits_per_byte(avg_loss)
            running_loss = 0.0

    val_batches = get_batch_iterator(config, split="val", batch_size=min(config.batch_size, 4))
    val_bpb = byte_weighted_bpb(
        model,
        val_batches,
        config.data.tokenizer,
        dev,
        max_batches=config.eval.val_batches,
    )

    out_dir = checkpoint_dir or Path("checkpoints") / config.name
    ckpt_path = out_dir / "latest.pt"
    save_checkpoint(ckpt_path, model, config, step)

    samples_path = out_dir / "samples.txt"
    try:
        enc = get_tokenizer(config.data.tokenizer)
        model.eval()
        prompt = "To be, or not to be"
        ids = enc.encode(prompt, disallowed_special=()) or [enc.eot_token]
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            out_ids = model.generate(
                x, max_new_tokens=min(64, config.block_size), temperature=0.8,
            )
        gen = out_ids[0].tolist()[len(ids):]
        segments, cur = [], []
        for tok in gen:
            if tok == enc.eot_token:
                if cur:
                    segments.append(enc.decode(cur))
                    cur = []
            else:
                cur.append(tok)
        if cur:
            segments.append(enc.decode(cur))
        text = "\n".join(s for s in segments if s.strip()).strip()
        samples_path.write_text(f"# prompt: {prompt}\n{text}\n", encoding="utf-8")
    except Exception:
        pass

    metrics = {
        "steps": step,
        "train_bpb_approx": train_bpb,
        "val_bpb": val_bpb,
        "elapsed_seconds": time.time() - start,
        "params": model.count_parameters(),
        "checkpoint": str(ckpt_path),
        "data_source": config.data.source,
    }
    write_metrics(config, metrics)
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
