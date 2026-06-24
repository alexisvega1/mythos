"""Build assets for the Mythos visual demo.

Trains a small model on the Shakespeare corpus (recording the real bits/byte
curve), SFTs it on the instruction set, generates base-vs-SFT samples, and writes
everything to demo/assets/run.json. All numbers are real and reproducible.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from mythos.checkpoint import load_checkpoint, save_checkpoint
from mythos.config import MythosConfig
from mythos.data.stream import (
    RealTextStream,
    bits_per_byte,
    byte_weighted_bpb,
    get_batch_iterator,
    get_tokenizer,
    unigram_baseline_bpb,
)
from mythos.model import GPT
from mythos.optim import build_optimizer, lr_schedule, optimizer_step
from mythos.posttrain import SFT_TEMPLATE, run_sft

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
CKPT = ASSETS / "ckpt" / "latest.pt"
STEPS = 500


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _config() -> MythosConfig:
    cfg = MythosConfig.from_yaml(str(Path("configs/nano.yaml")))
    cfg.name = "mythos-demo"
    cfg.data.source = "fixture"
    cfg.data.fixture_path = "data/corpus/tinyshakespeare.txt"
    cfg.data.tokenizer = "gpt2"
    cfg.depth, cfg.n_embd, cfg.n_head, cfg.block_size = 4, 192, 6, 96
    cfg.batch_size, cfg.grad_accum, cfg.warmup_steps, cfg.learning_rate = 24, 1, 40, 6e-4
    cfg.eval.val_batches = 16
    return cfg


def _generate(model, enc, prompt: str, dev: str, n: int = 48, temp: float = 0.7) -> str:
    ids = enc.encode(prompt, disallowed_special=()) or [enc.eot_token]
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    out = model.generate(x, max_new_tokens=n, temperature=temp)
    gen = out[0].tolist()[len(ids):]
    segs, cur = [], []
    for t in gen:
        if t == enc.eot_token:
            if cur:
                segs.append(enc.decode(cur)); cur = []
        else:
            cur.append(t)
    if cur:
        segs.append(enc.decode(cur))
    return "\n".join(s for s in segs if s.strip()).strip()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build Mythos demo assets")
    parser.add_argument("--quick", action="store_true", help="80 train steps (CPU-friendly)")
    parser.add_argument("--skip-if-ready", action="store_true", help="Exit 0 if checkpoints exist")
    args = parser.parse_args()

    if args.skip_if_ready and CKPT.exists() and (ASSETS / "ckpt-sft" / "latest.pt").exists() and (ASSETS / "run.json").exists():
        print("demo assets ready — skip build")
        return

    steps = int(os.environ.get("MYTHOS_DEMO_STEPS", "80" if args.quick else str(STEPS)))
    _run(steps)


def _run(steps: int) -> None:
    cfg = _config()
    dev = _device()
    enc = get_tokenizer(cfg.data.tokenizer)
    cfg.sync_vocab_from_tokenizer(enc.n_vocab)
    print(f"device={dev} params-config depth={cfg.depth} d={cfg.n_embd} block={cfg.block_size}")

    model = GPT.from_config(cfg).to(dev)
    muon, adam = build_optimizer(model, cfg.optimizer, cfg.learning_rate, cfg.weight_decay)
    batches = get_batch_iterator(cfg, split="train")

    curve = []
    t0 = time.time()
    running = 0.0
    model.train()
    for step in range(steps):
        x, y = next(batches)
        x, y = x.to(dev), y.to(dev)
        _, loss = model(x, y)
        loss.backward()
        lr = lr_schedule(step, cfg.warmup_steps, steps, cfg.learning_rate)
        for opt in (muon, adam):
            for pg in opt.param_groups:
                pg["lr"] = lr
        optimizer_step(muon, adam)
        running += loss.item()
        if (step + 1) % 20 == 0:
            avg = running / 20
            curve.append({"step": step + 1, "bpb": round(bits_per_byte(avg), 4)})
            running = 0.0
    train_secs = time.time() - t0

    val_bpb = byte_weighted_bpb(
        model, get_batch_iterator(cfg, split="val", batch_size=4),
        cfg.data.tokenizer, dev, max_batches=16,
    )
    uni = unigram_baseline_bpb(cfg.data.tokenizer, RealTextStream(cfg, split="val").tokens[:8000])
    save_checkpoint(CKPT, model, cfg, steps, metrics={"val_bpb": val_bpb})
    base_sample = _generate(model, enc, "ROMEO:", dev)
    base_instruct = _generate(model, enc, SFT_TEMPLATE.format(instruction="What is the capital of France?"), dev, n=24)

    # SFT stage
    sft = run_sft(cfg, CKPT, steps=160, lr=1e-3, batch_size=8, device=dev, out_dir=ASSETS / "ckpt-sft")
    sft_model, sft_cfg, _ = load_checkpoint(sft["checkpoint"], device=dev)
    sft_instruct = _generate(sft_model, enc, SFT_TEMPLATE.format(instruction="What is the capital of France?"), dev, n=24, temp=0.5)

    run = {
        "model": {"params": model.count_parameters(), "depth": cfg.depth, "n_embd": cfg.n_embd,
                   "block_size": cfg.block_size, "vocab_size": cfg.vocab_size, "device": dev},
        "pretrain": {"steps": steps, "seconds": round(train_secs, 1), "curve": curve,
                      "val_bpb": round(val_bpb, 4), "unigram_baseline_bpb": round(uni, 4),
                      "beats_baseline": val_bpb < uni},
        "sft": {"loss_start": round(sft["sft_loss_start"], 3), "loss_end": round(sft["sft_loss_end"], 3),
                 "examples": sft["examples"], "steps": sft["steps"]},
        "samples": {"base_shakespeare": base_sample, "base_instruction": base_instruct,
                     "sft_instruction": sft_instruct},
        "checkpoint": str(sft["checkpoint"]),
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "run.json").write_text(json.dumps(run, indent=2))
    print(json.dumps({"val_bpb": run["pretrain"]["val_bpb"], "unigram": run["pretrain"]["unigram_baseline_bpb"],
                       "sft": [run["sft"]["loss_start"], run["sft"]["loss_end"]]}, indent=2))
    print("wrote", ASSETS / "run.json")


if __name__ == "__main__":
    main()
