"""Build assets for the Mythos visual demo.

Trains the demo model from configs/demo.yaml (~48M params on Shakespeare),
SFTs on the expanded instruction set, generates base-vs-SFT samples, and writes
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
from mythos.posttrain import SFT_TEMPLATE, load_sft_examples, run_sft
from mythos.serve.inference import MythosEngine

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
CKPT = ASSETS / "ckpt" / "latest.pt"
CONFIG = Path("configs/demo.yaml")
STEPS = 800
SFT_STEPS = 280
QUICK_STEPS = 120
QUICK_SFT = 100

_GEN_BASE = dict(max_tokens=56, temperature=0.85, top_k=40, top_p=0.95, repetition_penalty=1.05)
_GEN_SFT = dict(max_tokens=48, temperature=0.65, top_k=40, top_p=0.92, repetition_penalty=1.12)


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _config() -> MythosConfig:
    return MythosConfig.from_yaml(str(CONFIG))


def _sample(engine: MythosEngine, prompt: str, *, base: bool = False) -> str:
    kw = _GEN_BASE if base else _GEN_SFT
    text, _, _ = engine.generate(prompt, **kw)
    return text or "(empty)"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build Mythos demo assets")
    parser.add_argument("--quick", action="store_true", help="120 train steps (CPU-friendly)")
    parser.add_argument("--skip-if-ready", action="store_true", help="Exit 0 if checkpoints exist")
    args = parser.parse_args()

    if args.skip_if_ready and CKPT.exists() and (ASSETS / "ckpt-sft" / "latest.pt").exists() and (ASSETS / "run.json").exists():
        print("demo assets ready — skip build")
        return

    quick = args.quick
    steps = int(os.environ.get("MYTHOS_DEMO_STEPS", str(QUICK_STEPS if quick else STEPS)))
    sft_steps = int(os.environ.get("MYTHOS_DEMO_SFT_STEPS", str(QUICK_SFT if quick else SFT_STEPS)))
    _run(steps, sft_steps)


def _run(steps: int, sft_steps: int) -> None:
    cfg = _config()
    dev = _device()
    enc = get_tokenizer(cfg.data.tokenizer)
    cfg.sync_vocab_from_tokenizer(enc.n_vocab)
    n_sft = len(load_sft_examples())
    print(f"device={dev} config={CONFIG} params-target depth={cfg.depth} d={cfg.n_embd} sft_examples={n_sft}")

    model = GPT.from_config(cfg).to(dev)
    print(f"params={model.count_parameters():,}")
    muon, adam = build_optimizer(model, cfg.optimizer, cfg.learning_rate, cfg.weight_decay)
    batches = get_batch_iterator(cfg, split="train")

    curve = []
    t0 = time.time()
    running = 0.0
    log_every = max(20, steps // 25)
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
        if (step + 1) % log_every == 0:
            avg = running / log_every
            curve.append({"step": step + 1, "bpb": round(bits_per_byte(avg), 4)})
            running = 0.0
            print(f"step {step + 1}/{steps} bpb={curve[-1]['bpb']}")
    train_secs = time.time() - t0

    val_bpb = byte_weighted_bpb(
        model, get_batch_iterator(cfg, split="val", batch_size=4),
        cfg.data.tokenizer, dev, max_batches=cfg.eval.val_batches,
    )
    uni = unigram_baseline_bpb(cfg.data.tokenizer, RealTextStream(cfg, split="val").tokens[:8000])
    save_checkpoint(CKPT, model, cfg, steps, metrics={"val_bpb": val_bpb})

    base_engine = MythosEngine(CKPT, device=dev)
    base_sample = _sample(base_engine, "To be, or not to be", base=True)
    base_instruct = _sample(base_engine, SFT_TEMPLATE.format(instruction="What is the capital of France?"), base=True)

    sft = run_sft(
        cfg, CKPT, steps=sft_steps, lr=8e-4, batch_size=8,
        device=dev, out_dir=ASSETS / "ckpt-sft",
    )
    sft_engine = MythosEngine(sft["checkpoint"], device=dev)
    sft_instruct = _sample(sft_engine, SFT_TEMPLATE.format(instruction="What is the capital of France?"))
    sft_shakespeare = _sample(sft_engine, SFT_TEMPLATE.format(instruction="Continue in Shakespeare's style: O fair moon,"))

    run = {
        "model": {
            "params": model.count_parameters(), "depth": cfg.depth, "n_embd": cfg.n_embd,
            "block_size": cfg.block_size, "vocab_size": cfg.vocab_size, "device": dev,
            "config": str(CONFIG),
        },
        "pretrain": {
            "steps": steps, "seconds": round(train_secs, 1), "curve": curve,
            "val_bpb": round(val_bpb, 4), "unigram_baseline_bpb": round(uni, 4),
            "beats_baseline": val_bpb < uni,
        },
        "sft": {
            "loss_start": round(sft["sft_loss_start"], 3), "loss_end": round(sft["sft_loss_end"], 3),
            "examples": sft["examples"], "steps": sft["steps"], "dataset_examples": n_sft,
        },
        "samples": {
            "base_shakespeare": base_sample, "base_instruction": base_instruct,
            "sft_instruction": sft_instruct, "sft_shakespeare": sft_shakespeare,
        },
        "checkpoint": str(sft["checkpoint"]),
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "run.json").write_text(json.dumps(run, indent=2))
    print(json.dumps({
        "val_bpb": run["pretrain"]["val_bpb"], "unigram": run["pretrain"]["unigram_baseline_bpb"],
        "sft": [run["sft"]["loss_start"], run["sft"]["loss_end"]], "params": run["model"]["params"],
    }, indent=2))
    print("wrote", ASSETS / "run.json")


if __name__ == "__main__":
    main()
