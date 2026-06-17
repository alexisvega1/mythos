"""
Post-training: real instruction SFT on top of a pretrained checkpoint.

SFT fine-tunes the pretrained model on (instruction, response) pairs with
response-only loss masking (prompt tokens are set to -100 so the model is trained
to *produce* responses, not to repeat prompts). RFT/GRPO over agentic environments
remain honest stubs (`status: unavailable`) until a real env + oracle is wired —
no fabricated rewards (see SECURITY.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mythos.checkpoint import load_checkpoint, save_checkpoint
from mythos.config import MythosConfig
from mythos.data.stream import get_tokenizer
from mythos.optim import build_optimizer, optimizer_step
from mythos.paths import repo_root

SFT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
IGNORE_INDEX = -100
DEFAULT_SFT_DATA = "data/fixtures/sft_instructions.jsonl"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def load_sft_examples(path: str | Path = DEFAULT_SFT_DATA) -> list[dict]:
    examples: list[dict] = []
    for line in _resolve(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("instruction") and obj.get("response") is not None:
            examples.append(obj)
    return examples


def _encode_example(enc, ex: dict, block_size: int) -> tuple[list[int], list[int]]:
    """Return (x, y) for one example with the prompt region masked to IGNORE_INDEX."""
    prompt_ids = enc.encode(SFT_TEMPLATE.format(instruction=ex["instruction"]), disallowed_special=())
    resp_ids = enc.encode(ex["response"], disallowed_special=()) + [enc.eot_token]
    ids = (prompt_ids + resp_ids)[:block_size]
    x, y = ids[:-1], ids[1:]
    # y[i] predicts ids[i+1]; only supervise response tokens (i+1 >= len(prompt_ids)).
    y = [tok if (i + 1) >= len(prompt_ids) else IGNORE_INDEX for i, tok in enumerate(y)]
    return x, y


def build_sft_dataset(examples: list[dict], enc, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    width = block_size - 1
    for ex in examples:
        x, y = _encode_example(enc, ex, block_size)
        if not x or all(t == IGNORE_INDEX for t in y):
            continue  # skip examples with no supervised response token
        pad = width - len(x)
        if pad > 0:
            x = x + [enc.eot_token] * pad
            y = y + [IGNORE_INDEX] * pad
        xs.append(x[:width])
        ys.append(y[:width])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def run_sft(
    config: MythosConfig,
    checkpoint: str | Path,
    steps: int = 80,
    lr: float = 1e-3,
    batch_size: int = 8,
    sft_data: str | Path = DEFAULT_SFT_DATA,
    device: str | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Fine-tune a pretrained checkpoint on instruction data. Returns real metrics."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_cfg, meta = load_checkpoint(checkpoint, device=dev)
    enc = get_tokenizer(ckpt_cfg.data.tokenizer)

    examples = load_sft_examples(sft_data)
    X, Y = build_sft_dataset(examples, enc, ckpt_cfg.block_size)
    if X.numel() == 0:
        return {"stage": "sft", "status": "no_data", "checkpoint": str(checkpoint)}
    X, Y = X.to(dev), Y.to(dev)
    n = X.size(0)

    muon, adam = build_optimizer(model, ckpt_cfg.optimizer, lr, ckpt_cfg.weight_decay)
    gen = torch.Generator().manual_seed(0)

    model.train()
    loss_start: float | None = None
    loss_end = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=gen).to(dev)
        _, loss = model(X[idx], Y[idx])
        loss.backward()
        optimizer_step(muon, adam)
        loss_end = loss.item()
        if loss_start is None:
            loss_start = loss_end

    out = Path(out_dir) if out_dir else repo_root() / "checkpoints" / f"{ckpt_cfg.name}-sft"
    sft_ckpt = out / "latest.pt"
    metrics = {
        "stage": "sft",
        "status": "trained",
        "checkpoint": str(sft_ckpt),
        "base_checkpoint": str(checkpoint),
        "steps": steps,
        "examples": n,
        "sft_loss_start": loss_start,
        "sft_loss_end": loss_end,
    }
    save_checkpoint(sft_ckpt, model, ckpt_cfg, meta.get("step", 0), metrics=metrics)
    return metrics


def run_grpo(config: MythosConfig, checkpoint: Path, env: str = "swe") -> dict:
    """GRPO post-training entry — delegates to domain modules (honest stubs)."""
    if env == "swe":
        from mythos.agents import swe as swe_agent

        return swe_agent.run_swe_grpo(config, checkpoint)
    if env == "cyber":
        from mythos.agents import cyber as cyber_agent

        return cyber_agent.run_cyber_grpo(config, checkpoint)
    raise ValueError(f"Unknown env: {env}")


def run_rft(config: MythosConfig, checkpoint: Path) -> dict:
    from mythos.agents import swe as swe_agent

    return swe_agent.run_swe_rft(config, checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mythos post-training")
    parser.add_argument("--config", default="configs/nano.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", choices=["sft", "rft", "grpo"], default="sft")
    parser.add_argument("--env", default="swe")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sft-data", default=DEFAULT_SFT_DATA)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config = MythosConfig.from_yaml(args.config)
    ckpt = Path(args.checkpoint)
    if args.stage == "sft":
        result = run_sft(
            config, ckpt, steps=args.steps, lr=args.lr,
            sft_data=args.sft_data, device=args.device, out_dir=args.out,
        )
    elif args.stage == "rft":
        result = run_rft(config, ckpt)
    else:
        result = run_grpo(config, ckpt, env=args.env)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
