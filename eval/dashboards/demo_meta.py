"""Write demo metadata after a train/eval run for the live dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mythos.checkpoint import load_checkpoint


def write_demo_meta(
    checkpoint: str | Path,
    output: str | Path = "eval/dashboards/demo_meta.json",
    *,
    config_name: str = "",
    train_extra: dict[str, Any] | None = None,
    eval_extra: dict[str, Any] | None = None,
    api_port: int = 8765,
    samples_path: str | Path | None = None,
) -> Path:
    ckpt = Path(checkpoint)
    meta: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(ckpt.resolve()),
        "config_name": config_name,
        "api_port": api_port,
        "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    }
    if ckpt.exists():
        try:
            model, cfg, ckpt_meta = load_checkpoint(ckpt, device="cpu")
            m = ckpt_meta.get("metrics", {})
            meta.update(
                {
                    "config_name": cfg.name,
                    "params": model.count_parameters(),
                    "val_bpb": m.get("val_bpb"),
                    "train_bpb": m.get("train_bpb_approx"),
                    "steps": ckpt_meta.get("step"),
                    "data_source": cfg.data.source,
                }
            )
        except Exception as exc:
            meta["load_error"] = str(exc)
    if train_extra:
        meta["train"] = train_extra
    if eval_extra:
        meta["eval"] = eval_extra
    if samples_path and Path(samples_path).exists():
        meta["samples"] = Path(samples_path).read_text(encoding="utf-8")[:4000]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2))
    return out
