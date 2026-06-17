from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from mythos.config import MythosConfig
from mythos.model import GPT


def save_checkpoint(
    path: Path,
    model: GPT,
    config: MythosConfig,
    step: int,
    metrics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[GPT, MythosConfig, dict[str, Any]]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = MythosConfig.from_dict(payload["config"])
    model = GPT.from_config(config).to(device)
    model.load_state_dict(payload["model"])
    meta = {"step": payload.get("step", 0), "metrics": payload.get("metrics", {})}
    return model, config, meta


def metrics_path_for(config: MythosConfig) -> Path:
    return Path("checkpoints") / config.name / "metrics.json"


def latest_checkpoint_for(config: MythosConfig) -> Path:
    return Path("checkpoints") / config.name / "latest.pt"


def write_metrics(config: MythosConfig, metrics: dict[str, Any]) -> None:
    path = metrics_path_for(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
