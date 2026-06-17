"""Real checkpoint-backed inference for the serving layer.

Loads a trained Mythos checkpoint and generates text. If no checkpoint is
available the engine is absent and the API reports `unavailable` rather than
fabricating a response (honesty invariant — see SECURITY.md).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from mythos.checkpoint import load_checkpoint
from mythos.data.stream import get_tokenizer


def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MythosEngine:
    """Wraps a loaded checkpoint + tokenizer for text generation."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = pick_device(device)
        self.model, self.config, self.meta = load_checkpoint(checkpoint_path, device=self.device)
        self.model.eval()
        self.tokenizer = get_tokenizer(self.config.data.tokenizer)
        self.checkpoint_path = str(checkpoint_path)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
    ) -> tuple[str, int, int]:
        """Return (completion_text, prompt_tokens, completion_tokens)."""
        enc = self.tokenizer
        ids = enc.encode(prompt, disallowed_special=()) or [enc.eot_token]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            x, max_new_tokens=max(1, max_tokens), temperature=max(temperature, 1e-6)
        )
        gen_ids = out[0].tolist()[len(ids):]
        # Treat end-of-text as a line break (this base model uses it as a line
        # separator) rather than hard-stopping, so completions are not empty.
        segments, cur = [], []
        for tok in gen_ids:
            if tok == enc.eot_token:
                if cur:
                    segments.append(enc.decode(cur))
                    cur = []
            else:
                cur.append(tok)
        if cur:
            segments.append(enc.decode(cur))
        text = "\n".join(s for s in segments if s.strip()).strip()
        return text, len(ids), len(gen_ids)


def load_engine(
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> MythosEngine | None:
    """Load an engine from an explicit path or $MYTHOS_CHECKPOINT; None if absent."""
    path = checkpoint_path or os.environ.get("MYTHOS_CHECKPOINT")
    if not path or not Path(path).exists():
        return None
    try:
        return MythosEngine(path, device=device)
    except Exception:
        return None
