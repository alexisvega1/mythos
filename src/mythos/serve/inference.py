"""Real checkpoint-backed inference for the serving layer.

Loads a trained Mythos checkpoint and generates text. If no checkpoint is
available the engine is absent and the API reports `unavailable` rather than
fabricating a response (honesty invariant — see SECURITY.md).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F

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


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    penalty: float | None,
) -> torch.Tensor:
    if penalty is None or penalty <= 1.0 or not token_ids:
        return logits
    for tid in set(token_ids):
        logits[0, tid] /= penalty
    return logits


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < thresh, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return logits


def _decode_gen_ids(enc, gen_ids: list[int]) -> str:
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
    return "\n".join(s for s in segments if s.strip()).strip()


class MythosEngine:
    """Wraps a loaded checkpoint + tokenizer for text generation."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = pick_device(device)
        self.model, self.config, self.meta = load_checkpoint(checkpoint_path, device=self.device)
        self.model.eval()
        self.tokenizer = get_tokenizer(self.config.data.tokenizer)
        self.checkpoint_path = str(checkpoint_path)

    @torch.no_grad()
    def _sample_ids(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        temperature: float,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> list[int]:
        enc = self.tokenizer
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        gen: list[int] = []
        temp = max(temperature, 1e-6)
        for _ in range(max(1, max_tokens)):
            idx_cond = idx[:, -self.model.config.block_size :]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / temp
            logits = _apply_repetition_penalty(logits, gen[-48:], repetition_penalty)
            logits = _apply_top_k_top_p(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())
            gen.append(next_token)
            if next_token == enc.eot_token:
                break
            idx = torch.cat((idx, torch.tensor([[next_token]], device=self.device)), dim=1)
        return gen

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> tuple[str, int, int]:
        """Return (completion_text, prompt_tokens, completion_tokens)."""
        enc = self.tokenizer
        ids = enc.encode(prompt, disallowed_special=()) or [enc.eot_token]
        gen_ids = self._sample_ids(
            ids, max_tokens, temperature, top_k, top_p, repetition_penalty,
        )
        text = _decode_gen_ids(enc, gen_ids)
        return text, len(ids), len(gen_ids)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> Iterator[tuple[str, int, int]]:
        """Yield (token_text, prompt_tokens, completion_index) per generated token."""
        enc = self.tokenizer
        ids = enc.encode(prompt, disallowed_special=()) or [enc.eot_token]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        temp = max(temperature, 1e-6)
        prompt_len = len(ids)
        gen: list[int] = []
        for i in range(max(1, max_tokens)):
            idx_cond = idx[:, -self.model.config.block_size :]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / temp
            logits = _apply_repetition_penalty(logits, gen[-48:], repetition_penalty)
            logits = _apply_top_k_top_p(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())
            gen.append(next_token)
            idx = torch.cat((idx, torch.tensor([[next_token]], device=self.device)), dim=1)
            if next_token == enc.eot_token:
                yield "\n", prompt_len, i + 1
                break
            piece = enc.decode([next_token])
            yield piece, prompt_len, i + 1


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
