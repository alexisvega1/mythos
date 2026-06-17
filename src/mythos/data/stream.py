from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterator
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from mythos.config import MythosConfig
from mythos.paths import repo_root


def get_tokenizer(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def _is_val_block(block_idx: int, holdout_pct: int) -> bool:
    if holdout_pct <= 0:
        return False
    return block_idx % max(100 // holdout_pct, 1) == 0


def _load_text_documents(config: MythosConfig) -> list[str]:
    if config.data.source == "fixture":
        path = repo_root() / config.data.fixture_path
        text = path.read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]

    if config.data.source == "synthetic":
        return [" ".join(f"token{i}" for i in range(64)) for _ in range(32)]

    try:
        from datasets import load_dataset

        ds = load_dataset(config.data.dataset, split=config.data.dataset_split, streaming=True)
        docs: list[str] = []
        for row in ds:
            text = row.get("text") or row.get("content") or ""
            if text.strip():
                docs.append(text.strip())
            if len(docs) >= 512:
                break
        if docs:
            return docs
    except Exception:
        pass

    path = repo_root() / config.data.fixture_path
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _tokenize_corpus(config: MythosConfig) -> tuple[list[list[int]], list[list[int]]]:
    enc = get_tokenizer(config.data.tokenizer)
    config.sync_vocab_from_tokenizer(enc.n_vocab)

    docs = _load_text_documents(config)
    rng = random.Random(config.data.seed)
    rng.shuffle(docs)

    train_tokens: list[int] = []
    val_tokens: list[int] = []
    for doc_idx, doc in enumerate(docs):
        tokens = enc.encode(doc)
        tokens.append(enc.eot_token)
        bucket = val_tokens if doc_idx % max(100 // max(config.data.val_holdout_pct, 1), 1) == 0 else train_tokens
        bucket.extend(tokens)

    if not train_tokens:
        all_tokens = enc.encode("\n".join(docs))
        split = max(len(all_tokens) // 10, 1)
        val_tokens = all_tokens[:split]
        train_tokens = all_tokens[split:] or all_tokens

    return [train_tokens], [val_tokens] if val_tokens else [train_tokens[-max(len(train_tokens) // 10, config.block_size) :]]


class RealTextStream:
    """Token blocks from real text with a held-out validation split."""

    def __init__(self, config: MythosConfig, split: str = "train") -> None:
        self.config = config
        self.split = split
        train_docs, val_docs = _tokenize_corpus(config)
        self.tokens = val_docs[0] if split == "val" else train_docs[0]
        self.block_size = config.block_size
        self._cursor = 0

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        need = self.block_size + 1
        toks = self.tokens
        # If a split is shorter than one block, cycle it so we form a full block
        # instead of spinning forever (a short val split would otherwise hang).
        if 0 < len(toks) < need:
            toks = toks * (need // len(toks) + 1)
        while toks:
            if self._cursor + need > len(toks):
                self._cursor = 0
            chunk = toks[self._cursor : self._cursor + need]
            self._cursor += self.block_size
            if len(chunk) < need:
                self._cursor = 0
                continue
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            yield x, y


class SyntheticTokenStream:
    """Explicit test-only random stream — never default for training."""

    def __init__(self, config: MythosConfig, seed: int = 42) -> None:
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        rng = random.Random(self.seed)
        while True:
            x = torch.tensor(
                [rng.randrange(self.vocab_size) for _ in range(self.block_size)],
                dtype=torch.long,
            )
            y = torch.roll(x, shifts=-1, dims=0)
            yield x, y


def get_stream(config: MythosConfig, split: str = "train"):
    if config.data.source == "synthetic":
        if split == "val":
            stream = SyntheticTokenStream(config, seed=config.data.seed + 1)
        else:
            stream = SyntheticTokenStream(config, seed=config.data.seed)
    else:
        stream = RealTextStream(config, split=split)
    return stream


def get_batch_iterator(
    config: MythosConfig,
    split: str = "train",
    batch_size: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    bs = batch_size or config.batch_size
    stream = get_stream(config, split=split)
    data_iter = iter(stream)

    while True:
        xs, ys = [], []
        for _ in range(bs):
            x, y = next(data_iter)
            xs.append(x)
            ys.append(y)
        yield torch.stack(xs), torch.stack(ys)


def get_dataloader(config: MythosConfig, split: str = "train", batch_size: int | None = None):
    """Back-compat alias."""
    return lambda: get_batch_iterator(config, split=split, batch_size=batch_size)


def byte_weighted_bpb(
    model: torch.nn.Module,
    batches: Iterator[tuple[torch.Tensor, torch.Tensor]],
    tokenizer_name: str,
    device: str | torch.device,
    max_batches: int = 20,
) -> float:
    """Held-out bits per byte using token byte lengths (not vocab-size hack)."""
    enc = get_tokenizer(tokenizer_name)
    model.eval()
    total_nats = 0.0
    total_bytes = 0.0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(batches):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            log_probs = F.log_softmax(logits, dim=-1)
            token_nats = -log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)

            for row in range(y.size(0)):
                for col in range(y.size(1)):
                    token_id = int(y[row, col].item())
                    nbytes = len(enc.decode_single_token_bytes(token_id))
                    total_nats += float(token_nats[row, col].item())
                    total_bytes += max(nbytes, 1)

    if total_bytes <= 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)


def bits_per_byte(loss_nats: float, avg_bytes_per_token: float = 4.0) -> float:
    """Approximate bpb from mean token loss (training log only)."""
    return loss_nats / (math.log(2) * max(avg_bytes_per_token, 1.0))


def unigram_baseline_bpb(tokenizer_name: str, tokens: list[int]) -> float:
    enc = get_tokenizer(tokenizer_name)
    if not tokens:
        return float("inf")
    counts: dict[int, int] = {}
    byte_total = 0.0
    for tid in tokens:
        counts[tid] = counts.get(tid, 0) + 1
        byte_total += len(enc.decode_single_token_bytes(tid))
    n = len(tokens)
    nats = 0.0
    for tid in tokens:
        p = counts[tid] / n
        nats += -math.log(max(p, 1e-12))
    return nats / (math.log(2) * max(byte_total, 1.0))
