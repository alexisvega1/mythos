from __future__ import annotations

import random
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from mythos.config import MythosConfig


class SyntheticTokenStream(IterableDataset):
    """Deterministic synthetic token stream for smoke training and autoresearch."""

    def __init__(self, config: MythosConfig, seed: int = 42) -> None:
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.seed = seed
        self.mix = config.data.mix

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        rng = random.Random(self.seed)
        while True:
            x = torch.tensor(
                [rng.randrange(self.vocab_size) for _ in range(self.block_size)],
                dtype=torch.long,
            )
            y = torch.roll(x, shifts=-1, dims=0)
            yield x, y


def get_dataloader(config: MythosConfig, batch_size: int | None = None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Batch synthetic stream. Swap for FineWeb/HF datasets in production runs."""
    bs = batch_size or config.batch_size
    stream = SyntheticTokenStream(config)

    def batch_iter() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            xs, ys = [], []
            for _ in range(bs):
                x, y = next(iter(stream))
                xs.append(x)
                ys.append(y)
            yield torch.stack(xs), torch.stack(ys)

    return batch_iter


def bits_per_byte(loss_nats: float, vocab_size: int) -> float:
    """Validation bits per byte — autoresearch comparable metric."""
    import math

    return loss_nats / math.log(2) / math.log2(vocab_size) * math.log2(256)
