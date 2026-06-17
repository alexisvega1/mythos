from __future__ import annotations

import torch

from mythos.config import MythosConfig
from mythos.model import GPT


def test_gpt_forward_shape():
    config = MythosConfig(depth=4, n_head=4, n_embd=128, block_size=32, vocab_size=256)
    model = GPT.from_config(config)
    x = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 256)
    assert loss is not None
    assert torch.isfinite(loss)


def test_gpt_parameter_count():
    config = MythosConfig(depth=2, n_head=2, n_embd=64, vocab_size=128, block_size=16)
    model = GPT.from_config(config)
    assert model.count_parameters() > 0


def test_generate():
    config = MythosConfig(depth=2, n_head=2, n_embd=64, vocab_size=128, block_size=16)
    model = GPT.from_config(config)
    idx = torch.randint(0, 128, (1, 4))
    out = model.generate(idx, max_new_tokens=3)
    assert out.shape == (1, 7)
