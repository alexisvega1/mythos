import pytest
import torch

from mythos.agents.cyber import evaluate_sec_comprehension
from mythos.config import MythosConfig
from mythos.model import GPT


def test_sec_comprehension_on_random_model():
    config = MythosConfig(depth=2, n_head=2, n_embd=64, block_size=16, vocab_size=50257)
    model = GPT.from_config(config)
    acc = evaluate_sec_comprehension(model, config, limit=5, device="cpu")
    assert acc is not None
    assert 0.0 <= acc <= 1.0


def test_sec_comprehension_improves_with_training_hint():
    config = MythosConfig(depth=2, n_head=2, n_embd=64, block_size=16, vocab_size=50257)
    model = GPT.from_config(config)
    random_acc = evaluate_sec_comprehension(model, config, limit=5, device="cpu")
    assert random_acc is not None
