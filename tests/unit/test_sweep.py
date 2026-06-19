"""Tests for the scaling sweep runner's non-training logic.

The end-to-end predict-then-verify is exercised by the demo run (it trains
models); here we lock the cheap, deterministic pieces: token/step math, config
templating, size validation, non-embedding param counting, and TSV output.
"""
from __future__ import annotations

import pytest

from mythos.config import MythosConfig
from mythos.sweep import (
    SizeSpec,
    _non_embedding_N,
    config_for,
    steps_for_tokens,
    tokens_per_step,
    write_tsv,
)


@pytest.fixture
def base() -> MythosConfig:
    return MythosConfig.from_yaml("configs/test.yaml")


def test_sizespec_validates():
    SizeSpec(64, 4, 4)  # ok: head_dim 16
    with pytest.raises(ValueError):
        SizeSpec(64, 4, 6)  # 64 % 6 != 0
    with pytest.raises(ValueError):
        SizeSpec(48, 4, 16)  # head_dim 3 is odd -> breaks RoPE pairing


def test_token_step_math(base):
    tps = tokens_per_step(base)
    assert tps == base.batch_size * base.block_size * base.grad_accum
    assert steps_for_tokens(base, 10 * tps) == 10
    assert steps_for_tokens(base, 0) == 1  # never zero steps


def test_config_for_applies_size(base):
    cfg = config_for(base, SizeSpec(96, 5, 4), "x")
    assert (cfg.n_embd, cfg.depth, cfg.n_head, cfg.name) == (96, 5, 4, "x")
    assert base.n_embd != 96  # base untouched (deep-copied)


def test_non_embedding_N_scales_with_size(base):
    small = _non_embedding_N(config_for(base, SizeSpec(32, 2, 2), "s"))
    wide = _non_embedding_N(config_for(base, SizeSpec(64, 2, 2), "w"))
    deep = _non_embedding_N(config_for(base, SizeSpec(32, 4, 2), "d"))
    assert 0 < small < wide      # wider -> more non-embedding params
    assert small < deep          # deeper -> more non-embedding params
    # And it must exclude the embedding tables (which dominate the total).
    from mythos.model import GPT
    cfg = config_for(base, SizeSpec(32, 2, 2), "s")
    cfg.sync_vocab_from_tokenizer(50257)
    total = GPT.from_config(cfg).count_parameters()
    assert small < total


def test_write_tsv(tmp_path):
    rows = [{"label": "a", "n_embd": 32, "depth": 2, "n_head": 2,
             "N": 1000, "D": 5000, "min_val_bpb": 1.23, "steps": 7, "seed": 42}]
    path = tmp_path / "sub" / "scaling_results.tsv"
    write_tsv(path, rows)
    text = path.read_text()
    assert text.splitlines()[0].split("\t")[:2] == ["label", "n_embd"]
    assert "1.23" in text and "\ta\t" in "\t" + text.splitlines()[1] + "\t"
