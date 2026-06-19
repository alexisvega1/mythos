"""Property tests for rotary position embeddings (RoPE).

These guard the two defining invariants of a correct RoPE. They fail if the
cos/sin cache layout is ever de-aligned from apply_rope's interleaved (GPT-J)
pair convention — the bug that previously shipped as cat((freqs, freqs)).
"""
from __future__ import annotations

import torch

from mythos.model.gpt import apply_rope, build_rope_cache


def _rope_at(x: torch.Tensor, pos: int, seq_len: int, head_dim: int) -> torch.Tensor:
    """Apply RoPE to a single-token tensor x[...,1,d] as if it sat at `pos`."""
    cos, sin = build_rope_cache(seq_len, head_dim, x.device)
    return apply_rope(x, cos[:, :, pos : pos + 1, :], sin[:, :, pos : pos + 1, :])


def test_rope_preserves_per_token_norm():
    """A rotation must preserve each token's L2 norm exactly (the cat-layout bug did not)."""
    torch.manual_seed(0)
    b, h, t, d = 2, 3, 16, 8
    x = torch.randn(b, h, t, d)
    cos, sin = build_rope_cache(t, d, x.device)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5), (
        "RoPE changed per-token norm — cos/sin layout is misaligned with apply_rope"
    )


def test_rope_position_zero_is_identity():
    """At position 0 (angle 0) RoPE is the identity map."""
    torch.manual_seed(1)
    x = torch.randn(1, 1, 1, 8)
    y = _rope_at(x, pos=0, seq_len=4, head_dim=8)
    assert torch.allclose(x, y, atol=1e-6)


def test_rope_dot_product_depends_only_on_relative_position():
    """The defining RoPE property: <rope(q, m), rope(k, n)> depends only on (m - n)."""
    torch.manual_seed(2)
    d, seq_len = 8, 32
    q = torch.randn(1, 1, 1, d)
    k = torch.randn(1, 1, 1, d)

    def dot(m: int, n: int) -> float:
        qm = _rope_at(q, m, seq_len, d)
        kn = _rope_at(k, n, seq_len, d)
        return (qm * kn).sum().item()

    # Same offset (= +2) at three different absolute positions must match.
    base = dot(2, 0)
    assert abs(dot(5, 3) - base) < 1e-3
    assert abs(dot(10, 8) - base) < 1e-3
    # A different offset should generally give a different score (sanity, not vacuous).
    assert abs(dot(7, 0) - base) > 1e-3
