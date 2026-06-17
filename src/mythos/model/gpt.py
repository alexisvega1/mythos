from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mythos.config import MythosConfig


def relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos + rotated * sin


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.use_qk_norm = config.use_qk_norm
        self.q_norm = RMSNorm(self.head_dim) if config.use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if config.use_qk_norm else nn.Identity()
        self.block_size = config.block_size
        self.use_rope = config.use_rope
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            cos, sin = build_rope_cache(t, self.head_dim, x.device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool)).view(1, 1, t, t)
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.activation = relu2 if config.activation == "relu2" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.activation(self.fc(x)))


class Block(nn.Module):
    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-style decoder with modded-nanogpt-inspired defaults (RoPE, QK-Norm, ReLU²)."""

    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        if config.untied_embeddings:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        else:
            self.lm_head = None
        self.apply(self._init_weights)
        if config.untied_embeddings:
            nn.init.zeros_(self.lm_head.weight)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.T if self.lm_head is None else self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config: MythosConfig) -> GPT:
        return cls(config)
