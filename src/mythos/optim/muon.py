from __future__ import annotations

import math
from typing import Iterable

import torch


def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonalization via Newton-Schulz (Muon core)."""
    assert g.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    if g.size(-2) > g.size(-1):
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        a_mat = x @ x.mT
        x = a * x + (b * a_mat + c * a_mat @ a_mat) @ x
    if g.size(-2) > g.size(-1):
        x = x.mT
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D weight matrices."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                g = g.lerp(buf, momentum) if momentum else g
                if p.ndim >= 2:
                    update = _zeropower_via_newtonschulz5(g, steps=ns_steps)
                else:
                    update = g
                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class NorMuon(Muon):
    """NorMuon: Muon with per-row RMS normalization."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                g = g.lerp(buf, momentum) if momentum else g
                if p.ndim >= 2:
                    row_rms = g.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
                    g = g / row_rms
                    update = _zeropower_via_newtonschulz5(g, steps=ns_steps)
                else:
                    update = g
                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class Aurora(NorMuon):
    """Aurora: leverage-aware updates for MLP up/gate projections."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                g = g.lerp(buf, momentum) if momentum else g
                if p.ndim >= 2 and p.shape[0] >= p.shape[1]:
                    row_mean = g.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
                    g = g / row_mean
                if p.ndim >= 2:
                    row_rms = g.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
                    g = g / row_rms
                    update = _zeropower_via_newtonschulz5(g, steps=ns_steps)
                else:
                    update = g
                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


def build_optimizer(model: torch.nn.Module, name: str, lr: float, weight_decay: float):
    """Split params: 2D matrices → Muon family, rest → AdamW."""
    muon_params: list[torch.nn.Parameter] = []
    adam_params: list[torch.nn.Parameter] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "wte" not in n and "lm_head" not in n:
            muon_params.append(p)
        else:
            adam_params.append(p)

    opt_cls = {"muon": Muon, "nor_muon": NorMuon, "aurora": Aurora}.get(name, NorMuon)
    muon = opt_cls(muon_params, lr=lr * 0.5, weight_decay=weight_decay)
    adam = torch.optim.AdamW(adam_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    return muon, adam


def optimizer_step(muon: torch.optim.Optimizer, adam: torch.optim.Optimizer) -> None:
    muon.step()
    adam.step()
    muon.zero_grad(set_to_none=True)
    adam.zero_grad(set_to_none=True)


def lr_schedule(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
