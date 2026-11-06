"""Muon optimizer (orthogonalized momentum) with an AdamW fallback.

Muon (Liu et al., 2025 -- used for the Moonlight MoE) treats each 2-D weight
matrix's update as an orthogonalized momentum step via a quintic
Newton-Schulz iteration, and demonstrates ~2x compute efficiency vs AdamW at
compute-optimal settings. Non-2-D parameters (embeddings, head, norms,
biases) are delegated to an embedded AdamW. This is an optional lever of the
CROP recipe: set ``schedule.optimizer: muon`` to enable.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple
import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalization (Keller Jordan's Muon)."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(Optimizer):
    """Muon for 2-D params + AdamW for everything else.

    Args:
        muon_params: 2-D weight matrices (Linear weights).
        adamw_params: embeddings / heads / norms / biases.
        lr: Muon learning rate (applied to orthogonalized updates).
        adamw_lr: learning rate for the AdamW group.
        momentum, nesterov: Muon momentum.
        weight_decay: AdamW weight decay (Muon groups are decay-free here).
    """

    def __init__(
        self,
        muon_params: Iterable[torch.nn.Parameter],
        adamw_params: Iterable[torch.nn.Parameter] = (),
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        adamw_lr: float = 3e-3,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
    ):
        params = list(muon_params)
        adam = list(adamw_params)
        super().__init__(
            [
                {"params": params, "use_muon": True, "lr": lr, "weight_decay": 0.0},
                {
                    "params": adam,
                    "use_muon": False,
                    "lr": adamw_lr,
                    "betas": adamw_betas,
                    "weight_decay": weight_decay,
                },
            ],
            defaults={},
        )
        self.momentum = momentum
        self.nesterov = nesterov
        assert all(p.ndim == 2 for p in params), "Muon groups must be 2-D matrices"

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if group["use_muon"]:
                    if "mom" not in state:
                        state["mom"] = torch.zeros_like(g)
                    buf = state["mom"]
                    buf.lerp_(g, 1 - self.momentum)
                    u = g.lerp_(buf, self.momentum) if self.nesterov else buf
                    u = zeropower_via_newtonschulz5(u, steps=5)
                    scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                    p.add_(u.reshape(p.shape), alpha=-lr * scale)
                else:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                        state["step"] = 0
                    state["step"] += 1
                    a, b = group["betas"]
                    ea, eas = state["exp_avg"], state["exp_avg_sq"]
                    ea.lerp_(g, 1 - a)
                    eas.mul_(b).addcmul_(g, g, value=1 - b)
                    bias1 = 1 - a ** state["step"]
                    bias2 = 1 - b ** state["step"]
                    denom = (eas / bias2).sqrt_().add_(1e-8)
                    step_size = lr / bias1
                    if group["weight_decay"] > 0:
                        p.mul_(1 - group["weight_decay"] * lr)
                    p.addcdiv_(ea, denom, value=-step_size)
        return loss


def build_optimizer(model: nn.Module, kind: str, peak_lr: float, weight_decay: float,
                    muon_momentum: float, adam_betas: Tuple[float, float]):
    """Factory used by the trainer. Returns a torch Optimizer whose groups all
    expose ``lr`` so the WSD schedule can drive them uniformly."""
    if kind == "adamw":
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if (p.ndim < 2 or "norm" in n.lower()) else decay).append(p)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=peak_lr,
            betas=adam_betas,
        )
    if kind == "muon":
        muon_p = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim == 2]
        adam_p = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim != 2]
        # Muon lr convention differs from AdamW; scale peak_lr accordingly.
        return Muon(
            muon_params=muon_p,
            adamw_params=adam_p,
            lr=peak_lr * 4.0,
            momentum=muon_momentum,
            adamw_lr=peak_lr,
            adamw_betas=adam_betas,
            weight_decay=weight_decay,
        )
    raise ValueError(f"unknown optimizer: {kind}")
