"""Built-in decoder-only transformer (TinyGPT).

A compact, dependency-free causal LM used for smoke tests, TinyStories-scale
demos, and as the small reference scorer theta_ref. Supports pre-norm blocks,
RoPE, GQA, tied embeddings, and optional SwiGLU -- the same architecture
family the paper's survey (MobileLLM, Llama-3.2, Qwen2) credits for SLM
quality-per-parameter.
"""

from __future__ import annotations

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, n_heads, T, head_dim]
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, : x.shape[-2], :]
    sin = sin[None, None, : x.shape[-2], :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., ::2], out[..., 1::2] = out1, out2
    return out


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int], dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cos, sin) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.n_kv_heads != self.n_heads:  # GQA
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, ffn_mult: float, use_swiglu: bool, dropout: float):
        super().__init__()
        hidden = int(ffn_mult * d_model)
        self.swiglu = use_swiglu
        if use_swiglu:
            self.gate_proj = nn.Linear(d_model, hidden, bias=False)
            self.up_proj = nn.Linear(d_model, hidden, bias=False)
            self.down_proj = nn.Linear(hidden, d_model, bias=False)
        else:
            self.fc = nn.Linear(d_model, hidden, bias=False)
            self.proj = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.swiglu:
            h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        else:
            h = F.gelu(self.fc(x))
        return self.drop(self.down_proj(h) if self.swiglu else self.proj(h))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = nn.LayerNorm(cfg["d_model"])
        self.attn = Attention(cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"], cfg["dropout"])
        self.n2 = nn.LayerNorm(cfg["d_model"])
        self.mlp = MLP(cfg["d_model"], cfg["ffn_mult"], cfg["use_swiglu"], cfg["dropout"])

    def forward(self, x, cos, sin) -> torch.Tensor:
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class TinyGPT(nn.Module):
    """Decoder-only transformer with a simple ``model(ids) -> logits`` API
    (compatible with both the trainer and the reference scorer)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        n_kv_heads: Optional[int] = None,
        ffn_mult: float = 4.0,
        use_swiglu: bool = False,
        tie_embeddings: bool = True,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        init_std: float = 0.02,
    ):
        super().__init__()
        assert n_heads % (n_kv_heads or n_heads) == 0 or n_kv_heads is None
        self.cfg = dict(
            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads or n_heads,
            ffn_mult=ffn_mult, use_swiglu=use_swiglu, dropout=dropout,
        )
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(self.cfg) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        for name, p in self.named_parameters():  # scaled residual init
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight") or name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=init_std / math.sqrt(2 * n_layers))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        cos, sin = rope_cache(T, self.cfg["n_heads"] and self.cfg["d_model"] // self.cfg["n_heads"], idx.device)
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        return self.head(x)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n
