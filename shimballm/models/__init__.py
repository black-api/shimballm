"""Model builders: the built-in TinyGPT or any Hugging Face causal LM."""

from __future__ import annotations

from typing import Optional
import torch.nn as nn

from ..config import ModelConfig, RefConfig
from .tiny_gpt import TinyGPT


def build_model(cfg: ModelConfig, vocab_size: Optional[int] = None) -> nn.Module:
    """Build the student. ``arch="tiny"`` -> TinyGPT; ``arch="hf"`` ->
    ``AutoModelForCausalLM`` (requires the optional ``transformers`` extra)."""
    v = vocab_size or cfg.vocab_size
    if cfg.arch == "tiny":
        return TinyGPT(
            vocab_size=v,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            ffn_mult=cfg.ffn_mult,
            use_swiglu=cfg.use_swiglu,
            tie_embeddings=cfg.tie_embeddings,
            dropout=cfg.dropout,
            max_seq_len=cfg.max_seq_len,
            init_std=cfg.init_std,
        )
    if cfg.arch == "hf":
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "transformers is required for model.arch='hf'. "
                "Install with: pip install 'shimballm[hf]'"
            ) from e
        if not cfg.hf_name:
            raise ValueError("model.arch='hf' requires model.hf_name")
        return AutoModelForCausalLM.from_pretrained(cfg.hf_name)
    raise ValueError(f"unknown model arch: {cfg.arch}")


def build_reference_model(cfg: RefConfig, vocab_size: Optional[int] = None) -> nn.Module:
    """Build the small frozen scorer theta_ref. Keep it ~5x smaller than the
    student (paper: 0.3B scorer for a 1.5B student)."""
    v = vocab_size or cfg.vocab_size
    if cfg.arch == "tiny":
        return TinyGPT(
            vocab_size=v,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            ffn_mult=cfg.ffn_mult,
            use_swiglu=cfg.use_swiglu,
            tie_embeddings=True,
            dropout=0.0,
            max_seq_len=cfg.max_seq_len,
        )
    if cfg.arch == "hf":
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "transformers is required for ref.arch='hf'. "
                "Install with: pip install 'shimballm[hf]'"
            ) from e
        assert cfg.hf_name, "ref.arch='hf' requires ref.hf_name"
        return AutoModelForCausalLM.from_pretrained(cfg.hf_name)
    raise ValueError(f"unknown ref arch: {cfg.arch}")


def logits_of(output) -> "object":  # -> Tensor[B, T, V]
    """Normalize HF ``CausalLMOutput`` vs plain Tensor to a logits tensor."""
    return output.logits if hasattr(output, "logits") else output
