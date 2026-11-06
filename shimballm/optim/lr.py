"""Warmup-Stable-Decay (WSD) learning-rate schedule.

The WSD backbone (MiniCPM) keeps the LR constant through Stages A-B and puts
the entire decay inside Stage C. That is what makes the Stage-B checkpoint a
reusable asset: reload it and re-anneal on a fresh slice for each new compute
budget (compute recycling).

The schedule is defined over *cumulative tokens* (not steps) because stage
boundaries are token budgets and can move when the adaptive gate fires.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WSDLRSchedule:
    peak_lr: float
    end_lr_fraction: float = 0.1
    warmup_fraction: float = 0.05   # of the total run budget
    total_tokens: int = 1_000_000

    # actual (possibly gate-shifted) boundaries in cumulative tokens
    _warmup_end: int = 0
    _decay_start: int = 0
    _decay_end: int = 0

    def __post_init__(self) -> None:
        self.set_boundaries(0.0, 0.85, 1.0)

    def set_boundaries(self, warmup_end_frac: float, decay_start_frac: float,
                       decay_end_frac: float) -> None:
        t = self.total_tokens
        self._warmup_end = int(t * warmup_end_frac)
        self._decay_start = int(t * decay_start_frac)
        self._decay_end = max(int(t * decay_end_frac), self._decay_start + 1)

    def set_stage_boundaries(self, a_end: int, b_end: int, total: int) -> None:
        """Convenience: linear warmup over the first ``warmup_fraction`` of the
        run (capped at the end of Stage A), stable through Stage B, decay
        across Stage C. Boundaries may be gate-shifted; pass actual values."""
        self._warmup_end = min(int(total * self.warmup_fraction), a_end)
        self._decay_start = b_end
        self._decay_end = total

    def lr_at(self, cum_tokens: int) -> float:
        if cum_tokens < self._warmup_end:
            # linear warmup (at least 1 token of warmup to avoid div by zero)
            w = max(self._warmup_end, 1)
            return self.peak_lr * (cum_tokens + 1) / w
        if cum_tokens < self._decay_start:
            return self.peak_lr
        frac = (cum_tokens - self._decay_start) / max(self._decay_end - self._decay_start, 1)
        frac = min(max(frac, 0.0), 1.0)
        # linear WSD decay (MiniCPM uses a truncated-linear decay; a cosine
        # variant is available via decay_shape="cosine")
        return self.peak_lr * (self.end_lr_fraction + (1 - self.end_lr_fraction) * (1 - frac))

    def lr_at_cosine(self, cum_tokens: int) -> float:
        """Cosine-shaped decay inside Stage C (optional variant)."""
        if cum_tokens < self._warmup_end:
            w = max(self._warmup_end, 1)
            return self.peak_lr * (cum_tokens + 1) / w
        if cum_tokens < self._decay_start:
            return self.peak_lr
        import math

        frac = (cum_tokens - self._decay_start) / max(self._decay_end - self._decay_start, 1)
        frac = min(max(frac, 0.0), 1.0)
        cos = 0.5 * (1 + math.cos(math.pi * frac))
        return self.peak_lr * (self.end_lr_fraction + (1 - self.end_lr_fraction) * cos)

    def state_dict(self) -> dict:
        return {
            "peak_lr": self.peak_lr,
            "end_lr_fraction": self.end_lr_fraction,
            "warmup_fraction": self.warmup_fraction,
            "total_tokens": self.total_tokens,
            "warmup_end": self._warmup_end,
            "decay_start": self._decay_start,
            "decay_end": self._decay_end,
        }

    def load_state_dict(self, st: dict) -> None:
        self.peak_lr = st["peak_lr"]
        self.end_lr_fraction = st["end_lr_fraction"]
        self.warmup_fraction = st["warmup_fraction"]
        self.total_tokens = st["total_tokens"]
        self._warmup_end = st["warmup_end"]
        self._decay_start = st["decay_start"]
        self._decay_end = st["decay_end"]
