"""Adaptive stage gating (Eq. 6): advance a stage when marginal learning stalls.

After every evaluation window we compute the slope of the *validation excess
loss* (student validation CE minus reference validation CE on the same
batches). The gate measures marginal learning: improvement = -(slope of the
excess loss), positive while the student still improves. A stage transition
fires when

    improvement_W < gamma

equivalently ``slope_W < gamma`` in the paper's convention, for ``patience``
consecutive windows. This is the rule that makes the curriculum adaptive: a
noisy corpus stalls early and advances to selectivity sooner; a rich corpus
earns more stable-phase compute. Together with theta_doc, gamma is one of
only two hyper-parameters CROP adds to a standard WSD run.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple


class ExcessLossGate:
    def __init__(
        self,
        gamma: float = 1e-5,
        window_points: int = 4,
        patience: int = 2,
        min_points: int = 2,
    ):
        self.gamma = gamma
        self.window_points = window_points
        self.patience = patience
        self.min_points = min_points
        self.history: List[Tuple[int, float]] = []   # (cum_tokens, val_excess_loss)
        self._recent: Deque[float] = deque(maxlen=window_points)
        self._below_streak = 0

    def record(self, cum_tokens: int, val_excess_loss: float) -> None:
        self.history.append((int(cum_tokens), float(val_excess_loss)))
        self._recent.append(float(val_excess_loss))

    def slope(self) -> Optional[float]:
        """Least-squares slope of the last ``window_points`` excess-loss
        evaluations, returned as *improvement per evaluation* (positive when
        the student is still learning), or None if too few points."""
        if len(self._recent) < self.min_points:
            return None
        pts = list(self._recent)
        n = len(pts)
        xs = list(range(n))
        mx, my = sum(xs) / n, sum(pts) / n
        var = sum((x - mx) ** 2 for x in xs)
        if var == 0:
            return 0.0
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, pts))
        return -(cov / var)

    def should_advance(self, tokens_into_stage: int, stage_budget: int,
                       min_stage_fraction: float = 0.25) -> Tuple[bool, str]:
        """Return (fire?, reason). Fires when improvement per eval stays below
        ``gamma`` for ``patience`` windows, and never before
        ``min_stage_fraction`` of the stage's budget has been consumed."""
        if tokens_into_stage < min_stage_fraction * stage_budget:
            return False, "min-stage-not-reached"
        improvement = self.slope()
        if improvement is None:
            return False, "not-enough-evals"
        if improvement < self.gamma:
            self._below_streak += 1
        else:
            self._below_streak = 0
        if self._below_streak >= self.patience:
            return True, (f"improvement={improvement:.3e}<{self.gamma:.1e} "
                          f"for {self._below_streak} windows")
        return False, "plateau-not-reached"

    def reset_stage(self) -> None:
        """Clear the window history at a stage boundary."""
        self._recent.clear()
        self._below_streak = 0

    def state_dict(self) -> dict:
        return {
            "history": self.history,
            "below_streak": self._below_streak,
            "gamma": self.gamma,
            "patience": self.patience,
            "window_points": self.window_points,
        }

    def load_state_dict(self, st: dict) -> None:
        self.history = [tuple(x) for x in st["history"]]
        self._below_streak = int(st["below_streak"])
        self._recent.clear()
        for _, v in self.history[-self.window_points:]:
            self._recent.append(v)
