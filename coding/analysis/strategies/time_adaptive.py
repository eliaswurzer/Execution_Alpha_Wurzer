"""S2 -- Time-Adaptive Strategy (Thesis §4.3.4, Eq. 4.9-4.11).

``delta^{(2)}(t) = delta_max * g(t) * h(sigma_t)``

mit linearem Urgency-Schedule ``g(t) = (T1 - t)/(T1 - T0)`` und Vol-Scalar
``h(sigma) = sigma_t / sigma_bar_i``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg
from .base import ExecutionStrategy, MarketState


class TimeAdaptiveStrategy(ExecutionStrategy):
    name = "S2_TIME_ADAPTIVE"

    def __init__(self, *, window_start: pd.Timestamp | None = None,
                 moc_cutoff: pd.Timestamp | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._T0 = window_start
        self._T1 = moc_cutoff

    def _urgency(self, t: pd.Timestamp) -> float:
        if self._T0 is None or self._T1 is None:
            return 1.0
        total = (self._T1 - self._T0).total_seconds()
        if total <= 0:
            return 0.0
        remaining = (self._T1 - t).total_seconds()
        return float(np.clip(remaining / total, 0.0, 1.0))

    def _vol_scalar(self, t: pd.Timestamp, state: MarketState, sigma_bar: float) -> float:
        if state.rv is None or state.rv.empty or sigma_bar is None or sigma_bar <= 0:
            return 1.0
        idx = int(np.searchsorted(state.rv_times, t.value, side="right")) - 1
        if idx < 0:
            return 1.0
        sigma_t = float(state.rv.iloc[idx])
        if not np.isfinite(sigma_t):
            return 1.0
        return sigma_t / sigma_bar

    def limit_offset_bps(
        self,
        t: pd.Timestamp,
        side: str,
        state: MarketState,
        sigma_bar: float,
        delta_max_bps: float,
    ) -> float:
        g = self._urgency(t)
        h = self._vol_scalar(t, state, sigma_bar)
        return float(delta_max_bps) * g * h
