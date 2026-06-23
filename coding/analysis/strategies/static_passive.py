"""S1 -- Static Passive (Thesis §4.3.3, Eq. 4.7/4.8)."""

from __future__ import annotations

import pandas as pd

from .base import ExecutionStrategy, MarketState


class StaticPassiveStrategy(ExecutionStrategy):
    """Konstanter Offset ``delta*`` am Touch, Refresh alle ``Delta_r`` Sekunden."""

    name = "S1_STATIC"

    def limit_offset_bps(
        self,
        t: pd.Timestamp,
        side: str,
        state: MarketState,
        sigma_bar: float,
        delta_max_bps: float,
    ) -> float:
        # delta_max_bps kommt als Tier-spezifisches delta* rein.
        return float(delta_max_bps)
