"""S4 -- Optimal Schedule Strategy (Thesis extension).

Uses a presample-calibrated TODSchedule (XGBoost regressor predicting expected
adverse selection) to dynamically allocate slice sizes across refresh intervals.

Posts proportionally MORE in intervals where predicted AS is LOW (favorable
execution conditions) and LESS when AS is HIGH. Price aggressiveness is
inherited from S2 (time-adaptive urgency function).

The slice-size hook keeps the shared passive simulator and replaces only the
TWAP quantity schedule.
"""

from __future__ import annotations

import pandas as pd

from ..fill_model.state_vector import state_at
from ..fill_model.tod_schedule import TODSchedule
from .base import MarketState
from .time_adaptive import TimeAdaptiveStrategy


class OptimalScheduleStrategy(TimeAdaptiveStrategy):
    """S4: Time-adaptive price + XGBoost-learned quantity schedule."""

    name = "S4_TOD"

    def __init__(self, tod_schedule: TODSchedule | None, **kwargs):
        if tod_schedule is None:
            raise ValueError("S4_TOD requires a fitted TODSchedule artifact")
        super().__init__(**kwargs)
        self._tod = tod_schedule

    def slice_size(
        self,
        t: pd.Timestamp,
        cutoff: pd.Timestamp,
        qty_remaining: int,
        side: str,
        state: MarketState,
    ) -> int:
        """Learned quantity schedule plugged into the shared simulator."""
        intervals_remaining = max(
            1, int((cutoff - t).total_seconds() / self.refresh_seconds),
        )
        if not self._tod.fitted:
            return super().slice_size(t, cutoff, qty_remaining, side, state)

        sv = state_at(
            t, state.nbbo, state.ofi, state.rv, side,
            nbbo_times=state.nbbo_times,
            ofi_times=state.ofi_times,
            rv_times=state.rv_times,
        )
        frac = self._tod.fraction(t, intervals_remaining, sv)
        return max(1, int(qty_remaining * frac))
