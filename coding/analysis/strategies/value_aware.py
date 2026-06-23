"""Value-aware passive posting strategy.

S5 evaluates a grid of passive posting distances with an optional value model.
It preserves the standard residual MOC routing by sharing the base simulation
loop with S1-S4.
"""

from __future__ import annotations

import pandas as pd

from .. import config as cfg
from ..fill_model.state_vector import state_at
from .base import ExecutionStrategy, MarketState


class ValueAwareXGBStrategy(ExecutionStrategy):
    """Choose the posting offset with the highest predicted execution value."""

    name = "S5_VALUE_AWARE_XGB"

    def __init__(
        self,
        *,
        value_model=None,
        tier: int | None = None,
        size_frac: float = cfg.PARENT_ORDER_PRIMARY_FRACTION,
        sector: str = "",
        listing_exchange: str = "",
        offset_grid_bps: tuple[float, ...] = cfg.VALUE_MODEL_OFFSET_GRID_BPS,
        min_expected_value_bps: float = cfg.VALUE_MODEL_MIN_EXPECTED_ALPHA_BPS,
        window_start: pd.Timestamp | None = None,
        moc_cutoff: pd.Timestamp | None = None,
        refresh_seconds: int = cfg.REFRESH_SECONDS_DEFAULT,
    ):
        super().__init__(refresh_seconds=refresh_seconds)
        self.value_model = value_model
        self.tier = tier
        self.size_frac = float(size_frac)
        self.sector = sector or ""
        self.listing_exchange = listing_exchange or ""
        self.offset_grid_bps = tuple(max(0.0, float(x)) for x in offset_grid_bps)
        self.min_expected_value_bps = float(min_expected_value_bps)
        self.window_start = window_start
        self.moc_cutoff = moc_cutoff
        self._last_key: tuple[pd.Timestamp, str] | None = None
        self._last_value_bps: float = float("-inf")
        self._last_offset_bps: float = 0.0

    def _candidate_frame(
        self,
        t: pd.Timestamp,
        side: str,
        state: MarketState,
    ) -> pd.DataFrame:
        rows = []
        cutoff = self.moc_cutoff or pd.Timestamp.combine(state.date, cfg.MOC_CUTOFF)
        time_to_cutoff = max(0.0, float((cutoff - t).total_seconds()))
        for offset in self.offset_grid_bps:
            sv = state_at(
                t,
                state.nbbo,
                state.ofi,
                state.rv,
                side,
                limit_offset_bps=offset,
                nbbo_times=state.nbbo_times,
                ofi_times=state.ofi_times,
                rv_times=state.rv_times,
            )
            rows.append({
                **sv,
                "symbol": state.symbol,
                "date": state.date,
                "side": side,
                "tier": self.tier if self.tier is not None else 0,
                "size_frac": self.size_frac,
                "time_to_cutoff_seconds": time_to_cutoff,
                "sector": self.sector,
                "listing_exchange": self.listing_exchange,
            })
        return pd.DataFrame(rows)

    def _predict_values(self, candidates: pd.DataFrame) -> list[float]:
        if self.value_model is None or candidates.empty:
            return [float("-inf")] * len(candidates)
        if hasattr(self.value_model, "predict_candidates"):
            return list(self.value_model.predict_candidates(candidates))
        if hasattr(self.value_model, "predict_frame"):
            return list(self.value_model.predict_frame(candidates))
        raise TypeError("value_model must implement predict_candidates or predict_frame")

    def limit_offset_bps(
        self,
        t: pd.Timestamp,
        side: str,
        state: MarketState,
        sigma_bar: float,
        delta_max_bps: float,
    ) -> float:
        del sigma_bar, delta_max_bps
        key = (pd.Timestamp(t), str(side).upper())
        candidates = self._candidate_frame(pd.Timestamp(t), str(side).upper(), state)
        values = self._predict_values(candidates)
        if not values:
            self._last_key = key
            self._last_value_bps = float("-inf")
            self._last_offset_bps = 0.0
            return 0.0
        best_pos = int(max(range(len(values)), key=lambda i: float(values[i])))
        self._last_key = key
        self._last_value_bps = float(values[best_pos])
        self._last_offset_bps = float(candidates["limit_offset_bps"].iloc[best_pos])
        return self._last_offset_bps

    def slice_size(
        self,
        t: pd.Timestamp,
        cutoff: pd.Timestamp,
        qty_remaining: int,
        side: str,
        state: MarketState,
    ) -> int:
        key = (pd.Timestamp(t), str(side).upper())
        if self._last_key == key and self._last_value_bps <= self.min_expected_value_bps:
            return 0
        return super().slice_size(t, cutoff, qty_remaining, side, state)