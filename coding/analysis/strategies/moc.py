"""S0 -- Market-on-Close Benchmark (Thesis §4.3.2)."""

from __future__ import annotations

import pandas as pd

from .base import FillResult, MarketState


class MOCStrategy:
    """Gesamt-x wird als MOC geroutet; Alpha = 0 by construction."""

    name = "S0_MOC"

    def simulate(
        self,
        order: pd.Series,
        state: MarketState,
        fill_model=None,
        sigma_bar: float | None = None,
        delta_max_bps: float | None = None,
        **_: object,
    ) -> FillResult:
        qty = int(order["qty"])
        close = float(state.close_price) if state.close_price else float("nan")
        return FillResult(
            order_id=str(order["order_id"]),
            symbol=state.symbol,
            date=state.date,
            side=order["side"],
            strategy=self.name,
            window=str(order.get("window", "")),
            qty_intended=qty,
            qty_filled_passive=0,
            qty_filled_moc=qty,
            vwap_passive=float("nan"),
            close_price=close,
            avg_fill_price=close,
            fill_rate=0.0,
        )
