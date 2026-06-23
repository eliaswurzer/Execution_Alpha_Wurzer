"""
signing.py -- Trade-Signing-Methoden.

Zwei Methoden:

* ``lee_ready``         -- Standard Lee-Ready (1991): Quote zum Trade-Zeitpunkt
                            (asof-backward), Trade ueber Mid = BUY, unter Mid =
                            SELL, gleich Mid -> Tick-Test.
* ``holden_jacobsen``   -- Holden-Jacobsen (2014): identisch zu Lee-Ready, aber
                            der prevailing-Quote wird **eine Millisekunde vor**
                            dem Trade-Timestamp gesucht (Quote-Race-Korrektur),
                            mit Edge-Case-Behandlung wenn der Quote im selben
                            Millisekunden-Bucket landet.

Konfiguration via ``cfg.TRADE_SIGN_METHOD`` ("lee_ready" oder "holden_jacobsen").
Konsumenten (OFI, Imbalance, Spread) greifen auf ``sign_trades`` zu.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .. import config as cfg


SignMethod = Literal["lee_ready", "holden_jacobsen"]


def _tick_test_signs(prices: pd.Series) -> pd.Series:
    diffs = prices.astype(float).diff()
    s = np.sign(diffs)
    # Forward-fill 0er via vorherigem Sign (uptick = +1, downtick = -1)
    s = s.replace(0.0, np.nan).ffill().fillna(0.0)
    return s


def lee_ready_sign(
    trades: pd.DataFrame, nbbo: pd.DataFrame,
) -> pd.Series:
    """Lee-Ready (1991) Trade-Direction (+1 BUY, -1 SELL, 0 unklar)."""
    if trades.empty:
        return pd.Series([], dtype=float, index=trades.index)
    ns = nbbo[["time", "mid"]].sort_values("time")
    tr = trades[["time", "price"]].sort_values("time").copy()
    merged = pd.merge_asof(tr, ns, on="time", direction="backward")
    direction = np.sign(merged["price"] - merged["mid"])
    direction = direction.where(direction != 0, _tick_test_signs(merged["price"]))
    return pd.Series(direction.to_numpy(), index=tr.index, name="sign")


def holden_jacobsen_sign(
    trades: pd.DataFrame, nbbo: pd.DataFrame, lag_ms: int = 1,
) -> pd.Series:
    """Holden-Jacobsen (2014): NBBO ``lag_ms`` Millisekunden vor dem Trade.

    Edge-Case: wenn das gewuenschte Lag-Timestamp vor dem ersten NBBO-Update
    des Tages liegt, faellt die Routine auf den ersten verfuegbaren Quote
    zurueck (entspricht stilles Behandeln von Pre-Open-Trades).
    """
    if trades.empty:
        return pd.Series([], dtype=float, index=trades.index)
    ns = nbbo[["time", "mid"]].sort_values("time")
    tr = trades.sort_values("time").copy()
    tr["_lookup_ts"] = tr["time"] - pd.Timedelta(milliseconds=lag_ms)
    merged = pd.merge_asof(
        tr[["_lookup_ts", "price"]].rename(columns={"_lookup_ts": "time"}),
        ns, on="time", direction="backward",
    )
    direction = np.sign(merged["price"] - merged["mid"])
    direction = direction.where(direction != 0, _tick_test_signs(merged["price"]))
    return pd.Series(direction.to_numpy(), index=tr.index, name="sign")


def sign_trades(
    trades: pd.DataFrame, nbbo: pd.DataFrame,
    method: SignMethod | None = None,
) -> pd.Series:
    """Dispatch nach ``cfg.TRADE_SIGN_METHOD`` (oder explizitem ``method``)."""
    use = method or cfg.TRADE_SIGN_METHOD
    if use == "lee_ready":
        return lee_ready_sign(trades, nbbo)
    if use == "holden_jacobsen":
        return holden_jacobsen_sign(trades, nbbo, lag_ms=cfg.HJ_LAG_MS)
    raise ValueError(f"Unknown signing method: {use!r}")
