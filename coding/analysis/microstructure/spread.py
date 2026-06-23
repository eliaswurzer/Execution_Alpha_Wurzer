"""
spread.py -- Spread- und Midquote-Hilfen (Thesis §5.4.2).

Alle Funktionen operieren auf einem bereits RTH-gefilterten NBBO-DataFrame
mit Spalten ``time``, ``best_bid``, ``best_offer``, ``mid``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def midquote(best_bid: pd.Series, best_offer: pd.Series) -> pd.Series:
    """Arithmetisches Mittel zwischen Bid und Ask."""
    return (best_bid.astype(float) + best_offer.astype(float)) / 2.0


def quoted_spread(best_bid: pd.Series, best_offer: pd.Series) -> pd.Series:
    """Absoluter Quoted Spread ``P_A - P_B``."""
    return best_offer.astype(float) - best_bid.astype(float)


def half_spread_bps(nbbo: pd.DataFrame) -> pd.Series:
    """Half-Spread in Basispunkten relativ zum Midquote.

    ``hs_bps = 0.5 * (P_A - P_B) / mid * 10000``
    """
    mid = nbbo["mid"].astype(float)
    qs = nbbo["best_offer"].astype(float) - nbbo["best_bid"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 * qs / mid.replace(0, np.nan) * 1e4


def effective_spread_bps(
    trades: pd.DataFrame,
    nbbo: pd.DataFrame,
) -> pd.Series:
    """Effective Spread je Trade in Basispunkten (Hasbrouck-Konvention).

    ``es = 2 * D * (P - m)`` mit D = +1 fuer buyer-initiated, -1 fuer seller-
    initiated. Trade-Direction wird nach Lee-Ready klassifiziert (Trade ueber
    Mid = BUY, unter Mid = SELL, am Mid = Tick-Test).
    """
    if trades.empty:
        return pd.Series([], dtype=float)

    ns = nbbo[["time", "best_bid", "best_offer", "mid"]].sort_values("time")
    tr = trades[["time", "price"]].sort_values("time").copy()
    merged = pd.merge_asof(tr, ns, on="time", direction="backward")

    direction = np.sign(merged["price"] - merged["mid"])
    # Tick-Test Fallback bei direction == 0
    tick_direction = np.sign(merged["price"].diff().fillna(0.0))
    direction = np.where(direction == 0, tick_direction, direction)

    es = 2 * direction * (merged["price"] - merged["mid"])
    es_bps = es / merged["mid"].replace(0, np.nan) * 1e4
    return es_bps
