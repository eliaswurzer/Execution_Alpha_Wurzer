"""
parent_orders.py -- Konstruktion synthetischer Parent-Orders fuer Thesis §5.5.

Fuer jeden Symbol-Tag werden je Windowsatz (A, B, C) und je Groesse
``x in {0.005, 0.01, 0.02, 0.05, 0.10} * E[V_C]`` Parent-Orders erzeugt. BUY
und SELL alternieren deterministisch ueber die Windows; alle Groessen eines
Windows teilen dieselbe Seite (matched Cross-Size-Vergleiche). Die Schaetzung
``E[V_C]`` erfolgt als rollierender 20-Tage-Average ueber tatsaechlich
beobachtete Closing-Auction-Volumina aus TAQ.

Schema
------
``order_id``: deterministisch, ``<date>_<symbol>_<window>_<size_bps>bp_<side>``
``symbol``, ``date``, ``side`` (BUY/SELL), ``qty`` (shares)
``arrival_time`` (pd.Timestamp), ``moc_cutoff`` (T1)
``size_frac`` (0.005/0.01/0.02/0.05/0.10), ``window`` (A/B/C), ``expected_vc``
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from typing import Iterable

import pandas as pd

from .. import config as cfg


_SIDE_CYCLE = {0: "BUY", 1: "SELL"}


def _arrival_timestamp(date: _dt.date, t: _dt.time) -> pd.Timestamp:
    return pd.Timestamp.combine(date, t)


def _side_start_offset(symbol: str, date: _dt.date, seed: int) -> int:
    """Deterministic per-symbol-day side offset for near-balanced panels."""
    key = f"{seed}|{date.isoformat()}|{symbol}".encode("utf-8")
    return hashlib.sha256(key).digest()[0] % 2


def _size_tag(size_frac: float) -> str:
    """Stable decimal-safe parent-size tag in basis points of expected close volume."""
    bps = int(round(float(size_frac) * 10_000))
    return f"{bps:04d}bp"


def build_parent_orders(
    symbol: str,
    date: _dt.date,
    expected_vc: float,
    *,
    size_fractions: Iterable[float] = cfg.PARENT_ORDER_SIZE_FRACTIONS,
    windows: dict[str, _dt.time] | None = None,
    moc_cutoff: _dt.time = cfg.MOC_CUTOFF,
    seed: int = cfg.DEFAULT_SEED,
) -> pd.DataFrame:
    """Gibt Parent-Order-DataFrame fuer einen Symbol-Tag zurueck.

    ``expected_vc`` ist der 20-Tage Trailing-Average des Closing-Auction-Volumens
    in Shares. Wenn der nicht verfuegbar ist (z.B. erste Sample-Tage), liefert
    die Funktion eine leere Liste.
    """
    if expected_vc is None or expected_vc <= 0:
        return pd.DataFrame()

    wins = windows or cfg.EXECUTION_WINDOWS
    orders = []
    side_offset = _side_start_offset(symbol, date, seed)
    # Side is keyed on the WINDOW index, not on a running order index: all
    # size buckets of one (symbol, day, window) share the same side, so
    # cross-size differences in the parent-size grid are matched within the
    # cell instead of mixing opposite sides. For single-size runs (headline
    # and bounds specifications) this is bit-identical to the previous
    # running-index rule because the order index equals the window index.
    for w_idx, (wname, wtime) in enumerate(wins.items()):
        side = _SIDE_CYCLE[(w_idx + side_offset) % 2]
        for frac in size_fractions:
            qty = int(round(frac * expected_vc))
            if qty <= 0:
                continue
            oid = f"{date.isoformat()}_{symbol}_{wname}_{_size_tag(frac)}_{side}"
            orders.append({
                "order_id": oid,
                "symbol": symbol,
                "date": date,
                "side": side,
                "qty": qty,
                "arrival_time": _arrival_timestamp(date, wtime),
                "moc_cutoff": _arrival_timestamp(date, moc_cutoff),
                "size_frac": float(frac),
                "window": wname,
                "expected_vc": float(expected_vc),
            })
    return pd.DataFrame(orders)


# ---------------------------------------------------------------------------
# E[V_C] Schaetzung
# ---------------------------------------------------------------------------

def rolling_expected_vc(
    vc_history: pd.DataFrame,
    lookback_days: int = 20,
) -> pd.DataFrame:
    """Rolling 20-Tage-Avg Closing-Auction-Volumen je Symbol.

    Parameters
    ----------
    vc_history : DataFrame mit ``symbol``, ``date``, ``vc_shares``.

    Returns
    -------
    DataFrame ``symbol``, ``date``, ``expected_vc`` -- ``expected_vc`` ist
    das Trailing-Average *vor* dem jeweiligen Tag (exclusive).
    """
    if vc_history.empty:
        return pd.DataFrame(columns=["symbol", "date", "expected_vc"])

    dup_mask = vc_history.duplicated(["symbol", "date"], keep=False)
    if dup_mask.any():
        sample = (
            vc_history.loc[dup_mask, ["symbol", "date"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "vc_history contains duplicated (symbol, date) pairs; the trailing "
            "window would shrink and leak same-day V_C into expected_vc. "
            f"Examples: {sample}"
        )

    out = vc_history.sort_values(["symbol", "date"]).copy()
    out["expected_vc"] = (
        out.groupby("symbol")["vc_shares"]
           .transform(lambda s: s.shift(1).rolling(lookback_days, min_periods=5).mean())
    )
    return out[["symbol", "date", "expected_vc"]]


def same_day_vc_fallback(vc_history: pd.DataFrame) -> pd.DataFrame:
    """Pilot-Fallback: nimmt ``vc_shares`` selbst als ``expected_vc``.

    THESIS_DEVIATION: nutzt zukuenftige Information (das tatsaechliche V_C des
    Tages) als Erwartungswert. Nur fuer Smoke-/Pilot-Runs mit weniger als
    5 Tagen Historie. Im echten Run ``rolling_expected_vc`` verwenden.
    """
    if vc_history.empty:
        return pd.DataFrame(columns=["symbol", "date", "expected_vc"])
    out = vc_history.copy()
    out["expected_vc"] = out["vc_shares"].astype(float)
    return out[["symbol", "date", "expected_vc"]]
