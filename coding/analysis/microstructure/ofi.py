"""
ofi.py -- Best-Level Order Flow Imbalance (OFI) nach Cont-Kukanov-Stoikov
(2014), siehe Thesis §3.5, Eq. (eq:ofi).

Die OFI-Event-Kontributionen sind::

    ofi_e = 1[P_B^e >= P_B^{e-1}] * Q_B^e  -  1[P_B^e <  P_B^{e-1}] * Q_B^{e-1}
            - 1[P_A^e <= P_A^{e-1}] * Q_A^e  +  1[P_A^e >  P_A^{e-1}] * Q_A^{e-1}

Hier wird eine mathematisch aequivalente, aber numpy-vektorisierte Form
implementiert.

Die Bucket-OFI wird im Headline-Code kausal auf rechts-geschlossenen Buckets
berechnet: ein Bucket mit Timestamp ``t`` enthaelt nur Events bis einschliesslich
``t``. Der Default-Z-Score ist rolling/kausal; ``zscore_mode="symbol_day"``
bleibt nur fuer Replikation aelterer Pilot-Artefakte verfuegbar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Event-level OFI
# ---------------------------------------------------------------------------

def event_level_ofi(nbbo: pd.DataFrame) -> pd.Series:
    """Pro-NBBO-Snapshot OFI-Beitrag (erste Zeile = 0).

    Erwartet Spalten ``best_bid``, ``best_offer``, ``best_bid_size``,
    ``best_offer_size``.
    """
    if nbbo.empty:
        return pd.Series([], dtype=float, index=nbbo.index)

    bid = nbbo["best_bid"].to_numpy(dtype=float)
    ask = nbbo["best_offer"].to_numpy(dtype=float)
    bsz = nbbo["best_bid_size"].to_numpy(dtype=float)
    asz = nbbo["best_offer_size"].to_numpy(dtype=float)

    # Laut Thesis: positive Bid-Seite = mehr Buy-Pressure, positive Ask-Seite
    # (= hoehere Asks oder groesserer Ask-Depth Abzug) = mehr Sell-Pressure.
    d_bid = np.zeros_like(bid)
    d_ask = np.zeros_like(ask)

    # Bid side
    bid_up = bid[1:] > bid[:-1]
    bid_eq = bid[1:] == bid[:-1]
    bid_dn = bid[1:] < bid[:-1]
    d_bid[1:] = np.where(
        bid_up, bsz[1:],
        np.where(bid_eq, bsz[1:] - bsz[:-1], -bsz[:-1])
    )
    # Ask side (inverse sign convention)
    ask_dn = ask[1:] < ask[:-1]
    ask_eq = ask[1:] == ask[:-1]
    ask_up = ask[1:] > ask[:-1]
    d_ask[1:] = np.where(
        ask_dn, asz[1:],
        np.where(ask_eq, asz[1:] - asz[:-1], -asz[:-1])
    )
    return pd.Series(d_bid - d_ask, index=nbbo.index)


# ---------------------------------------------------------------------------
# Bucketed OFI
# ---------------------------------------------------------------------------

def compute_ofi(
    nbbo: pd.DataFrame,
    bucket_seconds: int = 60,
    zscore_window: int = 60,
    time_col: str = "time",
    zscore_mode: str = "rolling",
) -> pd.DataFrame:
    """Aggregiert Event-OFI auf feste Buckets und bildet einen Z-Score.

    Erwartet NBBO-Daten *fuer einen einzelnen (Symbol, Tag)*. Headline-
    Simulation und Kalibrierung verwenden ``zscore_mode="rolling"``, damit
    keine Information aus spaeteren Tages-Buckets in Entscheidungen vor dem
    Close eingeht.

    Parameters
    ----------
    nbbo : pd.DataFrame
        Spalten ``time``, ``best_bid``, ``best_offer``, ``best_bid_size``,
        ``best_offer_size``. Erwartet einen einzelnen (Symbol, Tag).
    bucket_seconds : int
        Aggregationsbreite (Default 60 s, Thesis §3.5).
    zscore_window : int
        Rolling-Window-Groesse (nur relevant fuer ``zscore_mode="rolling"``).
    zscore_mode : {"rolling", "symbol_day"}
        ``"rolling"`` (Default): kausaler Rolling-Z-Score mit
        ``zscore_window`` Buckets Look-Back. ``"symbol_day"``: z-Score mit
        Mittelwert und Std ueber den gesamten Tag; nur fuer Legacy-Audits.

    Returns
    -------
    DataFrame mit Spalten ``timestamp``, ``ofi``, ``ofi_cumulative``,
    ``ofi_zscore``.
    """
    if nbbo.empty:
        return pd.DataFrame(columns=["timestamp", "ofi", "ofi_cumulative", "ofi_zscore"])

    df = nbbo.sort_values(time_col).copy()
    df["_ofi_raw"] = event_level_ofi(df)
    freq = f"{bucket_seconds}s"
    # Right-edge labels: the value timestamped 15:30:00 contains only updates
    # in (15:29:30, 15:30:00], so state_at(15:30:00) cannot see future updates.
    df["_bucket"] = df[time_col].dt.ceil(freq)

    bucketed = (
        df.groupby("_bucket", sort=True)["_ofi_raw"].sum().reset_index()
          .rename(columns={"_bucket": "timestamp", "_ofi_raw": "ofi"})
    )

    full = pd.date_range(
        bucketed["timestamp"].min(),
        bucketed["timestamp"].max(),
        freq=freq,
    )
    bucketed = (
        bucketed.set_index("timestamp").reindex(full, fill_value=0.0)
                .rename_axis("timestamp").reset_index()
    )

    bucketed["ofi_cumulative"] = bucketed["ofi"].cumsum()

    if zscore_mode == "symbol_day":
        day_mean = bucketed["ofi"].mean()
        day_std = bucketed["ofi"].std()
        if day_std == 0 or np.isnan(day_std):
            bucketed["ofi_zscore"] = 0.0
        else:
            bucketed["ofi_zscore"] = (bucketed["ofi"] - day_mean) / day_std
    elif zscore_mode == "rolling":
        roll_mean = bucketed["ofi"].rolling(zscore_window, min_periods=1).mean()
        roll_std = bucketed["ofi"].rolling(zscore_window, min_periods=2).std()
        bucketed["ofi_zscore"] = (
            (bucketed["ofi"] - roll_mean) / roll_std.replace(0.0, np.nan)
        ).fillna(0.0)
    else:
        raise ValueError(
            f"zscore_mode must be 'symbol_day' or 'rolling', got {zscore_mode!r}"
        )

    return bucketed
