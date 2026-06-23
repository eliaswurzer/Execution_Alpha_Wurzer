"""
state_vector.py -- Konstruktion des State-Vektors X_{t_0} fuer das Cox-PH
Fill-Modell (Thesis §4.2.3).

State-Vektor::

    X_t0 = (q0, D0, OFI, sigma, limit distance, spread, ToD)

wobei

* ``q0`` = Queue-Depth-Proxy (Shares). At the touch this is 0.5 * NBBO-Size
  on the same side, converted from Daily-TAQ round lots to shares via
  ``cfg.NBBO_SIZE_SHARES_PER_LOT``. For deeper limits true queue depth is
  unobserved in TAQ/NBBO, so ``limit_offset_bps`` and spread features
  identify the posting distance.
  [THESIS_DEVIATION: exakt waere OpenBook-Ultra-Queue-Position.]
* ``D0`` = gesamte Depth am Best (Shares). Proxy = NBBO-Size * shares/lot.
* ``OFI`` = aktueller OFI-Z-Score (aus ``microstructure.ofi``).
* ``sigma`` = rolling 5-min realised Volatility (Thesis Eq. 5.3).
* ``ToD`` = Time-of-Day-One-Hot-Encoding fuer Stunden 10..15 (Baseline 9).

Diese Datei liefert Vektoren fuer einen *einzelnen Submission-Zeitpunkt* und
die Hilfsfunktion ``build_event_panel`` fuer die Kalibrierung.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from .. import config as cfg
from ..data.features import compute_realised_volatility
from ..microstructure.ofi import compute_ofi
from ..utils.ticks import effective_limit_offset_bps, snap_limit_to_tick


STATE_COLUMNS = [
    "q0", "D0", "ofi_z", "sigma",
    "limit_offset_bps", "half_spread_bps",
] + [f"tod_{h}" for h in cfg.TOD_HOUR_BINS]


# ---------------------------------------------------------------------------
# Single-point State-Vector
# ---------------------------------------------------------------------------

def state_at(
    ts: pd.Timestamp,
    nbbo: pd.DataFrame,
    ofi: pd.DataFrame,
    rv: pd.Series,
    side: str,
    *,
    limit_offset_bps: float = 0.0,
    nbbo_times: np.ndarray | None = None,
    ofi_times: np.ndarray | None = None,
    rv_times: np.ndarray | None = None,
    bids: np.ndarray | None = None,
    asks: np.ndarray | None = None,
    bid_sizes: np.ndarray | None = None,
    ask_sizes: np.ndarray | None = None,
) -> dict[str, float]:
    """Baut den State-Vektor X_{t_0} zum Zeitpunkt ``ts``.

    Uses binary search (searchsorted) on pre-sorted arrays for O(log n) lookups.
    When the flat quote arrays (``bids``/``asks``/``bid_sizes``/``ask_sizes``,
    aligned with ``nbbo_times``) are provided, the per-call pandas row access
    is skipped entirely (hot-path optimisation).
    """
    ts_ns = ts.value  # nanoseconds since epoch for searchsorted

    # NBBO snapshot: last row with time <= ts
    limit_offset_bps = max(0.0, float(limit_offset_bps))
    half_spread_bps = np.nan

    use_quote_arrays = (
        bids is not None and asks is not None
        and bid_sizes is not None and ask_sizes is not None
        and len(bids) == len(nbbo)
    )

    if nbbo.empty:
        q0 = D0 = np.nan
    else:
        nbbo_arr = nbbo_times if nbbo_times is not None else nbbo["time"].values.astype("int64")
        idx = int(np.searchsorted(nbbo_arr, ts_ns, side="right")) - 1
        if idx < 0:
            q0 = D0 = np.nan
        else:
            if use_quote_arrays:
                bid = float(bids[idx])
                ask = float(asks[idx])
                bid_size = float(bid_sizes[idx])
                ask_size = float(ask_sizes[idx])
            else:
                last = nbbo.iloc[idx]
                bid = float(last.get("best_bid", np.nan))
                ask = float(last.get("best_offer", np.nan))
                bid_size = float(last.get("best_bid_size", np.nan))
                ask_size = float(last.get("best_offer_size", np.nan))
            mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else np.nan
            if np.isfinite(mid) and mid > 0 and ask >= bid:
                spread = ask - bid
                half_spread_bps = (spread / 2.0) / mid * 1e4
            # NBBO sizes are Daily-TAQ round lots; the state vector is
            # share-denominated (thesis notation, matches trade volumes).
            same_side_lots = bid_size if side == "BUY" else ask_size
            D0 = same_side_lots * cfg.NBBO_SIZE_SHARES_PER_LOT
            q0 = 0.5 * D0 if np.isfinite(D0) else np.nan

    # OFI-Z-Score: last bucket with timestamp <= ts
    if ofi is None or ofi.empty:
        ofi_z = 0.0
    else:
        ofi_arr = ofi_times if ofi_times is not None else ofi["timestamp"].values.astype("int64")
        idx = int(np.searchsorted(ofi_arr, ts_ns, side="right")) - 1
        if idx < 0:
            ofi_z = 0.0
        else:
            ofi_z = float(ofi["ofi_zscore"].iloc[idx])
            if side == "SELL":
                ofi_z = -ofi_z

    # Volatility: last 5-min bucket <= ts
    if rv is None or rv.empty:
        sigma = np.nan
    else:
        rv_arr = rv_times if rv_times is not None else rv.index.values.astype("int64")
        idx = int(np.searchsorted(rv_arr, ts_ns, side="right")) - 1
        sigma = float(rv.iloc[idx]) if idx >= 0 else np.nan

    # ToD one-hot
    hour = ts.hour
    tod = {f"tod_{h}": float(h == hour) for h in cfg.TOD_HOUR_BINS}

    return {
        "q0": q0,
        "D0": D0,
        "ofi_z": ofi_z,
        "sigma": sigma,
        "limit_offset_bps": limit_offset_bps,
        "half_spread_bps": half_spread_bps,
        **tod,
    }


# ---------------------------------------------------------------------------
# Event-Panel fuer Cox-PH Kalibrierung
# ---------------------------------------------------------------------------

def build_event_panel(
    nbbo: pd.DataFrame,
    trades: pd.DataFrame,
    symbol: str,
    date: _dt.date,
    horizon_seconds: int = cfg.FILL_MODEL_HORIZON_SECONDS,
    sample_every_seconds: int = cfg.REFRESH_SECONDS_DEFAULT,
    offset_grid_bps: tuple[float, ...] = cfg.FILL_MODEL_OFFSET_GRID_BPS,
    max_rows: int | None = None,
    sample_seed: int | None = None,
) -> pd.DataFrame:
    """Sampelt synthetische "passive limit order events" fuer Training.

    Fuer jeden Zeitpunkt t0 auf einem ``sample_every_seconds``-Grid werden fuer
    mehrere passive Abstaende vom Touch BUY- und SELL-Events mit State-Vektor und
    duration/event-Flag erzeugt:

    * ``duration`` = min(Zeit bis ein Trade am Limit oder darueber erscheint,
                       horizon_seconds)
    * ``event``    = 1 wenn ein Fill-Signal innerhalb ``horizon_seconds`` stattfand,
                     sonst 0 (right-censored)

    Horizont und Grid-Schrittweite sind per Default identisch
    (``cfg.FILL_MODEL_HORIZON_SECONDS == cfg.REFRESH_SECONDS_DEFAULT``). Dadurch
    sind aufeinanderfolgende Beobachtungsfenster eines Offset/Side disjunkt
    ``(t0, t0+H]`` -> ``(t0+H, t0+2H]``, was die Unabhaengigkeitsannahme der
    Cox-Partial-Likelihood respektiert (vgl. Thesis-Anmerkung A5). Der Horizont
    entspricht zugleich der Slice-Lebensdauer im Simulations-Engine, sodass das
    Survival-Modell genau dort kalibriert wird, wo es abgefragt wird.
    """
    if nbbo.empty or trades.empty:
        return pd.DataFrame(
            columns=["symbol", "date", "side", "t0", "limit_price", "duration", "event", *STATE_COLUMNS],
        )

    ofi = compute_ofi(
        nbbo,
        bucket_seconds=cfg.OFI_WINDOW_SECONDS,
        zscore_window=cfg.OFI_ZSCORE_WINDOW_BUCKETS,
        zscore_mode="rolling",
    )
    rv = compute_realised_volatility(nbbo)

    day_start = pd.Timestamp.combine(date, cfg.RTH_OPEN)
    day_end = pd.Timestamp.combine(date, cfg.MOC_CUTOFF)
    # Grid
    grid = pd.date_range(day_start, day_end, freq=f"{sample_every_seconds}s")
    grid = grid[grid < day_end]
    grid = grid[grid.hour >= 10]   # ersten 30 min ueberspringen

    trades_sorted = trades.sort_values("time")[["time", "price"]].reset_index(drop=True)
    nbbo_sorted = nbbo.sort_values("time")[[
        "time", "best_bid", "best_offer", "best_bid_size", "best_offer_size",
    ]].reset_index(drop=True)

    nbbo_times = nbbo_sorted["time"].values.astype("datetime64[ns]").astype("int64")
    trade_times = trades_sorted["time"].values.astype("datetime64[ns]").astype("int64")
    trade_prices = trades_sorted["price"].to_numpy(dtype=float)
    bids = nbbo_sorted["best_bid"].to_numpy(dtype=float)
    asks = nbbo_sorted["best_offer"].to_numpy(dtype=float)
    bid_sizes = nbbo_sorted["best_bid_size"].to_numpy(dtype=float)
    ask_sizes = nbbo_sorted["best_offer_size"].to_numpy(dtype=float)
    grid_ns = grid.values.astype("datetime64[ns]").astype("int64")
    nbbo_idx = np.searchsorted(nbbo_times, grid_ns, side="right") - 1
    valid = nbbo_idx >= 0
    valid &= np.isfinite(bids[np.maximum(nbbo_idx, 0)])
    valid &= np.isfinite(asks[np.maximum(nbbo_idx, 0)])
    valid &= bids[np.maximum(nbbo_idx, 0)] > 0
    valid &= asks[np.maximum(nbbo_idx, 0)] > 0
    valid &= asks[np.maximum(nbbo_idx, 0)] > bids[np.maximum(nbbo_idx, 0)]
    grid = grid[valid]
    grid_ns = grid_ns[valid]
    nbbo_idx = nbbo_idx[valid]

    if len(grid) == 0:
        return pd.DataFrame(
            columns=["symbol", "date", "side", "t0", "limit_price", "duration", "event", *STATE_COLUMNS],
        )

    offsets = tuple(max(0.0, float(x)) for x in offset_grid_bps)
    total_candidates = len(grid) * len(offsets) * 2
    selected_ordinals: set[int] | None = None
    if max_rows is not None and max_rows > 0 and total_candidates > max_rows:
        rng = np.random.default_rng(sample_seed if sample_seed is not None else cfg.DEFAULT_SEED)
        selected_ordinals = set(
            int(x) for x in rng.choice(total_candidates, size=int(max_rows), replace=False)
        )

    ofi_times = None if ofi is None or ofi.empty else ofi["timestamp"].values.astype("datetime64[ns]").astype("int64")
    rv_times = None if rv is None or rv.empty else rv.index.values.astype("datetime64[ns]").astype("int64")

    rows = []
    horizon_ns = int(horizon_seconds * 1_000_000_000)
    moc_cutoff_ns = int(day_end.value)
    row_ordinal = 0
    for t0, t0_ns, q_idx in zip(grid, grid_ns, nbbo_idx):
        bid = float(bids[q_idx])
        ask = float(asks[q_idx])
        t_end_ns = int(min(t0_ns + horizon_ns, moc_cutoff_ns))
        if t_end_ns <= t0_ns:
            continue
        tr_start = int(np.searchsorted(trade_times, t0_ns, side="right"))
        tr_end = int(np.searchsorted(trade_times, t_end_ns, side="right"))
        win_prices = trade_prices[tr_start:tr_end]
        win_times = trade_times[tr_start:tr_end]

        buy_sv = state_at(
            t0, nbbo_sorted, ofi, rv, "BUY",
            limit_offset_bps=0.0, nbbo_times=nbbo_times,
            ofi_times=ofi_times, rv_times=rv_times,
            bids=bids, asks=asks, bid_sizes=bid_sizes, ask_sizes=ask_sizes,
        )
        sell_sv = state_at(
            t0, nbbo_sorted, ofi, rv, "SELL",
            limit_offset_bps=0.0, nbbo_times=nbbo_times,
            ofi_times=ofi_times, rv_times=rv_times,
            bids=bids, asks=asks, bid_sizes=bid_sizes, ask_sizes=ask_sizes,
        )

        for offset_bps in offsets:
            for side, base_state in (("BUY", buy_sv), ("SELL", sell_sv)):
                include = selected_ordinals is None or row_ordinal in selected_ordinals
                row_ordinal += 1
                if not include:
                    continue
                if side == "BUY":
                    raw_limit_price = bid - (offset_bps / 1e4) * bid
                    limit_price = snap_limit_to_tick(raw_limit_price, side)
                    effective_offset = effective_limit_offset_bps(bid, limit_price, side)
                    hit_idx = np.flatnonzero(win_prices <= limit_price)
                else:
                    raw_limit_price = ask + (offset_bps / 1e4) * ask
                    limit_price = snap_limit_to_tick(raw_limit_price, side)
                    effective_offset = effective_limit_offset_bps(ask, limit_price, side)
                    hit_idx = np.flatnonzero(win_prices >= limit_price)

                if hit_idx.size:
                    duration = (int(win_times[int(hit_idx[0])]) - int(t0_ns)) / 1e9
                    event = 1
                else:
                    # Right-censored at the observation-window end. For late-day
                    # samples that end is the MOC cutoff (t_end_ns is clamped to
                    # the cutoff above), not the full horizon, so a slice posted a
                    # few seconds before the cutoff is censored at that short
                    # observable time rather than recorded as a full 30s survival.
                    duration = float((int(t_end_ns) - int(t0_ns)) / 1e9)
                    event = 0

                sv = dict(base_state)
                sv["limit_offset_bps"] = float(effective_offset)
                rows.append({
                    "symbol": symbol,
                    "date": date,
                    "side": side,
                    "t0": t0,
                    "limit_price": float(limit_price),
                    "duration": float(duration),
                    "event": int(event),
                    **sv,
                })

    return pd.DataFrame(rows)
