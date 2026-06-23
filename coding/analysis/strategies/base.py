"""
base.py -- Abstract Base fuer Execution-Strategien (Thesis §4.3).

Jede Strategie mapped (parent_order, market_state, fill_model) -> FillResult.
Die Simulations-Mechanik (Refresh-Grid, MOC-Residual-Routing) lebt in der
Basisklasse; konkrete Strategien implementieren ``limit_offset_bps`` fuer
die dynamische Preisberechnung.
"""

from __future__ import annotations

import datetime as _dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config as cfg
from ..data.features import compute_realised_volatility
from ..fill_model.state_vector import state_at
from ..microstructure.imbalance import compute_auction_imbalance_proxy
from ..microstructure.ofi import compute_ofi
from ..utils.ticks import effective_limit_offset_bps, snap_limit_to_tick


def _snap_limit_to_tick(limit_price: float, side: str) -> float:
    """Snap a model limit price onto the penny grid in the passive direction.

    Reg NMS Rule 612 forbids sub-penny limit prices for stocks >= $1 (the
    universe filters at >= $5), so an unsnapped ``bid * (1 - offset/1e4)``
    is not a placeable order. BUY snaps down, SELL snaps up — never more
    aggressive than the model offset. The tiny epsilon (in tick units)
    protects exactly-on-grid prices from float noise without ever moving a
    price across a full tick. Disabled via ``cfg.SNAP_LIMIT_TO_TICK``.
    """
    return snap_limit_to_tick(limit_price, side)


# ---------------------------------------------------------------------------
# Shared result containers
# ---------------------------------------------------------------------------

@dataclass
class FillResult:
    """Aggregiertes Execution-Ergebnis eines Parent-Orders."""
    order_id: str
    symbol: str
    date: _dt.date
    side: str
    strategy: str
    window: str
    qty_intended: int
    qty_filled_passive: int
    qty_filled_moc: int
    vwap_passive: float      # volumenbew. Preis der passiven Fills (NaN wenn 0 Fills)
    close_price: float       # P_C
    avg_fill_price: float    # blended mit MOC-Residual
    fill_rate: float         # passive / intended in [0,1]
    adverse_selection_bps: float = 0.0  # side-signed post-fill mid drift
    detail_fills: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Market state bundle
# ---------------------------------------------------------------------------

@dataclass
class MarketState:
    """Vor-berechnete Feature-Views fuer einen Symbol-Tag."""
    symbol: str
    date: _dt.date
    nbbo: pd.DataFrame
    trades: pd.DataFrame
    close_price: float
    close_volume: float
    ofi: pd.DataFrame                # aus microstructure.ofi.compute_ofi
    rv: pd.Series                    # 5-min realised vol Zeitserie
    imbalance: pd.DataFrame          # aus microstructure.imbalance.compute_auction_imbalance_proxy
    # Pre-computed sorted views for O(log n) lookups — built once in build()
    nbbo_times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="int64"))
    nbbo_mid: pd.DataFrame = field(default_factory=pd.DataFrame)
    ofi_times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="int64"))
    imbalance_times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="int64"))
    rv_times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="int64"))
    # Quote columns as flat float arrays aligned with nbbo_times. The
    # per-interval simulation loop indexes these directly; a pandas
    # .iloc row access per refresh interval costs ~100x more.
    # NOTE: bid_sizes/ask_sizes are in Daily-TAQ round lots (x100 shares).
    bids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    asks: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    bid_sizes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    ask_sizes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    mids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

    @classmethod
    def build(cls, symbol: str, date: _dt.date, trades: pd.DataFrame, nbbo: pd.DataFrame,
              close_price: float, close_volume: float) -> "MarketState":
        nbbo_sorted = nbbo.sort_values("time").reset_index(drop=True)
        ofi_df = compute_ofi(
            nbbo_sorted,
            bucket_seconds=cfg.OFI_WINDOW_SECONDS,
            zscore_window=cfg.OFI_ZSCORE_WINDOW_BUCKETS,
            zscore_mode="rolling",
        )
        rv_series = compute_realised_volatility(nbbo_sorted)
        imb_df = compute_auction_imbalance_proxy(nbbo_sorted)

        nbbo_times = nbbo_sorted["time"].values.astype("int64")
        nbbo_mid = nbbo_sorted[["time", "mid"]].copy()

        ofi_times = ofi_df["timestamp"].values.astype("int64") if not ofi_df.empty else np.empty(0, dtype="int64")
        imb_times = imb_df["time"].values.astype("int64") if not imb_df.empty else np.empty(0, dtype="int64")
        rv_times = rv_series.index.values.astype("int64") if not rv_series.empty else np.empty(0, dtype="int64")

        def _col(name: str) -> np.ndarray:
            if name in nbbo_sorted.columns:
                return nbbo_sorted[name].to_numpy(dtype=float)
            return np.full(len(nbbo_sorted), np.nan)

        return cls(
            symbol=symbol,
            date=date,
            trades=trades,
            nbbo=nbbo_sorted,
            close_price=close_price,
            close_volume=close_volume,
            ofi=ofi_df,
            rv=rv_series,
            imbalance=imb_df,
            nbbo_times=nbbo_times,
            nbbo_mid=nbbo_mid,
            ofi_times=ofi_times,
            imbalance_times=imb_times,
            rv_times=rv_times,
            bids=_col("best_bid"),
            asks=_col("best_offer"),
            bid_sizes=_col("best_bid_size"),
            ask_sizes=_col("best_offer_size"),
            mids=_col("mid"),
        )


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------

class ExecutionStrategy(ABC):
    """Gemeinsames Simulations-Gerust fuer S1/S2/S3.

    ``S0`` ist trivial und hat eine eigene, kompaktere Implementierung
    in ``moc.py``.
    """

    name: str = "BASE"

    def __init__(self, refresh_seconds: int = cfg.REFRESH_SECONDS_DEFAULT):
        self.refresh_seconds = refresh_seconds

    # ---- pricing hook (subclass impl) -----------------------------------

    @abstractmethod
    def limit_offset_bps(
        self,
        t: pd.Timestamp,
        side: str,
        state: MarketState,
        sigma_bar: float,
        delta_max_bps: float,
    ) -> float:
        """Gewuenschter Preisabstand *vom Touch weg* in Basispunkten.

        Fuer BUY: Limit = best_bid - offset_bps/1e4 * best_bid (mehr offset =
        weiter vom Mid weg = passiver).
        Fuer SELL: Limit = best_ask + offset_bps/1e4 * best_ask.
        """

    def slice_size(
        self,
        t: pd.Timestamp,
        cutoff: pd.Timestamp,
        qty_remaining: int,
        side: str,
        state: MarketState,
    ) -> int:
        """Quantity posted for one refresh interval.

        The default is the TWAP carry-forward schedule used by S1-S3.
        Quantity-schedule variants override this hook while sharing the same
        fill, residual-routing, and adverse-selection path below.
        """
        intervals_remaining = max(
            1, int((cutoff - t).total_seconds() / self.refresh_seconds),
        )
        return max(1, qty_remaining // intervals_remaining)

    # ---- full simulation ------------------------------------------------

    def simulate(
        self,
        order: pd.Series,
        state: MarketState,
        fill_model,
        sigma_bar: float,
        delta_max_bps: float,
        *,
        max_slice_shares: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> FillResult:
        rng = rng or np.random.default_rng(cfg.DEFAULT_SEED)
        side = order["side"]
        qty = int(order["qty"])
        arrival = pd.Timestamp(order["arrival_time"])
        cutoff = pd.Timestamp(order["moc_cutoff"])

        qty_remaining = qty
        passive_fills: list[dict] = []
        t = arrival
        refresh = pd.Timedelta(seconds=self.refresh_seconds)
        horizon = float(self.refresh_seconds)

        # Detect tape-replay fill model once before the loop to avoid
        # repeated isinstance/hasattr checks inside the hot path.
        use_tape_replay = hasattr(fill_model, "fill_event")
        # Flat quote arrays from MarketState.build; states constructed by hand
        # (tests, smoke scripts) may lack them and fall back to row access.
        use_quote_arrays = len(state.bids) == len(state.nbbo) and len(state.bids) > 0

        while t < cutoff and qty_remaining > 0:
            idx = int(np.searchsorted(state.nbbo_times, t.value, side="right")) - 1
            if idx < 0:
                t += refresh
                continue
            if use_quote_arrays:
                bid = float(state.bids[idx]); ask = float(state.asks[idx])
            else:
                last = state.nbbo.iloc[idx]
                bid = float(last["best_bid"]); ask = float(last["best_offer"])
            if bid <= 0 or ask <= 0 or ask <= bid:
                t += refresh
                continue

            offset_bps = self.limit_offset_bps(t, side, state, sigma_bar, delta_max_bps)
            half_spread_bps = (ask - bid) / ((ask + bid) / 2) * 1e4 / 2
            # Positive offsets are already posted away from the touch
            # (BUY below bid, SELL above ask). Do not cap them at the spread;
            # doing so turns wide passive orders into near-touch orders.
            offset_bps = max(0.0, float(offset_bps))

            if side == "BUY":
                limit_price = bid - (offset_bps / 1e4) * bid
                touch_price = bid
            else:
                limit_price = ask + (offset_bps / 1e4) * ask
                touch_price = ask
            limit_price = _snap_limit_to_tick(limit_price, side)
            effective_offset_bps = effective_limit_offset_bps(touch_price, limit_price, side)

            # Displayed same-side depth at the touch is the queue ahead of a
            # newly posted order joining that level (Daily TAQ NBBO sizes are
            # round lots); a limit away from the touch opens a new visible
            # level, so the displayed queue ahead is zero. Only the "queue"
            # tape-replay rule consumes this estimate.
            if side == "BUY":
                same_side_size = (
                    float(state.bid_sizes[idx]) if use_quote_arrays
                    else float(last.get("best_bid_size", np.nan))
                )
                at_touch = abs(limit_price - bid) < cfg.TICK_SIZE / 2
            else:
                same_side_size = (
                    float(state.ask_sizes[idx]) if use_quote_arrays
                    else float(last.get("best_offer_size", np.nan))
                )
                at_touch = abs(limit_price - ask) < cfg.TICK_SIZE / 2
            if at_touch and np.isfinite(same_side_size):
                queue_ahead = same_side_size * cfg.NBBO_SIZE_SHARES_PER_LOT
            else:
                queue_ahead = 0.0

            # Clamp the fill window at the MOC cutoff so a final interval can
            # never consume tape prints after the residual routes to MOC.
            t_end = min(t + refresh, cutoff)
            actual_horizon_seconds = max(0.0, float((t_end - t).total_seconds()))
            fill_time = t + (t_end - t) / 2
            available_fill_qty: float | None = None
            if use_tape_replay:
                # Tape replay checks realized lit trades in (t, t_end] against
                # our posted limit. The model returns the first tape timestamp
                # that satisfies the fill rule, plus an optional volume cap for
                # queue-/volume-aware specs; any haircut probability is sampled
                # below by the per-order RNG.
                if hasattr(fill_model, "fill_event_details"):
                    p_fill, tape_fill_time, available_fill_qty = fill_model.fill_event_details(
                        t, t_end, limit_price, side, queue_ahead=queue_ahead,
                    )
                else:
                    p_fill, tape_fill_time = fill_model.fill_event(
                        t, t_end, limit_price, side,
                    )
                if tape_fill_time is not None:
                    fill_time = tape_fill_time
            else:
                # Model-based specs (Cox-PH, KM, XGB, touch proxies) do not
                # inspect realized trades in this interval. Build the causal
                # state vector at submission time; fill_probability(...) maps it
                # to a probability over the actual remaining interval before
                # the MOC cutoff, sampled below.
                if actual_horizon_seconds <= 0:
                    p_fill = 0.0
                else:
                    sv = state_at(
                        t, state.nbbo, state.ofi, state.rv, side,
                        limit_offset_bps=effective_offset_bps,
                        nbbo_times=state.nbbo_times,
                        ofi_times=state.ofi_times,
                        rv_times=state.rv_times,
                        bids=state.bids if use_quote_arrays else None,
                        asks=state.asks if use_quote_arrays else None,
                        bid_sizes=state.bid_sizes if use_quote_arrays else None,
                        ask_sizes=state.ask_sizes if use_quote_arrays else None,
                    )
                    # Submission timestamp for time-to-cutoff-stratified models
                    # (KM). Cox/XGB select only their covariate columns and
                    # ignore this key.
                    sv["t0"] = t
                    try:
                        p_fill = float(fill_model.fill_probability(actual_horizon_seconds, sv))
                    except Exception:
                        p_fill = 0.0
            p_fill = float(np.clip(p_fill, 0.0, 1.0))

            slice_size = self.slice_size(t, cutoff, qty_remaining, side, state)
            if max_slice_shares is not None:
                slice_size = min(slice_size, max_slice_shares)
            if slice_size <= 0:
                t += refresh
                continue

            if use_tape_replay:
                # Tape-replay: deterministic given haircut_p already baked in.
                # Still use rng for the haircut draw so results are reproducible.
                filled = p_fill >= 1.0 or (p_fill > 0.0 and rng.random() < p_fill)
            else:
                filled = rng.random() < p_fill

            if filled:
                filled_qty = int(slice_size)
                if available_fill_qty is not None:
                    filled_qty = min(filled_qty, max(0, int(np.floor(available_fill_qty))))
                if filled_qty <= 0:
                    t += refresh
                    continue
                passive_fills.append({
                    "time": fill_time,
                    "price": float(limit_price),
                    "qty": filled_qty,
                    "p_fill": p_fill,
                    "limit_offset_bps": float(offset_bps),
                    "effective_limit_offset_bps": float(effective_offset_bps),
                    "bid": bid,
                    "ask": ask,
                    "half_spread_bps": float(half_spread_bps),
                })
                qty_remaining -= filled_qty
            t += refresh

        filled_passive = int(sum(f["qty"] for f in passive_fills))
        filled_moc = int(max(0, qty - filled_passive))
        if passive_fills:
            tot_q = sum(f["qty"] for f in passive_fills)
            vwap_p = sum(f["price"] * f["qty"] for f in passive_fills) / tot_q
        else:
            vwap_p = float("nan")

        close_price = float(state.close_price) if state.close_price else float("nan")
        if qty > 0 and (filled_passive + filled_moc) > 0:
            num = (vwap_p * filled_passive if filled_passive > 0 else 0.0) + close_price * filled_moc
            avg_fill = num / (filled_passive + filled_moc)
        else:
            avg_fill = float("nan")

        # Per-fill realized adverse selection: side-signed mid-quote drift
        # from fill time to fill time + AS_HORIZON_SECONDS (Thesis eq:as).
        as_bps = 0.0
        if passive_fills and not np.isnan(close_price) and close_price > 0:
            fill_times = pd.to_datetime([f["time"] for f in passive_fills])
            fill_qtys = np.array([f["qty"] for f in passive_fills], dtype=float)
            horizon_ns = int(cfg.AS_HORIZON_SECONDS * 1_000_000_000)

            if len(state.mids) == len(state.nbbo_times) and len(state.mids) > 0:
                # Backward as-of lookup via binary search on the precomputed
                # mid array — same semantics as merge_asof, without the
                # per-order pandas overhead.
                fill_ns = fill_times.values.astype("int64")
                idx_at = np.searchsorted(state.nbbo_times, fill_ns, side="right") - 1
                idx_after = np.searchsorted(
                    state.nbbo_times, fill_ns + horizon_ns, side="right",
                ) - 1
                mid_at = np.where(
                    idx_at >= 0, state.mids[np.maximum(idx_at, 0)], np.nan,
                )
                mid_after = np.where(
                    idx_after >= 0, state.mids[np.maximum(idx_after, 0)], np.nan,
                )
            else:
                horizon = pd.Timedelta(seconds=cfg.AS_HORIZON_SECONDS)
                fills_df = pd.DataFrame({"time": fill_times})
                fills_after_df = pd.DataFrame({"time": fill_times + horizon})
                mid_at = pd.merge_asof(
                    fills_df.sort_values("time"), state.nbbo_mid, on="time", direction="backward",
                )["mid"].to_numpy()
                mid_after = pd.merge_asof(
                    fills_after_df.sort_values("time"), state.nbbo_mid, on="time", direction="backward",
                )["mid"].to_numpy()
            # Fall back to close_price where post-horizon NBBO is unavailable
            mid_after = np.where(np.isnan(mid_after), close_price, mid_after)
            mid_at = np.where(np.isnan(mid_at), close_price, mid_at)

            side_sign_val = 1.0 if side == "BUY" else -1.0
            per_fill_as = side_sign_val * (mid_after - mid_at) / close_price * 1e4
            as_bps = float(np.average(per_fill_as, weights=fill_qtys))

        return FillResult(
            order_id=str(order["order_id"]),
            symbol=state.symbol,
            date=state.date,
            side=side,
            strategy=self.name,
            window=str(order.get("window", "")),
            qty_intended=qty,
            qty_filled_passive=filled_passive,
            qty_filled_moc=filled_moc,
            vwap_passive=vwap_p,
            close_price=close_price,
            avg_fill_price=avg_fill,
            fill_rate=filled_passive / qty if qty > 0 else 0.0,
            adverse_selection_bps=as_bps,
            detail_fills=passive_fills,
        )
