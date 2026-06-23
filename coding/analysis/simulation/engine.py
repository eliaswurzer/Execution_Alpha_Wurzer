"""
engine.py -- Orchestriert Strategy x Symbol x Date x Size x Window.

Ein Run produziert ein Long-Format DataFrame mit einer Zeile pro
(order_id, strategy). Diese Zeilen werden in ``metrics.attach_alpha_columns``
in Alpha und Net-Alpha umgewandelt.

Fill-Spezifikationen, ueber ``fill_specification`` waehlbar. Der finale
Headline-Run verwendet ``tape_replay_queue``: eine queue-aware Replay-Logik
gegen das gefilterte lit Trade Tape. Cox/KM/XGB und die Touch-Proxies bleiben
als modellbasierte bzw. stilisierte Robustheitsspezifikationen verfuegbar.

* ``tape_replay_queue``        -- Headline: Trades durch den Limitpreis fillen;
                                  Trades genau am Limit fillen erst nach
                                  Verbrauch der angezeigten Same-Side-Queue.
* ``tape_replay_strict``       -- konservativer Replay-Bound: Fill nur bei
                                  Trades strikt durch den Limitpreis.
* ``tape_replay``              -- optimistischer Replay-Bound: Fill bei Trades
                                  am oder durch den Limitpreis.
* ``cox``                      -- Cox-PH Survival-Modell.
* ``km``                       -- empirisches Kaplan-Meier-Modell.
* ``xgb``                      -- XGBoost-basiertes Survival-Modell.
* ``infinite_depth``           -- Touch-deterministisch: fillt 1.0 wenn der
                                  Markt im Refresh-Intervall den Limit-Preis
                                  erreicht; sonst 0.
* ``infinite_depth_haircut``   -- wie ``infinite_depth``, aber Fill-Wahr-
                                  scheinlichkeit 0.5 wenn erreicht (Approx.
                                  fuer Queue-Position-Effekte ohne Modell).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import logging
from dataclasses import asdict
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .. import config as cfg
from ..data.taq_loader import (
    ClosingAuction,
    extract_closing_auction_details,
    filter_regular_hours,
    filter_trades_near_quotes,
    filter_valid_quotes,
    filter_valid_trades,
    load_symbol_day,
)
from ..strategies import MarketState, get_strategy
from ..strategies.base import FillResult

log = logging.getLogger(__name__)


FillSpec = Literal["cox", "km", "infinite_depth", "infinite_depth_haircut",
                   "tape_replay", "tape_replay_haircut",
                   "tape_replay_volume", "tape_replay_volume_haircut",
                   "tape_replay_strict", "tape_replay_queue", "xgb"]


# ---------------------------------------------------------------------------
# Touch-basierte Fill-Approximationen
# ---------------------------------------------------------------------------

class _TouchModel:
    """Predictor mit derselben Schnittstelle wie Cox/KM, der nur prueft, ob
    der Markt im naechsten Refresh-Intervall den Limit-Preis erreicht hat.

    ``haircut_p`` < 1.0 modelliert Queue-Position-Effekte stilisiert.
    """

    def __init__(self, haircut_p: float = 1.0):
        self.haircut_p = float(haircut_p)

    def survival(self, horizon_seconds: float, x) -> float:
        return 1.0 - self.haircut_p

    def fill_probability(self, horizon_seconds: float, x) -> float:
        return self.haircut_p


class _TouchDispatcher:
    """API-kompatibel zu ``TieredFillModel`` mit konstantem Touch-Modell."""

    def __init__(self, haircut_p: float = 1.0):
        self._model = _TouchModel(haircut_p=haircut_p)

    def for_symbol(self, symbol: str):  # noqa: ARG002
        return self._model


# ---------------------------------------------------------------------------
# Tape-Replay Fill Model
# ---------------------------------------------------------------------------

TAPE_FILL_RULES = ("at_or_through", "strict_through", "queue")


class TapeReplayModel:
    """Deterministic fill check against the realized lit-trade tape.

    A passive order resting at ``limit_price`` is checked in cadence interval
    (t_start, t_end] against lit-exchange trades. Three fill rules implement
    the standard bounds bracket from the limit-order backtesting literature
    (Harris & Hasbrouck 1996; Bacidore, Battalio & Jennings 2003;
    Lo, MacKinlay & Zhang 2002):

      - ``at_or_through`` (optimistic upper bound): BUY at L filled if any
        trade printed at or below L; ignores queue priority entirely, which
        saturates fill rates near the touch in liquid names.
      - ``strict_through`` (conservative lower bound): filled only if a trade
        printed strictly through the limit (BUY: price < L by more than half
        a tick). Justified by Reg-NMS trade-through protection: a lit print
        below a displayed bid implies the displayed bid executed first.
      - ``queue`` (mechanical volume-ahead model): trades through the limit
        always fill; trades exactly at the limit fill only once their
        cumulative volume exceeds ``queue_ahead`` — the displayed same-side
        depth (in shares) at posting time. Fillable quantity is
        ``vol_through + max(0, vol_at - queue_ahead)`` times the
        participation rate, so the queue rule is inherently volume-aware.

    The ``trades`` input must already be filtered (``filter_valid_trades`` +
    ``filter_trades_near_quotes``). The model additionally excludes configured
    non-lit venue buckets before checking passive fills.
    Using the trade tape instead of the NBBO quote stream avoids the quote-bounce
    problem: a one-tick bid movement every 30 seconds in a mega-cap would
    trivially trigger an NBBO-based touch check even when no trade occurs at
    the limit, producing artificially high fill rates (~99%).

    ``haircut_p`` approximates queue-position effects stylistically: even when
    the rule signals a fill, the order fills with probability ``haircut_p``
    (legacy bound; the ``queue`` rule supersedes it with a mechanical model).

    The ``fill_probability`` method is intentionally absent; strategies/base.py
    detects this class via duck-typing on ``fill_event``.
    """

    def __init__(
        self,
        trades: pd.DataFrame,
        haircut_p: float = 1.0,
        *,
        volume_cap: bool = False,
        volume_participation: float = cfg.TAPE_REPLAY_VOLUME_PARTICIPATION,
        fill_rule: str = "at_or_through",
    ):
        if fill_rule not in TAPE_FILL_RULES:
            raise ValueError(
                f"fill_rule must be one of {TAPE_FILL_RULES}, got {fill_rule!r}"
            )
        tape = trades
        if "exchange" in tape.columns and cfg.TAPE_REPLAY_EXCLUDED_EXCHANGES:
            exch = tape["exchange"].astype(str).str.strip().str.upper()
            tape = tape[~exch.isin(cfg.TAPE_REPLAY_EXCLUDED_EXCHANGES)]
        keep_cols = ["time", "price"]
        if "volume" in tape.columns:
            keep_cols.append("volume")
        self._trades = tape[keep_cols].sort_values("time").reset_index(drop=True)
        self._times = self._trades["time"].values.astype("int64")
        self._prices = self._trades["price"].to_numpy(dtype=float)
        if "volume" in self._trades.columns:
            self._volumes = self._trades["volume"].to_numpy(dtype=float)
        else:
            self._volumes = np.ones(len(self._trades), dtype=float)
        self.haircut_p = float(haircut_p)
        self.volume_cap = bool(volume_cap)
        self.volume_participation = float(volume_participation)
        self.fill_rule = fill_rule
        # Half-tick tolerance separates "at the limit" from "through the
        # limit" under float prices without changing the legacy at-or-through
        # comparison (which stays an exact <=/>= against limit_price).
        self._half_tick = cfg.TICK_SIZE / 2.0

    def fill_event(
        self,
        t_start: pd.Timestamp,
        t_end: pd.Timestamp,
        limit_price: float,
        side: str,
        *,
        queue_ahead: float | None = None,
    ) -> tuple[float, pd.Timestamp | None]:
        """Return fill probability and first tape-cross timestamp."""
        p_fill, fill_time, _ = self.fill_event_details(
            t_start, t_end, limit_price, side, queue_ahead=queue_ahead,
        )
        return p_fill, fill_time

    def fill_event_details(
        self,
        t_start: pd.Timestamp,
        t_end: pd.Timestamp,
        limit_price: float,
        side: str,
        *,
        queue_ahead: float | None = None,
    ) -> tuple[float, pd.Timestamp | None, float | None]:
        """Return fill probability, fill timestamp, and optional volume cap.

        ``queue_ahead`` (shares displayed ahead of us at the limit at posting
        time) is only used by the ``queue`` fill rule; ``None`` degrades to an
        empty queue, which makes the queue rule equivalent to at-or-through.
        """
        emit_volume = self.volume_cap or self.fill_rule == "queue"
        left = int(np.searchsorted(self._times, t_start.value, side="right"))
        right = int(np.searchsorted(self._times, t_end.value, side="right"))
        if left >= right:
            return 0.0, None, 0.0 if emit_volume else None
        prices = self._prices[left:right]

        if side.upper() == "BUY":
            # at-or-through keeps the legacy exact comparison; the half-tick
            # tolerance only splits it into "through" vs. "at the limit".
            touched = prices <= limit_price
            through = prices <= limit_price - self._half_tick
        else:
            touched = prices >= limit_price
            through = prices >= limit_price + self._half_tick

        if self.fill_rule == "strict_through":
            fill_mask = through
            fillable_mask = through
            queue_excess = 0.0
        elif self.fill_rule == "queue":
            at_limit = touched & ~through
            q0 = max(0.0, float(queue_ahead)) if queue_ahead is not None and np.isfinite(queue_ahead) else 0.0
            vols = self._volumes[left:right]
            cum_at = np.cumsum(np.where(at_limit, vols, 0.0))
            # Through prints execute us via price priority; at-limit prints
            # execute us only once the displayed queue ahead is depleted.
            fill_mask = through | (at_limit & (cum_at > q0))
            fillable_mask = through
            queue_excess = max(0.0, float(cum_at[-1]) - q0) if len(cum_at) else 0.0
        else:  # at_or_through (legacy)
            fill_mask = touched
            fillable_mask = touched
            queue_excess = 0.0

        first_pos = int(np.argmax(fill_mask)) if len(fill_mask) else 0
        if not len(fill_mask) or not bool(fill_mask[first_pos]):
            return 0.0, None, 0.0 if emit_volume else None

        available = None
        if emit_volume:
            vols = self._volumes[left:right]
            available = float(
                (float(vols[fillable_mask].sum()) + queue_excess)
                * self.volume_participation
            )
        return (
            self.haircut_p,
            pd.Timestamp(self._times[left + first_pos]),
            available,
        )

    def check_tape(
        self,
        t_start: pd.Timestamp,
        t_end: pd.Timestamp,
        limit_price: float,
        side: str,
    ) -> float:
        """Compatibility wrapper for callers that only need a probability."""
        p_fill, _ = self.fill_event(t_start, t_end, limit_price, side)
        return p_fill


class _TapeReplayDispatcher:
    """Wraps a per-symbol-day TapeReplayModel with the TieredFillModel API."""

    def __init__(
        self,
        trades: pd.DataFrame,
        haircut_p: float = 1.0,
        *,
        volume_cap: bool = False,
        fill_rule: str = "at_or_through",
    ):
        self._model = TapeReplayModel(
            trades, haircut_p=haircut_p, volume_cap=volume_cap,
            fill_rule=fill_rule,
        )

    def for_symbol(self, symbol: str):  # noqa: ARG002
        return self._model


def _resolve_fill_dispatcher(
    fill_specification: FillSpec, cox_dispatcher, km_dispatcher,
):
    """Resolve all non-tape specs.  Tape specs are handled in simulate_symbol_day
    after trades are loaded."""
    if fill_specification in ("cox", "xgb"):
        return cox_dispatcher  # caller passes the appropriate model (Cox or XGB)
    if fill_specification == "km":
        if km_dispatcher is None:
            raise ValueError("fill_specification='km' braucht km_dispatcher")
        return km_dispatcher
    if fill_specification == "infinite_depth":
        return _TouchDispatcher(haircut_p=1.0)
    if fill_specification == "infinite_depth_haircut":
        return _TouchDispatcher(haircut_p=0.5)
    if fill_specification in (
        "tape_replay", "tape_replay_haircut",
        "tape_replay_volume", "tape_replay_volume_haircut",
        "tape_replay_strict", "tape_replay_queue",
    ):
        raise RuntimeError(
            "tape_replay dispatcher must be created after trades are loaded; "
            "do not call _resolve_fill_dispatcher for tape_replay specs."
        )
    raise ValueError(f"Unknown fill_specification: {fill_specification!r}")


def _model_for_symbol_or_tier(dispatcher, symbol: str, tier: int):
    """Resolve a fill model from the symbol map, falling back to the run tier.

    ``calibrated_plus_fallback`` assigns data-complete symbols missing from the
    calibration map to tier 3 at run time. Persisted model dispatchers may still
    carry only the calibration-time symbol map, so use the already audited run
    tier when the dispatcher has a tier model available.
    """
    try:
        return dispatcher.for_symbol(symbol)
    except KeyError:
        models = getattr(dispatcher, "models", None)
        if isinstance(models, dict):
            try:
                tier_key = int(tier)
            except (TypeError, ValueError):
                tier_key = None
            if tier_key is not None and tier_key in models:
                log.debug(
                    "No symbol-level fill model for %s; using tier %s model",
                    symbol, tier_key,
                )
                return models[tier_key]
        raise


def _causal_sigma_bar(rv: pd.Series, arrival_time: pd.Timestamp) -> float:
    """Mean RV known at ``arrival_time``; never uses later same-day values."""
    if rv is None or rv.empty:
        return 1e-6
    series = pd.to_numeric(rv, errors="coerce").dropna()
    if series.empty:
        return 1e-6
    arrival = pd.Timestamp(arrival_time)
    known = series[series.index <= arrival]
    value = float(known.mean()) if not known.empty else float(series.iloc[0])
    if not np.isfinite(value) or value <= 0:
        return 1e-6
    return value


def _max_slice_from_expected_vc(order: pd.Series) -> int:
    """5% slice cap from trailing expected closing volume, not realized close."""
    expected_vc = float(order.get("expected_vc", 0.0) or 0.0)
    return int(max(1, cfg.MAX_SLICE_FRACTION_OF_VC * max(expected_vc, 0.0)))


def _stable_hash_int(key: str, n_bytes: int = 8) -> int:
    """Deterministic, process-independent integer from a string key."""
    return int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:n_bytes], "little",
    )


def _order_rng(
    day_seed: int, strategy: str, order_id: str,
) -> np.random.Generator:
    """Independent RNG per (symbol-day, strategy, order).

    Using one shared stream seeded with a global constant would (a) reuse the
    identical uniform sequence on every symbol-day, mechanically correlating
    stochastic fill outcomes across the panel, and (b) make a strategy's
    draws depend on which strategies ran before it. Keying the generator on
    (day_seed, strategy, order_id) removes both effects while keeping runs
    bit-reproducible.
    """
    return np.random.default_rng([
        day_seed,
        _stable_hash_int(strategy, 4),
        _stable_hash_int(order_id, 4),
    ])


def _result_to_row(r: FillResult, order: pd.Series) -> dict:
    d = asdict(r)
    detail = d.pop("detail_fills", None) or []
    # First passive fill timestamp (diagnostics: time-to-fill per spec).
    d["first_fill_time"] = min((f["time"] for f in detail), default=pd.NaT)
    d["size_frac"] = order.get("size_frac")
    d["window"] = order.get("window", r.window)
    for col in ("arrival_time", "moc_cutoff", "expected_vc"):
        if col in order:
            d[col] = order[col]
    return d


def _attach_auction_metadata(row: dict, auction: ClosingAuction) -> dict:
    row["close_price_source"] = auction.price_source
    row["close_volume_source"] = auction.volume_source
    row["close_trade_volume"] = auction.close_trade_volume
    row["close_trade_rows"] = auction.close_trade_rows
    row["official_close_marker_volume"] = auction.official_marker_volume
    row["official_close_marker_rows"] = auction.official_marker_rows
    row["official_close_marker_fallback_volume"] = auction.official_marker_fallback_volume
    return row


def simulate_symbol_day(
    symbol: str,
    date: _dt.date,
    parent_orders: pd.DataFrame,
    strategies: Iterable[str],
    fill_model,
    delta_max_bps_by_tier: dict[int, float],
    tier: int,
    s3_params: dict | None = None,
    *,
    seed: int = cfg.DEFAULT_SEED,
    fill_specification: FillSpec = "cox",
    km_model=None,
    tod_schedule=None,
    value_model=None,
    sector: str = "",
    listing_exchange: str = "",
    skip_reason_out: dict | None = None,
) -> pd.DataFrame:
    """Fuehrt alle angefragten Strategien auf einem Symbol-Tag aus.

    Liefert einen DataFrame mit einer Zeile pro (order_id x strategy).
    sigma_bar and ofi_scale are computed internally after data loading to
    avoid the caller having to load the same Parquet files twice.
    When ``skip_reason_out`` is given, an early empty return records its
    structured cause under the ``"reason"`` key so callers can report
    missing-auction days separately from filter-empty days.
    """
    def _skip(reason: str) -> pd.DataFrame:
        if skip_reason_out is not None:
            skip_reason_out["reason"] = reason
        return pd.DataFrame()

    if parent_orders.empty:
        return _skip("no_parent_orders")

    try:
        trades_all, nbbo_all = load_symbol_day(date, symbol, rth_only=False)
    except FileNotFoundError:
        log.debug("Missing parquet for %s on %s", symbol, date)
        return _skip("missing_parquet")

    trades_for_auction = filter_valid_trades(trades_all)
    trades = filter_valid_trades(filter_regular_hours(trades_all))
    nbbo = filter_valid_quotes(filter_regular_hours(nbbo_all))
    trades = filter_trades_near_quotes(trades, nbbo)
    if trades.empty or nbbo.empty:
        return _skip("empty_after_filter")

    adv_shares = float(trades["volume"].sum())
    ofi_scale = cfg.OFI_SIGNAL_SCALE

    # Tape-replay dispatchers require the filtered lit-trade tape.
    if fill_specification == "tape_replay":
        dispatcher = _TapeReplayDispatcher(trades, haircut_p=1.0)
    elif fill_specification == "tape_replay_haircut":
        dispatcher = _TapeReplayDispatcher(trades, haircut_p=0.5)
    elif fill_specification == "tape_replay_volume":
        dispatcher = _TapeReplayDispatcher(trades, haircut_p=1.0, volume_cap=True)
    elif fill_specification == "tape_replay_volume_haircut":
        dispatcher = _TapeReplayDispatcher(trades, haircut_p=0.5, volume_cap=True)
    elif fill_specification == "tape_replay_strict":
        # Conservative lower bound: only trades strictly through the limit.
        dispatcher = _TapeReplayDispatcher(
            trades, haircut_p=1.0, fill_rule="strict_through",
        )
    elif fill_specification == "tape_replay_queue":
        # Mechanical volume-ahead queue model (headline): inherently
        # volume-aware, so the cap is always on.
        dispatcher = _TapeReplayDispatcher(
            trades, haircut_p=1.0, volume_cap=True, fill_rule="queue",
        )
    else:
        dispatcher = _resolve_fill_dispatcher(fill_specification, fill_model, km_model)

    auction = extract_closing_auction_details(trades_for_auction)
    close_price, close_volume = auction.price, auction.volume
    if close_volume <= 0 or not np.isfinite(close_price):
        log.debug(
            "No usable closing auction for %s on %s "
            "(price_source=%s, volume_source=%s); skipping symbol-day",
            symbol, date, auction.price_source, auction.volume_source,
        )
        return _skip("missing_auction")
    state = MarketState.build(symbol, date, trades, nbbo, close_price, close_volume)

    delta_max = delta_max_bps_by_tier.get(tier, cfg.DELTA_MAX_BPS.get(tier, 5.0))

    day_seed = _stable_hash_int(f"{seed}|{date.isoformat()}|{symbol}")
    rows: list[dict] = []

    try:
        fm = _model_for_symbol_or_tier(dispatcher, symbol, tier)
    except KeyError:
        log.debug("No fill-model for %s; skipping all strategies", symbol)
        return _skip("no_fill_model")

    for strat_name in strategies:
        if strat_name == "S0_MOC":
            strat = get_strategy(strat_name)
            for _, order in parent_orders.iterrows():
                fr = strat.simulate(order, state)
                rows.append(_attach_auction_metadata(_result_to_row(fr, order), auction))
            continue

        s3_variants = {"S3_SIGNAL", "S3_OFI", "S3_IMB", "S3_FULL"}
        # Share static kwargs; dynamic urgency horizons are set per parent order.
        base_kwargs = {"refresh_seconds": cfg.REFRESH_SECONDS_DEFAULT}
        if strat_name in s3_variants:
            p = dict(s3_params or {})
            p.setdefault("kappa", 0.5)
            p.setdefault("lambda_imb", 1.0)
            p["ofi_scale"] = ofi_scale
            p["adv_shares"] = adv_shares
            base_kwargs.update(p)
        if strat_name == "S4_TOD":
            base_kwargs["tod_schedule"] = tod_schedule
        if strat_name == "S5_VALUE_AWARE_XGB":
            base_kwargs.update({
                "value_model": value_model,
                "tier": tier,
                "sector": sector,
                "listing_exchange": listing_exchange,
            })

        for _, order in parent_orders.iterrows():
            kwargs = dict(base_kwargs)
            if strat_name in ({"S2_TIME_ADAPTIVE", "S4_TOD", "S5_VALUE_AWARE_XGB"} | s3_variants):
                kwargs["window_start"] = pd.Timestamp(order["arrival_time"])
                kwargs["moc_cutoff"] = pd.Timestamp(order["moc_cutoff"])
            if strat_name in s3_variants:
                kwargs["imbalance_scale_shares"] = max(
                    1.0, float(order.get("expected_vc", 0.0) or 0.0),
                )
            if strat_name == "S5_VALUE_AWARE_XGB":
                kwargs["size_frac"] = float(order.get(
                    "size_frac", cfg.PARENT_ORDER_PRIMARY_FRACTION,
                ) or 0.0)
            strat = get_strategy(strat_name, **kwargs)
            sigma_bar = _causal_sigma_bar(state.rv, pd.Timestamp(order["arrival_time"]))
            max_slice = _max_slice_from_expected_vc(order)
            fr = strat.simulate(
                order, state, fill_model=fm,
                sigma_bar=sigma_bar, delta_max_bps=delta_max,
                max_slice_shares=max_slice,
                rng=_order_rng(day_seed, strat_name, str(order["order_id"])),
            )
            rows.append(_attach_auction_metadata(_result_to_row(fr, order), auction))

    return pd.DataFrame(rows)
