"""Tests for the 2026-06-11 robustness batch: early-close calendar exclusion,
AdaptivePool thread fallback, structured skip reasons, summary aggregation,
parquet-layout memo, and the listing-exchange reference map."""

from __future__ import annotations

import datetime as dt
import os

import pandas as pd
import pytest

from analysis import config as cfg
from analysis.data import index_universe, taq_loader
from analysis.runners import _common, master_panel
from analysis.runners.build_listing_exchange_map import _sample_dates
from analysis.simulation import engine
from analysis.simulation.parent_orders import build_parent_orders
from analysis.utils import adaptive_pool
from analysis.utils.adaptive_pool import AdaptivePool


# ---------------------------------------------------------------------------
# Early-close exclusion
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_eval_dates_drop_documented_exclusions(monkeypatch, artifact_dir) -> None:
    root = artifact_dir / "taq"
    for ds in ("20180702", "20180703", "20180705"):
        (root / ds / "trades").mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(cfg.TAQ_PARQUET_DIR, 2018, root)
    dates = _common._eval_dates(dt.date(2018, 7, 1), dt.date(2018, 7, 31))
    assert dt.date(2018, 7, 3) in cfg.EXCLUDED_EVAL_DATES
    assert dates == [dt.date(2018, 7, 2), dt.date(2018, 7, 5)]


# ---------------------------------------------------------------------------
# AdaptivePool thread fallback
# ---------------------------------------------------------------------------

class _DeniedProcessPool:
    def __init__(self, *args, **kwargs):
        raise PermissionError("[WinError 5] Zugriff verweigert (pipe)")


@pytest.mark.unit
def test_adaptive_pool_falls_back_to_threads_on_permission_error(monkeypatch) -> None:
    monkeypatch.delenv("THESIS_POOL_BACKEND", raising=False)
    monkeypatch.setattr(adaptive_pool, "ProcessPoolExecutor", _DeniedProcessPool)
    monkeypatch.setattr(
        adaptive_pool, "_resources_available", lambda *_: True,
    )
    with AdaptivePool(max_workers=2, max_in_flight=2) as pool:
        assert pool.backend == "process"
        futures = [pool.submit(lambda x=i: x * 2) for i in range(4)]
        assert sorted(f.result() for f in futures) == [0, 2, 4, 6]
        assert pool.backend == "thread"


@pytest.mark.unit
def test_adaptive_pool_explicit_thread_backend_does_not_touch_processes(monkeypatch) -> None:
    monkeypatch.setenv("THESIS_POOL_BACKEND", "thread")
    monkeypatch.setattr(adaptive_pool, "ProcessPoolExecutor", _DeniedProcessPool)
    monkeypatch.setattr(
        adaptive_pool, "_resources_available", lambda *_: True,
    )
    with AdaptivePool(max_workers=1, max_in_flight=1) as pool:
        assert pool.submit(lambda: 7).result() == 7
        assert pool.backend == "thread"


# ---------------------------------------------------------------------------
# Structured skip reasons from simulate_symbol_day
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fill_dispatcher_falls_back_to_run_tier_for_missing_symbol_map() -> None:
    class Dispatcher:
        def __init__(self):
            self.models = {3: object()}

        def for_symbol(self, symbol: str):  # noqa: ARG002
            raise KeyError("symbol not calibrated")

    dispatcher = Dispatcher()

    assert (
        engine._model_for_symbol_or_tier(dispatcher, "CPRT", 3)
        is dispatcher.models[3]
    )


@pytest.mark.unit
def test_fill_dispatcher_keeps_no_model_when_tier_model_missing() -> None:
    class Dispatcher:
        models = {1: object()}

        def for_symbol(self, symbol: str):  # noqa: ARG002
            raise KeyError("symbol not calibrated")

    with pytest.raises(KeyError):
        engine._model_for_symbol_or_tier(Dispatcher(), "CPRT", 3)


def _frames_without_auction(day: dt.date) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.to_datetime([
        f"{day} 15:00:00", f"{day} 15:30:00", f"{day} 15:45:00",
    ])
    trades = pd.DataFrame({
        "time": times,
        "price": [100.0] * 3,
        "volume": [10] * 3,
        "sale_condition": ["", "", ""],
    })
    nbbo = pd.DataFrame({
        "time": times,
        "best_bid": [99.99] * 3,
        "best_offer": [100.01] * 3,
        "best_bid_size": [100] * 3,
        "best_offer_size": [100] * 3,
        "mid": [100.0] * 3,
    })
    return trades, nbbo


@pytest.mark.unit
def test_simulate_symbol_day_reports_missing_auction(monkeypatch) -> None:
    day = dt.date(2018, 2, 1)
    trades, nbbo = _frames_without_auction(day)
    monkeypatch.setattr(
        engine, "load_symbol_day",
        lambda date, symbol, rth_only=True: (trades, nbbo),
    )
    monkeypatch.setattr(engine, "filter_valid_trades", lambda frame: frame)
    monkeypatch.setattr(engine, "filter_valid_quotes", lambda frame: frame)
    monkeypatch.setattr(
        engine, "filter_trades_near_quotes", lambda t, q: t,
    )
    parents = build_parent_orders("AAPL", day, 1000, size_fractions=(0.01,))
    reason: dict = {}
    out = engine.simulate_symbol_day(
        "AAPL", day, parents, ["S2_TIME_ADAPTIVE"], fill_model=None,
        delta_max_bps_by_tier={1: 2.0}, tier=1,
        fill_specification="tape_replay", skip_reason_out=reason,
    )
    assert out.empty
    assert reason["reason"] == "missing_auction"


@pytest.mark.unit
def test_simulate_symbol_day_reports_empty_after_filter(monkeypatch) -> None:
    day = dt.date(2018, 2, 1)
    trades, nbbo = _frames_without_auction(day)
    monkeypatch.setattr(
        engine, "load_symbol_day",
        lambda date, symbol, rth_only=True: (trades, nbbo),
    )
    monkeypatch.setattr(engine, "filter_valid_trades", lambda frame: frame)
    monkeypatch.setattr(
        engine, "filter_valid_quotes", lambda frame: frame.iloc[0:0],
    )
    monkeypatch.setattr(engine, "filter_trades_near_quotes", lambda t, q: t)
    parents = build_parent_orders("AAPL", day, 1000, size_fractions=(0.01,))
    reason: dict = {}
    out = engine.simulate_symbol_day(
        "AAPL", day, parents, ["S2_TIME_ADAPTIVE"], fill_model=None,
        delta_max_bps_by_tier={1: 2.0}, tier=1,
        fill_specification="tape_replay", skip_reason_out=reason,
    )
    assert out.empty
    assert reason["reason"] == "empty_after_filter"


@pytest.mark.unit
def test_simulate_symbol_day_reports_missing_parquet(monkeypatch) -> None:
    day = dt.date(2018, 2, 1)

    def _raise(*_args, **_kwargs):
        raise FileNotFoundError("no parquet")

    monkeypatch.setattr(engine, "load_symbol_day", _raise)
    parents = build_parent_orders("AAPL", day, 1000, size_fractions=(0.01,))
    reason: dict = {}
    out = engine.simulate_symbol_day(
        "AAPL", day, parents, ["S2_TIME_ADAPTIVE"], fill_model=None,
        delta_max_bps_by_tier={1: 2.0}, tier=1,
        fill_specification="tape_replay", skip_reason_out=reason,
    )
    assert out.empty
    assert reason["reason"] == "missing_parquet"


# ---------------------------------------------------------------------------
# Summary aggregation: close_source_distribution
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_close_source_distribution_aggregates_shards(artifact_dir) -> None:
    shard_root = artifact_dir / "panel_shards"
    frame = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "date": ["2018-07-02"] * 3,
        "close_price_source": ["official_marker", "official_marker", "closing_trade"],
        "close_volume_source": ["closing_trade"] * 3,
    })
    day_dir = shard_root / "date=2018-07-02"
    day_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(day_dir / "panel.parquet", index=False)

    dist = master_panel._close_source_distribution(shard_root)
    by_source = {row["close_price_source"]: row for row in dist}
    assert by_source["official_marker"]["symbol_days"] == 1
    assert by_source["official_marker"]["panel_rows"] == 2
    assert by_source["closing_trade"]["symbol_days"] == 1


# ---------------------------------------------------------------------------
# Parquet layout memo
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_resolve_parquet_memoises_layout_per_date(monkeypatch, artifact_dir) -> None:
    root = artifact_dir / "taq"
    monkeypatch.setitem(cfg.TAQ_PARQUET_DIR, 2018, root)
    day = dt.date(2018, 1, 2)
    # Use the "new" layout (second candidate) so the first resolve needs two
    # probes; the memo should bring every later symbol down to one probe.
    for sym in ("AAPL", "MSFT"):
        path = root / "Trade" / "20180102" / f"{sym}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    calls = {"n": 0}
    real_exists = os.path.exists

    def counting_exists(self):
        calls["n"] += 1
        return real_exists(str(self))

    monkeypatch.setattr(type(root), "exists", counting_exists)
    taq_loader.clear_layout_cache()
    first = taq_loader.trades_parquet_path(day, "AAPL")
    probes_first = calls["n"]
    second = taq_loader.trades_parquet_path(day, "MSFT")
    probes_second = calls["n"] - probes_first
    assert first.name == "AAPL.parquet" and first.exists()
    assert second.name == "MSFT.parquet"
    # Second symbol on the same date reuses the memoised layout: exactly one
    # existence probe instead of walking the candidate list again.
    assert probes_second == 1
    assert probes_second < probes_first


# ---------------------------------------------------------------------------
# Listing-exchange reference map
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_membership_merges_listing_exchange_map(artifact_dir) -> None:
    root = artifact_dir / "index_membership"
    root.mkdir(parents=True, exist_ok=True)
    (root / "sp500_membership_intervals.csv").write_text(
        "index_id,symbol,effective_from,effective_to,listing_exchange\n"
        "sp500,AAPL,2018-01-02,2019-12-31,\n"
        "sp500,IBM,2018-01-02,2019-12-31,NYSE\n",
        encoding="utf-8",
    )
    (root / "sp500_listing_exchange.csv").write_text(
        "symbol,listing_exchange\nAAPL,NASDAQ\nIBM,NASDAQ\n",
        encoding="utf-8",
    )
    out = index_universe.load_index_membership("sp500", root)
    listing = dict(zip(out["symbol"], out["listing_exchange"]))
    # Blank value filled from the map; explicit membership value wins.
    assert listing["AAPL"] == "NASDAQ"
    assert listing["IBM"] == "NYSE"


@pytest.mark.unit
def test_sample_dates_spread_over_interval() -> None:
    available = [dt.date(2018, 1, 1) + dt.timedelta(days=i) for i in range(100)]
    picked = _sample_dates(
        available, dt.date(2018, 1, 1), dt.date(2018, 4, 10), 3,
    )
    assert picked[0] == available[0]
    assert picked[-1] == available[-1]
    assert len(picked) == 3
