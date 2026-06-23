from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import config as cfg
from analysis.data.taq_loader import (
    extract_closing_auction,
    extract_closing_auction_details,
)
from analysis.fill_model.state_vector import build_event_panel
from analysis.microstructure.imbalance import compute_auction_imbalance_proxy
from analysis.runners._common import _load_tod_schedule
from analysis.runners.h1_performance_gap import _attach_dissemination_flag
from analysis.simulation import engine
from analysis.simulation.parent_orders import build_parent_orders
from analysis.strategies.base import ExecutionStrategy, FillResult, MarketState
from analysis.strategies.optimal_schedule import OptimalScheduleStrategy
from analysis.strategies.signal_conditioned import SignalConditionedStrategy
from analysis.strategies.time_adaptive import TimeAdaptiveStrategy
from analysis.utils.symbols import expand_symbol_to_tier


def test_time_adaptive_urgency_is_window_specific() -> None:
    day = dt.date(2018, 2, 1)
    cutoff = pd.Timestamp.combine(day, cfg.MOC_CUTOFF)
    t = pd.Timestamp.combine(day, dt.time(15, 47))
    strategies = [
        TimeAdaptiveStrategy(window_start=pd.Timestamp.combine(day, start), moc_cutoff=cutoff)
        for start in cfg.EXECUTION_WINDOWS.values()
    ]

    urgencies = [strategy._urgency(t) for strategy in strategies]

    assert urgencies[0] < urgencies[1] < urgencies[2]


def test_imbalance_proxy_is_available_before_moc_cutoff() -> None:
    day = dt.date(2018, 2, 1)
    nbbo = pd.DataFrame({
        "time": pd.to_datetime([
            f"{day} 15:29:30",
            f"{day} 15:30:00",
            f"{day} 15:40:00",
            f"{day} 15:49:30",
            f"{day} 15:50:00",
        ]),
        "best_bid": [99.99, 100.00, 100.01, 100.01, 100.01],
        "best_offer": [100.01, 100.02, 100.03, 100.02, 100.02],
        "best_bid_size": [100, 120, 160, 200, 200],
        "best_offer_size": [100, 100, 90, 80, 80],
        "mid": [100.0, 100.01, 100.02, 100.015, 100.015],
    })

    proxy = compute_auction_imbalance_proxy(nbbo)

    assert not proxy.empty
    assert proxy["time"].min() == pd.Timestamp(f"{day} 15:30:00")
    assert (proxy["time"].dt.time < cfg.MOC_CUTOFF).any()
    assert proxy.loc[proxy["time"] == pd.Timestamp(f"{day} 15:49:30"), "imb_shares"].iloc[0] > 0


def test_closing_auction_finds_late_disseminated_closing_trade() -> None:
    day = dt.date(2018, 6, 15)
    trades = pd.DataFrame({
        "time": pd.to_datetime([
            f"{day} 16:00:00.100000",
            f"{day} 16:06:19.177533",
            f"{day} 16:06:19.177630",
        ]),
        "price": [115.10, 114.38, 114.38],
        "volume": [530, 12_363_707, 12_363_707],
        "sale_condition": ["M", "6", "M"],
    })

    auction = extract_closing_auction_details(trades)

    assert auction.price == 114.38
    assert auction.volume == 12_363_707
    assert auction.volume_source == "closing_trade"
    assert auction.close_trade_rows == 1


def test_closing_auction_uses_marked_m_fallback_when_no_closing_trade() -> None:
    day = dt.date(2018, 6, 15)
    trades = pd.DataFrame({
        "time": pd.to_datetime([
            f"{day} 16:00:00.100000",
            f"{day} 16:00:00.200000",
        ]),
        "price": [50.0, 50.0],
        "volume": [100, 250],
        "sale_condition": ["M", "M"],
    })

    auction = extract_closing_auction_details(trades)

    assert auction.volume == 250
    assert auction.volume_source == "official_marker_fallback"
    assert auction.official_marker_volume == 350


def test_s3_imbalance_factor_moves_pre_cutoff_limits() -> None:
    day = dt.date(2018, 2, 1)
    nbbo = pd.DataFrame({
        "time": pd.to_datetime([
            f"{day} 15:30:00",
            f"{day} 15:45:00",
            f"{day} 15:50:00",
        ]),
        "best_bid": [100.0, 100.0, 100.0],
        "best_offer": [100.02, 100.02, 100.02],
        "best_bid_size": [1000, 1000, 1000],
        "best_offer_size": [1000, 1000, 1000],
        "mid": [100.01, 100.01, 100.01],
    })
    trades = pd.DataFrame({
        "time": pd.to_datetime([f"{day} 16:00:00"]),
        "price": [100.0],
        "volume": [1000],
        "sale_condition": ["6"],
        "exchange": ["Q"],
    })
    state = MarketState.build("AAPL", day, trades, nbbo, 100.0, 1000.0)
    state.rv = pd.Series([], dtype=float)
    state.rv_times = np.array([], dtype="int64")
    state.imbalance = pd.DataFrame({
        "time": pd.to_datetime([f"{day} 15:45:00"]),
        "imb_shares": [100.0],
        "imb_sign": [1],
    })
    state.imbalance_times = state.imbalance["time"].values.astype("int64")
    strategy = SignalConditionedStrategy(
        mode="imb",
        window_start=pd.Timestamp(f"{day} 15:30:00"),
        moc_cutoff=pd.Timestamp(f"{day} 15:50:00"),
        lambda_imb=1.0,
        adv_shares=1_000_000.0,
        imbalance_scale_shares=1_000.0,
    )
    baseline = TimeAdaptiveStrategy(
        window_start=pd.Timestamp(f"{day} 15:30:00"),
        moc_cutoff=pd.Timestamp(f"{day} 15:50:00"),
    )
    t = pd.Timestamp(f"{day} 15:45:00")

    assert strategy.limit_offset_bps(t, "BUY", state, 1.0, 10.0) > baseline.limit_offset_bps(
        t, "BUY", state, 1.0, 10.0,
    )


def test_simulator_builds_dynamic_strategy_per_parent_window_and_keeps_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    day = dt.date(2018, 2, 1)
    times = pd.to_datetime([
        f"{day} 15:00:00",
        f"{day} 15:30:00",
        f"{day} 15:45:00",
        f"{day} 15:50:00",
        f"{day} 16:00:00",
    ])
    trades = pd.DataFrame({
        "time": times,
        "price": [100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [10, 10, 10, 10, 100],
        "sale_condition": ["", "", "", "", "6"],
    })
    nbbo = pd.DataFrame({
        "time": times,
        "best_bid": [99.99] * len(times),
        "best_offer": [100.01] * len(times),
        "best_bid_size": [100] * len(times),
        "best_offer_size": [100] * len(times),
        "mid": [100.0] * len(times),
    })
    starts: list[pd.Timestamp] = []

    class FakeStrategy:
        name = "S2_TIME_ADAPTIVE"

        def simulate(self, order, state, **kwargs):  # noqa: ARG002
            return FillResult(
                order_id=order["order_id"],
                symbol=state.symbol,
                date=state.date,
                side=order["side"],
                strategy=self.name,
                window=order["window"],
                qty_intended=int(order["qty"]),
                qty_filled_passive=0,
                qty_filled_moc=int(order["qty"]),
                vwap_passive=float("nan"),
                close_price=state.close_price,
                avg_fill_price=state.close_price,
                fill_rate=0.0,
            )

    def fake_get_strategy(name: str, **kwargs):
        assert name == "S2_TIME_ADAPTIVE"
        starts.append(kwargs["window_start"])
        return FakeStrategy()

    monkeypatch.setattr(engine, "load_symbol_day", lambda date, symbol, rth_only=True: (trades, nbbo))
    monkeypatch.setattr(engine, "filter_valid_trades", lambda frame: frame)
    monkeypatch.setattr(engine, "filter_valid_quotes", lambda frame: frame)
    monkeypatch.setattr(engine, "filter_trades_near_quotes", lambda trade_frame, quote_frame: trade_frame)
    monkeypatch.setattr(engine, "get_strategy", fake_get_strategy)

    parents = build_parent_orders("AAPL", day, 1000, size_fractions=(0.01,))
    out = engine.simulate_symbol_day(
        "AAPL", day, parents, ["S2_TIME_ADAPTIVE"], fill_model=None,
        delta_max_bps_by_tier={1: 2.0}, tier=1, fill_specification="tape_replay",
    )

    assert starts == list(pd.to_datetime(parents["arrival_time"]))
    assert set(("arrival_time", "moc_cutoff", "expected_vc")).issubset(out.columns)


def test_tape_replay_returns_first_cross_timestamp() -> None:
    trades = pd.DataFrame({
        "time": pd.to_datetime([
            "2018-02-01 15:30:01",
            "2018-02-01 15:30:05",
            "2018-02-01 15:30:07",
        ]),
        "price": [100.01, 99.99, 99.98],
    })
    tape = engine.TapeReplayModel(trades)

    p_fill, fill_time = tape.fill_event(
        pd.Timestamp("2018-02-01 15:30:00"),
        pd.Timestamp("2018-02-01 15:30:30"),
        100.0,
        "BUY",
    )

    assert p_fill == 1.0
    assert fill_time == pd.Timestamp("2018-02-01 15:30:05")


def test_tape_replay_excludes_configured_non_lit_exchange() -> None:
    trades = pd.DataFrame({
        "time": pd.to_datetime([
            "2018-02-01 15:30:05",
            "2018-02-01 15:30:10",
        ]),
        "price": [99.0, 100.0],
        "exchange": ["D", "Q"],
    })
    tape = engine.TapeReplayModel(trades)

    p_fill, fill_time = tape.fill_event(
        pd.Timestamp("2018-02-01 15:30:00"),
        pd.Timestamp("2018-02-01 15:30:30"),
        99.5,
        "BUY",
    )

    assert p_fill == 0.0
    assert fill_time is None


def test_passive_offsets_are_not_capped_to_the_quoted_spread() -> None:
    class WidePassiveStrategy(ExecutionStrategy):
        name = "WIDE_PASSIVE"

        def limit_offset_bps(self, t, side, state, sigma_bar, delta_max_bps):  # noqa: ARG002
            return 10.0

    day = dt.date(2018, 2, 1)
    nbbo = pd.DataFrame({
        "time": pd.to_datetime([
            f"{day} 15:30:00",
            f"{day} 15:30:30",
            f"{day} 16:00:00",
        ]),
        "best_bid": [99.99, 99.99, 99.99],
        "best_offer": [100.01, 100.01, 100.01],
        "best_bid_size": [1000, 1000, 1000],
        "best_offer_size": [1000, 1000, 1000],
        "mid": [100.0, 100.0, 100.0],
    })
    trades = pd.DataFrame({
        "time": pd.to_datetime([
            f"{day} 15:30:05",
            f"{day} 16:00:00",
        ]),
        "price": [99.965, 100.0],
        "volume": [100, 1000],
        "sale_condition": ["", "6"],
        "exchange": ["Q", "Q"],
    })
    state = MarketState.build("AAPL", day, trades, nbbo, 100.0, 1000.0)
    order = pd.Series({
        "order_id": "wide",
        "side": "BUY",
        "qty": 100,
        "arrival_time": pd.Timestamp(f"{day} 15:30:00"),
        "moc_cutoff": pd.Timestamp(f"{day} 15:30:30"),
        "window": "B",
    })

    result = WidePassiveStrategy().simulate(
        order,
        state,
        fill_model=engine.TapeReplayModel(trades),
        sigma_bar=1.0,
        delta_max_bps=10.0,
    )

    assert result.qty_filled_passive == 0
    assert result.fill_rate == 0.0


def test_model_fill_path_uses_effective_snapped_offset(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "SNAP_LIMIT_TO_TICK", True)

    class HalfBpStrategy(ExecutionStrategy):
        name = "HALF_BP"

        def limit_offset_bps(self, t, side, state, sigma_bar, delta_max_bps):  # noqa: ARG002
            return 0.5

    class CapturingModel:
        def __init__(self):
            self.rows = []

        def fill_probability(self, horizon_seconds, x):  # noqa: ARG002
            self.rows.append(dict(x))
            return 0.0

    day = dt.date(2018, 2, 1)
    nbbo = pd.DataFrame({
        "time": pd.to_datetime([f"{day} 15:30:00", f"{day} 15:30:30"]),
        "best_bid": [100.03, 100.03],
        "best_offer": [100.05, 100.05],
        "best_bid_size": [1000, 1000],
        "best_offer_size": [1000, 1000],
        "mid": [100.04, 100.04],
    })
    trades = pd.DataFrame({
        "time": pd.to_datetime([f"{day} 16:00:00"]),
        "price": [100.04],
        "volume": [1000],
        "sale_condition": ["6"],
        "exchange": ["Q"],
    })
    state = MarketState.build("AAPL", day, trades, nbbo, 100.04, 1000.0)
    order = pd.Series({
        "order_id": "effective-offset",
        "side": "BUY",
        "qty": 100,
        "arrival_time": pd.Timestamp(f"{day} 15:30:00"),
        "moc_cutoff": pd.Timestamp(f"{day} 15:30:30"),
        "window": "B",
    })
    model = CapturingModel()

    HalfBpStrategy().simulate(
        order,
        state,
        fill_model=model,
        sigma_bar=1.0,
        delta_max_bps=5.0,
    )

    assert model.rows
    assert model.rows[0]["limit_offset_bps"] > 0.9
    assert not np.isclose(model.rows[0]["limit_offset_bps"], 0.5)


def test_model_fill_path_uses_cutoff_capped_horizon() -> None:
    class AtTouchStrategy(ExecutionStrategy):
        name = "AT_TOUCH"

        def limit_offset_bps(self, t, side, state, sigma_bar, delta_max_bps):  # noqa: ARG002
            return 0.0

    class CapturingModel:
        def __init__(self):
            self.horizons = []

        def fill_probability(self, horizon_seconds, x):  # noqa: ARG002
            self.horizons.append(float(horizon_seconds))
            return 1.0

    day = dt.date(2018, 2, 1)
    nbbo = pd.DataFrame({
        "time": pd.to_datetime([f"{day} 15:49:45", f"{day} 15:50:00"]),
        "best_bid": [100.00, 100.00],
        "best_offer": [100.02, 100.02],
        "best_bid_size": [1000, 1000],
        "best_offer_size": [1000, 1000],
        "mid": [100.01, 100.01],
    })
    trades = pd.DataFrame({
        "time": pd.to_datetime([f"{day} 16:00:00"]),
        "price": [100.01],
        "volume": [1000],
        "sale_condition": ["6"],
        "exchange": ["Q"],
    })
    state = MarketState.build("AAPL", day, trades, nbbo, 100.01, 1000.0)
    order = pd.Series({
        "order_id": "short-final-interval",
        "side": "BUY",
        "qty": 100,
        "arrival_time": pd.Timestamp(f"{day} 15:49:45"),
        "moc_cutoff": pd.Timestamp(f"{day} 15:50:00"),
        "window": "B",
    })
    model = CapturingModel()

    result = AtTouchStrategy(refresh_seconds=30).simulate(
        order,
        state,
        fill_model=model,
        sigma_bar=1.0,
        delta_max_bps=0.0,
    )

    assert model.horizons == [15.0]
    assert result.detail_fills
    assert pd.Timestamp(result.detail_fills[0]["time"]) <= order["moc_cutoff"]


def test_event_panel_labels_use_snapped_limit_and_effective_offset(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "SNAP_LIMIT_TO_TICK", True)
    day = dt.date(2018, 2, 1)
    nbbo = pd.DataFrame({
        "time": pd.to_datetime([f"{day} 10:30:00", f"{day} 15:50:00"]),
        "best_bid": [100.03, 100.03],
        "best_offer": [100.05, 100.05],
        "best_bid_size": [1000, 1000],
        "best_offer_size": [1000, 1000],
        "mid": [100.04, 100.04],
    })
    trades = pd.DataFrame({
        "time": pd.to_datetime([f"{day} 10:30:05"]),
        "price": [100.02],
    })

    panel = build_event_panel(
        nbbo,
        trades,
        "AAPL",
        day,
        horizon_seconds=30,
        sample_every_seconds=3600,
        offset_grid_bps=(0.5,),
    )
    buy = panel[panel["side"] == "BUY"].iloc[0]

    assert np.isclose(buy["limit_price"], 100.02)
    assert buy["limit_offset_bps"] > 0.9
    assert buy["event"] == 1


def test_event_panel_excludes_and_caps_at_moc_cutoff() -> None:
    day = dt.date(2018, 2, 1)
    nbbo = pd.DataFrame({
        "time": pd.to_datetime([
            f"{day} 15:49:30",
            f"{day} 15:49:45",
            f"{day} 15:50:00",
        ]),
        "best_bid": [100.00, 100.00, 100.00],
        "best_offer": [100.02, 100.02, 100.02],
        "best_bid_size": [1000, 1000, 1000],
        "best_offer_size": [1000, 1000, 1000],
        "mid": [100.01, 100.01, 100.01],
    })
    trades = pd.DataFrame({
        "time": pd.to_datetime([
            f"{day} 15:50:05",
            f"{day} 15:50:10",
        ]),
        "price": [100.00, 100.02],
    })

    panel = build_event_panel(
        nbbo,
        trades,
        "AAPL",
        day,
        horizon_seconds=30,
        sample_every_seconds=15,
        offset_grid_bps=(0.0,),
    )

    assert not panel.empty
    assert (pd.to_datetime(panel["t0"]).dt.time < cfg.MOC_CUTOFF).all()
    final_rows = panel[pd.to_datetime(panel["t0"]) >= pd.Timestamp(f"{day} 15:49:30")]
    assert (final_rows["event"] == 0).all()
    # Non-fills are right-censored at the MOC cutoff, not at the full horizon:
    # a sample posted 15s before the cutoff must record a 15s censoring duration.
    cutoff = pd.Timestamp(f"{day} 15:50:00")
    expected_ttc = (cutoff - pd.to_datetime(final_rows["t0"])).dt.total_seconds()
    assert np.allclose(final_rows["duration"].to_numpy(), expected_ttc.to_numpy())
    assert (final_rows["duration"] <= 30.0 + 1e-9).all()


def test_volume_capped_tape_replay_partially_fills_slice() -> None:
    class AtTouchStrategy(ExecutionStrategy):
        name = "AT_TOUCH"

        def limit_offset_bps(self, t, side, state, sigma_bar, delta_max_bps):  # noqa: ARG002
            return 0.0

    day = dt.date(2018, 2, 1)
    nbbo = pd.DataFrame({
        "time": pd.to_datetime([
            f"{day} 15:30:00",
            f"{day} 15:30:30",
            f"{day} 16:00:00",
        ]),
        "best_bid": [100.0, 100.0, 100.0],
        "best_offer": [100.02, 100.02, 100.02],
        "best_bid_size": [1000, 1000, 1000],
        "best_offer_size": [1000, 1000, 1000],
        "mid": [100.01, 100.01, 100.01],
    })
    trades = pd.DataFrame({
        "time": pd.to_datetime([
            f"{day} 15:30:05",
            f"{day} 15:30:06",
            f"{day} 16:00:00",
        ]),
        "price": [100.0, 100.0, 100.0],
        "volume": [20, 10, 1000],
        "sale_condition": ["", "", "6"],
        "exchange": ["Q", "Q", "Q"],
    })
    state = MarketState.build("AAPL", day, trades, nbbo, 100.0, 1000.0)
    order = pd.Series({
        "order_id": "volume",
        "side": "BUY",
        "qty": 100,
        "arrival_time": pd.Timestamp(f"{day} 15:30:00"),
        "moc_cutoff": pd.Timestamp(f"{day} 15:30:30"),
        "window": "B",
    })

    result = AtTouchStrategy().simulate(
        order,
        state,
        fill_model=engine.TapeReplayModel(trades, volume_cap=True),
        sigma_bar=1.0,
        delta_max_bps=0.0,
    )

    assert result.qty_filled_passive == 30
    assert result.qty_filled_moc == 70
    assert result.fill_rate == 0.3


def test_closing_auction_uses_official_close_price_without_double_counting_volume() -> None:
    trades = pd.DataFrame({
        "time": pd.to_datetime([
            "2018-02-01 16:00:00.100",
            "2018-02-01 16:00:13.200",
            "2018-02-01 16:00:13.201",
        ]),
        "price": [99.9, 100.0, 100.0],
        "volume": [10, 500, 500],
        "sale_condition": ["M", "6", "M"],
    })

    close_price, close_volume = extract_closing_auction(trades)

    assert close_price == 100.0
    assert close_volume == 500.0


def test_s4_requires_tod_artifact() -> None:
    with pytest.raises(ValueError):
        OptimalScheduleStrategy(tod_schedule=None)
    with pytest.raises(FileNotFoundError):
        _load_tod_schedule(Path("coding/artifacts/missing_tod_artifact"), required=True)


def test_h1_dissemination_flag_uses_preserved_arrival_time() -> None:
    panel = pd.DataFrame({
        "arrival_time": pd.to_datetime(["2018-02-01 15:49:59", "2018-02-01 15:55:00"]),
        "listing_exchange": ["NYSE", "NASDAQ"],
    })

    out = _attach_dissemination_flag(panel)

    assert list(out["post_dissemination"]) == [False, True]


def test_symbol_tier_lookup_accepts_file_safe_aliases() -> None:
    mapping = expand_symbol_to_tier({"BRK B": 1})

    assert mapping["BRK B"] == 1
    assert mapping["BRK_B"] == 1
