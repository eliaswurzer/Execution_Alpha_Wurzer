from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis import config as cfg
from analysis.data.features import resample_midquote
from analysis.data.taq_loader import filter_valid_quotes
from analysis.metrics.alpha import attach_alpha_columns, attach_moc_differential_columns
from analysis.metrics.raear import break_even_eta, information_ratio, raear
from analysis.microstructure.ofi import compute_ofi
from analysis.runners import h2_signal_efficiency as h2
from analysis.simulation import engine
from analysis.simulation.parent_orders import build_parent_orders
from analysis.strategies.base import FillResult


@pytest.mark.unit
def test_ofi_bucket_timestamp_is_right_edge_not_future_leaking() -> None:
    day = dt.date(2018, 2, 1)
    nbbo = pd.DataFrame({
        "time": pd.to_datetime([f"{day} 15:30:00", f"{day} 15:30:05"]),
        "best_bid": [100.00, 100.01],
        "best_offer": [100.02, 100.02],
        "best_bid_size": [100, 150],
        "best_offer_size": [100, 100],
    })

    out = compute_ofi(nbbo, bucket_seconds=30, zscore_window=2)

    assert pd.Timestamp(f"{day} 15:30:00") in set(out["timestamp"])
    assert out.loc[out["timestamp"] == pd.Timestamp(f"{day} 15:30:00"), "ofi"].iloc[0] == 0.0
    assert out.loc[out["timestamp"] == pd.Timestamp(f"{day} 15:30:30"), "ofi"].iloc[0] != 0.0


@pytest.mark.unit
def test_ofi_rolling_zscore_is_unchanged_by_future_extreme_update() -> None:
    day = dt.date(2018, 2, 1)
    base = pd.DataFrame({
        "time": pd.to_datetime([
            f"{day} 10:00:00",
            f"{day} 10:00:30",
            f"{day} 10:01:00",
        ]),
        "best_bid": [100.00, 100.01, 100.02],
        "best_offer": [100.02, 100.02, 100.03],
        "best_bid_size": [100, 150, 160],
        "best_offer_size": [100, 90, 80],
    })
    future = pd.concat([
        base,
        pd.DataFrame({
            "time": pd.to_datetime([f"{day} 15:59:30"]),
            "best_bid": [101.00],
            "best_offer": [101.02],
            "best_bid_size": [100_000],
            "best_offer_size": [1],
        }),
    ], ignore_index=True)

    early = compute_ofi(base, bucket_seconds=30, zscore_window=3)
    with_future = compute_ofi(future, bucket_seconds=30, zscore_window=3)
    shared = early["timestamp"]

    np.testing.assert_allclose(
        early["ofi_zscore"].to_numpy(),
        with_future[with_future["timestamp"].isin(shared)]["ofi_zscore"].to_numpy(),
    )


@pytest.mark.unit
def test_midquote_resample_is_right_labeled_and_causal() -> None:
    day = dt.date(2018, 2, 1)
    nbbo = pd.DataFrame({
        "time": pd.to_datetime([
            f"{day} 15:30:00",
            f"{day} 15:30:01",
        ]),
        "mid": [100.0, 101.0],
    })

    mids = resample_midquote(nbbo, interval_seconds=5)

    assert mids.loc[pd.Timestamp(f"{day} 15:30:00")] == 100.0
    assert mids.loc[pd.Timestamp(f"{day} 15:30:05")] == 101.0


@pytest.mark.unit
def test_tape_replay_window_excludes_t0_and_includes_t_end() -> None:
    trades_at_start = pd.DataFrame({
        "time": pd.to_datetime(["2018-02-01 15:30:00"]),
        "price": [99.0],
    })
    model = engine.TapeReplayModel(trades_at_start)
    p_fill, fill_time = model.fill_event(
        pd.Timestamp("2018-02-01 15:30:00"),
        pd.Timestamp("2018-02-01 15:30:30"),
        100.0,
        "BUY",
    )
    assert p_fill == 0.0
    assert fill_time is None

    trades_at_end = pd.DataFrame({
        "time": pd.to_datetime(["2018-02-01 15:30:30"]),
        "price": [99.0],
    })
    model = engine.TapeReplayModel(trades_at_end)
    p_fill, fill_time = model.fill_event(
        pd.Timestamp("2018-02-01 15:30:00"),
        pd.Timestamp("2018-02-01 15:30:30"),
        100.0,
        "BUY",
    )
    assert p_fill == 1.0
    assert fill_time == pd.Timestamp("2018-02-01 15:30:30")


@pytest.mark.unit
def test_locked_and_crossed_quotes_are_removed() -> None:
    quotes = pd.DataFrame({
        "time": pd.to_datetime(["2018-02-01 10:00:00"] * 3),
        "best_bid": [100.00, 100.00, 100.02],
        "best_offer": [100.01, 100.00, 100.01],
        "rel_spread": [0.0001, 0.0, -0.0001],
    })

    out = filter_valid_quotes(quotes)

    assert len(out) == 1
    assert out.iloc[0]["best_offer"] > out.iloc[0]["best_bid"]


@pytest.mark.unit
def test_parent_order_sides_are_deterministic_alternating_and_panel_balanced() -> None:
    day = dt.date(2018, 2, 1)
    one = build_parent_orders("AAPL", day, 1000, size_fractions=(0.01,))
    two = build_parent_orders("AAPL", day, 1000, size_fractions=(0.01,))
    assert list(one["side"]) == list(two["side"])
    assert all(a != b for a, b in zip(one["side"], one["side"].iloc[1:]))

    panel = pd.concat(
        build_parent_orders(f"SYM{i}", day, 1000, size_fractions=(0.01,))
        for i in range(40)
    )
    counts = panel["side"].value_counts()
    assert set(counts.index) == {"BUY", "SELL"}
    assert abs(int(counts["BUY"]) - int(counts["SELL"])) <= 40


@pytest.mark.unit
def test_simulation_uses_causal_sigma_and_expected_vc_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    day = dt.date(2018, 2, 1)
    trades = pd.DataFrame({
        "time": pd.to_datetime([f"{day} 15:30:01", f"{day} 16:00:00"]),
        "price": [100.0, 100.0],
        "volume": [100, 1_000_000],
        "sale_condition": ["", "6"],
        "exchange": ["Q", "Q"],
    })
    nbbo = pd.DataFrame({
        "time": pd.to_datetime([f"{day} 15:30:00", f"{day} 16:00:00"]),
        "best_bid": [99.99, 99.99],
        "best_offer": [100.01, 100.01],
        "best_bid_size": [100, 100],
        "best_offer_size": [100, 100],
        "mid": [100.0, 100.0],
        "rel_spread": [0.0002, 0.0002],
    })
    captured: list[tuple[float, int]] = []

    class FakeStrategy:
        name = "S1_STATIC"

        def simulate(self, order, state, **kwargs):  # noqa: ARG002
            captured.append((kwargs["sigma_bar"], kwargs["max_slice_shares"]))
            qty = int(order["qty"])
            return FillResult(
                order_id=order["order_id"],
                symbol=state.symbol,
                date=state.date,
                side=order["side"],
                strategy=self.name,
                window=order["window"],
                qty_intended=qty,
                qty_filled_passive=0,
                qty_filled_moc=qty,
                vwap_passive=float("nan"),
                close_price=state.close_price,
                avg_fill_price=state.close_price,
                fill_rate=0.0,
            )

    def fake_state_build(symbol, date, trades_frame, nbbo_frame, close_price, close_volume):  # noqa: ARG001
        return SimpleNamespace(
            symbol=symbol,
            date=date,
            close_price=close_price,
            close_volume=close_volume,
            rv=pd.Series(
                [0.2, 99.0],
                index=pd.to_datetime([f"{day} 15:00:00", f"{day} 15:45:00"]),
            ),
        )

    monkeypatch.setattr(engine, "load_symbol_day", lambda date, symbol, rth_only=True: (trades, nbbo))
    monkeypatch.setattr(engine, "filter_valid_trades", lambda frame: frame)
    monkeypatch.setattr(engine, "filter_valid_quotes", lambda frame: frame)
    monkeypatch.setattr(engine, "filter_trades_near_quotes", lambda trade_frame, quote_frame: trade_frame)
    monkeypatch.setattr(engine.MarketState, "build", fake_state_build)
    monkeypatch.setattr(engine, "get_strategy", lambda name, **kwargs: FakeStrategy())

    parent = pd.DataFrame([{
        "order_id": "order",
        "symbol": "AAPL",
        "date": day,
        "side": "BUY",
        "qty": 100,
        "arrival_time": pd.Timestamp(f"{day} 15:30:00"),
        "moc_cutoff": pd.Timestamp(f"{day} 15:50:00"),
        "size_frac": 0.01,
        "window": "B",
        "expected_vc": 1000.0,
    }])

    out = engine.simulate_symbol_day(
        "AAPL",
        day,
        parent,
        ["S1_STATIC"],
        fill_model=None,
        delta_max_bps_by_tier={1: 2.0},
        tier=1,
        fill_specification="tape_replay",
    )

    assert not out.empty
    assert captured == [(0.2, 50)]


@pytest.mark.unit
def test_alpha_moc_differential_manual_values() -> None:
    panel = pd.DataFrame({
        "order_id": ["o1", "o1"],
        "strategy": ["S0_MOC", "S1_STATIC"],
        "side": ["BUY", "BUY"],
        "close_price": [100.0, 100.0],
        "vwap_passive": [float("nan"), 99.5],
        "fill_rate": [0.0, 0.5],
        "adverse_selection_bps": [0.0, -2.0],
        "size_frac": [0.01, 0.01],
    })

    out = attach_moc_differential_columns(attach_alpha_columns(panel))
    passive = out[out["strategy"] == "S1_STATIC"].iloc[0]

    assert passive["alpha_bps"] == pytest.approx(25.0)
    # Implementation shortfall: the AS diagnostic stays on the row but is not
    # deducted again from net alpha (it is already inside the gross term).
    expected_net = 25.0 + 0.5 * cfg.MAKER_REBATE_BPS - cfg.COMMISSION_BPS
    assert passive["net_alpha_bps"] == pytest.approx(expected_net)
    assert passive["adverse_selection_cost_bps"] == pytest.approx(2.0)
    assert passive["net_alpha_vs_moc_bps"] == pytest.approx(expected_net + cfg.COMMISSION_BPS)


@pytest.mark.unit
def test_h2_per_bin_differentials_are_order_matched() -> None:
    rows = []
    for order_id, fill_rate, s2, ofi, imb, full in [
        ("o1", 0.1, 1.0, 2.0, 4.0, 7.0),
        ("o2", 0.9, 10.0, 13.0, 15.0, 19.0),
    ]:
        for strategy, alpha in [
            ("S2_TIME_ADAPTIVE", s2),
            ("S3_OFI", ofi),
            ("S3_IMB", imb),
            ("S3_FULL", full),
        ]:
            rows.append({
                "order_id": order_id,
                "symbol": "AAPL",
                "date": dt.date(2018, 2, 1),
                "strategy": strategy,
                "fill_rate": fill_rate,
                "net_alpha_bps": alpha,
            })
    out = h2.compute_decomposition(pd.DataFrame(rows), n_bins=2)
    diffs = out["per_bin_differentials"]

    ofi = diffs[diffs["label"] == "OFI_marginal"].sort_values("bin")

    assert list(ofi["mean"]) == pytest.approx([1.0, 3.0])
    assert "per_bin_differentials" in out


@pytest.mark.unit
def test_h3_raear_manual_values() -> None:
    assert information_ratio(4.0, 16.0) == pytest.approx(1.0)
    assert raear(4.0, 16.0, 0.1) == pytest.approx(2.4)
    assert break_even_eta(4.0, 16.0) == pytest.approx(0.25)

