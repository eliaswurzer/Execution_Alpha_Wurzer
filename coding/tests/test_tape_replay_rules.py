"""Tests for the tape-replay fill-rule bracket (at_or_through / queue /
strict_through), tick snapping, queue-ahead handling, and the searchsorted
adverse-selection lookup."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from analysis import config as cfg
from analysis.simulation import engine
from analysis.strategies.base import MarketState, _snap_limit_to_tick
from analysis.strategies.static_passive import StaticPassiveStrategy


def _ts(hms: str) -> pd.Timestamp:
    return pd.Timestamp(f"2018-02-01 {hms}")


def _tape(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame({
        "time": [_ts(t) for t, _, _ in rows],
        "price": [p for _, p, _ in rows],
        "volume": [v for _, _, v in rows],
    })


_WINDOW = (_ts("15:30:00"), _ts("15:31:00"))


# ---------------------------------------------------------------------------
# strict_through
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_strict_through_ignores_at_limit_prints() -> None:
    tape = _tape([("15:30:05", 100.00, 300)])
    m = engine.TapeReplayModel(tape, fill_rule="strict_through")
    p, ftime, _ = m.fill_event_details(*_WINDOW, 100.00, "BUY")
    assert p == 0.0 and ftime is None


@pytest.mark.unit
def test_strict_through_fills_on_through_print_with_time() -> None:
    tape = _tape([("15:30:05", 100.00, 300), ("15:30:15", 99.99, 100)])
    m = engine.TapeReplayModel(tape, fill_rule="strict_through", volume_cap=True)
    p, ftime, avail = m.fill_event_details(*_WINDOW, 100.00, "BUY")
    assert p == 1.0
    assert ftime == _ts("15:30:15")
    assert avail == 100.0  # only the through volume is fillable


# ---------------------------------------------------------------------------
# queue (volume-ahead)
# ---------------------------------------------------------------------------

_QUEUE_TAPE = _tape([
    ("15:30:05", 100.00, 300),   # at limit
    ("15:30:10", 100.00, 300),   # at limit (cum 600)
    ("15:30:15", 99.99, 100),    # through
])


@pytest.mark.unit
def test_queue_fills_when_at_limit_volume_exceeds_queue_ahead() -> None:
    m = engine.TapeReplayModel(_QUEUE_TAPE, fill_rule="queue")
    p, ftime, avail = m.fill_event_details(*_WINDOW, 100.00, "BUY", queue_ahead=500.0)
    assert p == 1.0
    assert ftime == _ts("15:30:10")  # cum at-limit volume crosses 500 here
    assert avail == pytest.approx(100.0 + (600.0 - 500.0))  # through + queue excess


@pytest.mark.unit
def test_queue_blocked_by_deep_queue_until_through_print() -> None:
    m = engine.TapeReplayModel(_QUEUE_TAPE, fill_rule="queue")
    p, ftime, avail = m.fill_event_details(*_WINDOW, 100.00, "BUY", queue_ahead=10_000.0)
    assert p == 1.0
    assert ftime == _ts("15:30:15")  # only the through print executes us
    assert avail == pytest.approx(100.0)


@pytest.mark.unit
def test_queue_with_no_queue_estimate_equals_at_or_through() -> None:
    m_queue = engine.TapeReplayModel(_QUEUE_TAPE, fill_rule="queue")
    m_legacy = engine.TapeReplayModel(_QUEUE_TAPE, volume_cap=True)
    p_q, t_q, a_q = m_queue.fill_event_details(*_WINDOW, 100.00, "BUY", queue_ahead=None)
    p_l, t_l, a_l = m_legacy.fill_event_details(*_WINDOW, 100.00, "BUY")
    assert (p_q, t_q) == (p_l, t_l)
    assert a_q == pytest.approx(a_l)


@pytest.mark.unit
def test_queue_sell_side_symmetry() -> None:
    tape = _tape([
        ("15:30:05", 100.00, 200),   # at SELL limit
        ("15:30:20", 100.01, 50),    # through (above limit)
    ])
    m = engine.TapeReplayModel(tape, fill_rule="queue")
    p, ftime, avail = m.fill_event_details(*_WINDOW, 100.00, "SELL", queue_ahead=500.0)
    assert p == 1.0
    assert ftime == _ts("15:30:20")
    assert avail == pytest.approx(50.0)


@pytest.mark.unit
def test_fill_rule_monotonicity_strict_le_queue_le_at_or_through() -> None:
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(1, 12))
        rows = [
            (
                f"15:30:{int(rng.integers(1, 59)):02d}",
                float(np.round(100.0 + rng.integers(-5, 6) * 0.01, 2)),
                float(rng.integers(50, 500)),
            )
            for _ in range(n)
        ]
        tape = _tape(rows)
        q0 = float(rng.integers(0, 800))
        p_at = engine.TapeReplayModel(tape).fill_event_details(*_WINDOW, 100.00, "BUY")[0]
        p_qu = engine.TapeReplayModel(tape, fill_rule="queue").fill_event_details(
            *_WINDOW, 100.00, "BUY", queue_ahead=q0,
        )[0]
        p_st = engine.TapeReplayModel(tape, fill_rule="strict_through").fill_event_details(
            *_WINDOW, 100.00, "BUY",
        )[0]
        assert p_st <= p_qu <= p_at


# ---------------------------------------------------------------------------
# Tick snapping
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_snap_limit_to_tick_passive_direction() -> None:
    assert _snap_limit_to_tick(170.5559, "BUY") == pytest.approx(170.55)
    assert _snap_limit_to_tick(170.5541, "SELL") == pytest.approx(170.56)
    # On-grid prices are stable in both directions despite float noise.
    assert _snap_limit_to_tick(170.59, "BUY") == pytest.approx(170.59)
    assert _snap_limit_to_tick(170.59, "SELL") == pytest.approx(170.59)


@pytest.mark.unit
def test_snap_limit_disabled_via_config(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "SNAP_LIMIT_TO_TICK", False)
    assert _snap_limit_to_tick(170.5559, "BUY") == pytest.approx(170.5559)


# ---------------------------------------------------------------------------
# Simulation integration: snapped limit + queue ahead + AS lookup
# ---------------------------------------------------------------------------

def _nbbo(drift_per_step: float = 0.0) -> pd.DataFrame:
    times = pd.date_range(_ts("15:29:00"), _ts("16:00:00"), freq="5s")
    steps = np.arange(len(times), dtype=float)
    bid = np.round(170.59 + drift_per_step * steps, 2)
    ask = np.round(170.61 + drift_per_step * steps, 2)
    return pd.DataFrame({
        "time": times,
        "best_bid": bid,
        "best_offer": ask,
        "best_bid_size": np.full(len(times), 5.0),    # round lots
        "best_offer_size": np.full(len(times), 5.0),
        "mid": (bid + ask) / 2.0,
    })


def _order() -> pd.Series:
    return pd.Series({
        "order_id": "o1",
        "side": "BUY",
        "qty": 100,
        "arrival_time": _ts("15:30:00"),
        "moc_cutoff": _ts("15:50:00"),
        "window": "B",
    })


def _state(nbbo: pd.DataFrame, trades: pd.DataFrame) -> MarketState:
    return MarketState.build("AAPL", dt.date(2018, 2, 1), trades, nbbo, 170.50, 1e6)


@pytest.mark.unit
def test_simulate_snaps_limit_price_onto_penny_grid() -> None:
    nbbo = _nbbo()
    # delta = 2 bps on 170.59 => raw limit 170.5559, snapped to 170.55.
    tape = _tape([("15:30:10", 170.55, 1_000)])
    state = _state(nbbo, tape)
    strat = StaticPassiveStrategy(refresh_seconds=30)
    fr = strat.simulate(
        _order(), state, fill_model=engine.TapeReplayModel(tape),
        sigma_bar=1.0, delta_max_bps=2.0,
    )
    assert fr.detail_fills, "expected at least one passive fill"
    assert all(f["price"] == pytest.approx(170.55) for f in fr.detail_fills)


@pytest.mark.unit
def test_simulate_passes_queue_ahead_from_displayed_depth() -> None:
    nbbo = _nbbo()
    # At-touch BUY (delta 0) at 170.59; displayed bid depth = 5 lots = 500 sh.
    # 400 shares trade at the limit -> queue not depleted -> no fill under the
    # queue rule, but a fill under at-or-through.
    tape = _tape([("15:30:10", 170.59, 400)])
    state = _state(nbbo, tape)
    strat = StaticPassiveStrategy(refresh_seconds=30)
    fr_queue = strat.simulate(
        _order(), state,
        fill_model=engine.TapeReplayModel(tape, fill_rule="queue", volume_cap=True),
        sigma_bar=1.0, delta_max_bps=0.0,
    )
    fr_legacy = strat.simulate(
        _order(), state, fill_model=engine.TapeReplayModel(tape),
        sigma_bar=1.0, delta_max_bps=0.0,
    )
    assert fr_queue.qty_filled_passive == 0
    assert fr_legacy.qty_filled_passive > 0


# ---------------------------------------------------------------------------
# Round-lot -> share unit conventions (NBBO sizes are Daily-TAQ round lots)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_state_at_converts_depth_to_shares() -> None:
    from analysis.fill_model.state_vector import state_at

    nbbo = pd.DataFrame({
        "time": [_ts("15:30:00")],
        "best_bid": [170.59],
        "best_offer": [170.61],
        "best_bid_size": [4.0],     # round lots
        "best_offer_size": [7.0],
        "mid": [170.60],
    })
    sv_buy = state_at(_ts("15:30:10"), nbbo, None, None, "BUY")
    sv_sell = state_at(_ts("15:30:10"), nbbo, None, None, "SELL")
    assert sv_buy["D0"] == pytest.approx(4.0 * cfg.NBBO_SIZE_SHARES_PER_LOT)
    assert sv_buy["q0"] == pytest.approx(0.5 * 4.0 * cfg.NBBO_SIZE_SHARES_PER_LOT)
    assert sv_sell["D0"] == pytest.approx(7.0 * cfg.NBBO_SIZE_SHARES_PER_LOT)


@pytest.mark.unit
def test_imbalance_proxy_is_share_denominated() -> None:
    from analysis.microstructure.imbalance import compute_auction_imbalance_proxy

    nbbo = pd.DataFrame({
        "time": [_ts("15:31:00"), _ts("15:35:00")],
        "best_bid": [170.59, 170.59],
        "best_offer": [170.61, 170.61],
        "best_bid_size": [10.0, 20.0],   # +10 lots depth, +10 lots OFI
        "best_offer_size": [10.0, 10.0],
    })
    out = compute_auction_imbalance_proxy(nbbo)
    # Second row: rolling OFI = +10 lots, depth drift = +10 lots -> 20 lots
    # -> 2000 shares of buy-side pressure.
    assert out["imb_shares"].iloc[-1] == pytest.approx(
        20.0 * cfg.NBBO_SIZE_SHARES_PER_LOT,
    )


@pytest.mark.unit
def test_f_imb_responds_at_realistic_share_magnitudes() -> None:
    from analysis.strategies.signal_conditioned import SignalConditionedIMB

    imb = pd.DataFrame({
        "time": [_ts("15:30:00")],
        "imb_shares": [-50_000.0],   # sell-side closing pressure (shares)
        "imb_sign": [-1],
    })
    state = MarketState(
        symbol="AAPL", date=dt.date(2018, 2, 1),
        nbbo=pd.DataFrame(), trades=pd.DataFrame(),
        close_price=170.50, close_volume=1e6,
        ofi=pd.DataFrame(), rv=pd.Series(dtype=float), imbalance=imb,
        imbalance_times=imb["time"].values.astype("int64"),
    )
    strat = SignalConditionedIMB(
        lambda_imb=1.0, imbalance_scale_shares=100_000.0,
    )
    f = strat._f_imb(_ts("15:31:00"), "BUY", state)
    # Sellers dominate -> a BUY should post more aggressively (factor < 1);
    # before the lot->share fix this argument was ~100x too small (f ~ 1).
    assert f == pytest.approx(0.5)


@pytest.mark.unit
def test_as_searchsorted_matches_merge_asof_fallback() -> None:
    # Steady downward drift (~1.2 cents per 30s AS horizon) makes the
    # side-signed post-fill drift strictly negative for a BUY.
    nbbo = _nbbo(drift_per_step=-0.002)
    # Deep through-prints so the at-or-through fills trigger regardless of
    # where the drifting limit sits.
    tape = _tape([("15:30:10", 160.00, 10_000), ("15:40:10", 160.00, 10_000)])
    full_state = _state(nbbo, tape)
    assert len(full_state.mids) == len(full_state.nbbo_times)

    # Hand-built state without the mids array forces the merge_asof fallback.
    manual_state = MarketState(
        symbol="AAPL", date=dt.date(2018, 2, 1),
        nbbo=full_state.nbbo, trades=full_state.trades,
        close_price=170.50, close_volume=1e6,
        ofi=full_state.ofi, rv=full_state.rv, imbalance=full_state.imbalance,
        nbbo_times=full_state.nbbo_times, nbbo_mid=full_state.nbbo_mid,
        ofi_times=full_state.ofi_times,
        imbalance_times=full_state.imbalance_times,
        rv_times=full_state.rv_times,
    )

    strat = StaticPassiveStrategy(refresh_seconds=30)
    kwargs = dict(sigma_bar=1.0, delta_max_bps=0.0)
    fr_fast = strat.simulate(
        _order(), full_state, fill_model=engine.TapeReplayModel(tape), **kwargs,
    )
    fr_slow = strat.simulate(
        _order(), manual_state, fill_model=engine.TapeReplayModel(tape), **kwargs,
    )
    assert fr_fast.qty_filled_passive == fr_slow.qty_filled_passive
    assert np.isclose(fr_fast.adverse_selection_bps, fr_slow.adverse_selection_bps)
    assert fr_fast.adverse_selection_bps != 0.0  # drift makes AS non-trivial
