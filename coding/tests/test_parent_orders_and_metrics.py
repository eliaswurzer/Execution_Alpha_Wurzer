from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from analysis.metrics import alpha
from analysis.simulation.parent_orders import (
    build_parent_orders,
    rolling_expected_vc,
    same_day_vc_fallback,
)


@pytest.mark.unit
def test_parent_orders_have_expected_windows_sizes_and_side_cycle() -> None:
    day = dt.date(2018, 1, 2)
    orders = build_parent_orders(
        "AAPL",
        day,
        expected_vc=1000,
        size_fractions=(0.01, 0.02),
        windows={"A": dt.time(15, 0), "B": dt.time(15, 30)},
        seed=42,
    )

    assert list(orders["qty"]) == [10, 20, 10, 20]
    assert list(orders["side"]) == list(build_parent_orders(
        "AAPL",
        day,
        expected_vc=1000,
        size_fractions=(0.01, 0.02),
        windows={"A": dt.time(15, 0), "B": dt.time(15, 30)},
        seed=42,
    )["side"])
    # Side is keyed on the window: both sizes of one window share the side,
    # and consecutive windows alternate (matched cross-size comparisons).
    sides_by_window = orders.groupby("window")["side"].unique()
    assert all(len(s) == 1 for s in sides_by_window)
    assert sides_by_window["A"][0] != sides_by_window["B"][0]
    assert set(orders["side"]) == {"BUY", "SELL"}
    assert list(orders["window"]) == ["A", "A", "B", "B"]
    assert orders["arrival_time"].iloc[0] == pd.Timestamp("2018-01-02 15:00:00")
    assert orders["moc_cutoff"].iloc[0] == pd.Timestamp("2018-01-02 15:50:00")


@pytest.mark.unit
def test_parent_orders_empty_when_expected_vc_is_unusable() -> None:
    assert build_parent_orders("AAPL", dt.date(2018, 1, 2), expected_vc=0).empty
    assert build_parent_orders("AAPL", dt.date(2018, 1, 2), expected_vc=-1).empty


@pytest.mark.unit
def test_rolling_expected_vc_is_exclusive_and_requires_five_history_days() -> None:
    hist = pd.DataFrame({
        "symbol": ["AAPL"] * 6,
        "date": [dt.date(2018, 1, d) for d in range(2, 8)],
        "vc_shares": [100, 200, 300, 400, 500, 600],
    })

    out = rolling_expected_vc(hist)

    assert out["expected_vc"].iloc[:5].isna().all()
    assert out["expected_vc"].iloc[5] == 300.0


@pytest.mark.unit
def test_same_day_vc_fallback_uses_observed_same_day_volume() -> None:
    hist = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": [dt.date(2018, 1, 2)],
        "vc_shares": [1234],
    })

    out = same_day_vc_fallback(hist)

    assert out.loc[0, "expected_vc"] == 1234.0


@pytest.mark.unit
def test_alpha_fee_impact_and_adverse_selection_edge_cases() -> None:
    assert alpha.impact_bps(0.01, threshold=0.01, coef_bps=8.0) == 0.0
    assert np.isclose(alpha.impact_bps(0.04, threshold=0.01, coef_bps=8.0), 1.6)
    assert alpha.adverse_selection_cost_bps(3.0) == 0.0
    assert alpha.adverse_selection_cost_bps(-3.0) == 3.0
    # Implementation shortfall: 10 + 0.5*2 - 1 - 3 = 7; the AS diagnostic is
    # embedded in the close-relative gross and not deducted again.
    assert alpha.net_execution_alpha_bps(
        alpha_gross_bps=10.0,
        fill_rate=0.5,
        maker_rebate_bps=2.0,
        commission_bps=1.0,
        impact_component_bps=3.0,
    ) == 7.0


@pytest.mark.unit
def test_alpha_raises_on_unknown_side() -> None:
    with pytest.raises(ValueError, match="Unknown side"):
        alpha.side_sign("HOLD")


@pytest.mark.unit
def test_net_alpha_identity_and_as_invariance() -> None:
    from analysis import config as cfg

    frame = pd.DataFrame({
        "side": ["BUY", "SELL"],
        "strategy": ["S1_STATIC", "S1_STATIC"],
        "close_price": [100.0, 50.0],
        "vwap_passive": [99.9, 50.2],
        "fill_rate": [0.4, 1.0],
        "adverse_selection_bps": [-3.0, 1.0],
        "size_frac": [0.05, 0.01],
    })
    out = alpha.attach_alpha_columns(frame)
    expected = (
        out["alpha_bps"]
        + out["fill_rate"] * cfg.MAKER_REBATE_BPS
        - cfg.COMMISSION_BPS
        - out["impact_bps"]
    )
    assert np.allclose(out["net_alpha_bps"], expected)
    # The AS diagnostic is reported but does not feed net alpha: scaling the
    # signed AS input leaves net alpha untouched and only moves the diagnostic.
    out_scaled = alpha.attach_alpha_columns(
        frame.assign(adverse_selection_bps=[-30.0, 10.0])
    )
    assert np.allclose(out_scaled["net_alpha_bps"], out["net_alpha_bps"])
    assert out_scaled["adverse_selection_cost_bps"].iloc[0] == pytest.approx(30.0)
