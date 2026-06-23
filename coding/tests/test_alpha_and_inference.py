from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from analysis import config as cfg
from analysis.inference.tests import primary_ttest
from analysis.metrics.alpha import attach_alpha_columns
from analysis.metrics.raear import break_even_eta, information_ratio


def test_alpha_keeps_moc_and_zero_passive_fill_rows() -> None:
    rows = pd.DataFrame([
        {
            "order_id": "moc",
            "strategy": "S0_MOC",
            "side": "BUY",
            "close_price": 100.0,
            "vwap_passive": np.nan,
            "fill_rate": 0.0,
            "adverse_selection_bps": 0.0,
            "size_frac": 0.10,
        },
        {
            "order_id": "zero",
            "strategy": "S2_TIME_ADAPTIVE",
            "side": "SELL",
            "close_price": 100.0,
            "vwap_passive": np.nan,
            "fill_rate": 0.0,
            "adverse_selection_bps": 0.0,
            "size_frac": 0.01,
        },
        {
            "order_id": "partial",
            "strategy": "S2_TIME_ADAPTIVE",
            "side": "BUY",
            "close_price": 100.0,
            "vwap_passive": 99.9,
            "fill_rate": 0.5,
            "adverse_selection_bps": -2.0,
            "size_frac": 0.01,
        },
        {
            "order_id": "invalid",
            "strategy": "S2_TIME_ADAPTIVE",
            "side": "BUY",
            "close_price": np.nan,
            "vwap_passive": np.nan,
            "fill_rate": 0.0,
            "adverse_selection_bps": 0.0,
            "size_frac": 0.01,
        },
    ])

    out = attach_alpha_columns(rows).set_index("order_id")

    assert out.loc["moc", "alpha_bps"] == 0.0
    assert out.loc["moc", "impact_bps"] == 0.0
    assert out.loc["moc", "net_alpha_bps"] == -cfg.COMMISSION_BPS
    assert out.loc["zero", "alpha_bps"] == 0.0
    assert out.loc["zero", "net_alpha_bps"] == -cfg.COMMISSION_BPS
    assert out.loc["partial", "adverse_selection_cost_bps"] == 2.0
    assert out.loc["partial", "alpha_bps"] > 0.0
    assert np.isnan(out.loc["invalid", "alpha_bps"])


def test_primary_h1_uses_headline_window_and_matched_moc_differential() -> None:
    d1 = dt.date(2018, 2, 1)
    d2 = dt.date(2018, 2, 2)
    rows = pd.DataFrame([
        {"order_id": "o1", "strategy": "S0_MOC", "window": "B", "size_frac": 0.01,
         "symbol": "AAPL", "date": d1, "net_alpha_bps": -0.1},
        {"order_id": "o1", "strategy": "S3_FULL", "window": "B", "size_frac": 0.01,
         "symbol": "AAPL", "date": d1, "net_alpha_bps": 1.9},
        {"order_id": "o2", "strategy": "S0_MOC", "window": "B", "size_frac": 0.01,
         "symbol": "MSFT", "date": d2, "net_alpha_bps": -0.1},
        {"order_id": "o2", "strategy": "S3_FULL", "window": "B", "size_frac": 0.01,
         "symbol": "MSFT", "date": d2, "net_alpha_bps": 3.9},
        {"order_id": "large", "strategy": "S0_MOC", "window": "B", "size_frac": 0.05,
         "symbol": "AAPL", "date": d1, "net_alpha_bps": -0.1},
        {"order_id": "large", "strategy": "S3_FULL", "window": "B", "size_frac": 0.05,
         "symbol": "AAPL", "date": d1, "net_alpha_bps": 99.9},
        {"order_id": "short", "strategy": "S0_MOC", "window": "C", "size_frac": 0.01,
         "symbol": "AAPL", "date": d1, "net_alpha_bps": -0.1},
        {"order_id": "short", "strategy": "S3_FULL", "window": "C", "size_frac": 0.01,
         "symbol": "AAPL", "date": d1, "net_alpha_bps": 88.9},
    ])

    result = primary_ttest(rows)

    assert result.n == 2
    assert result.mean == 3.0
    assert "S3_FULL-S0_MOC" in result.label


def test_near_zero_tracking_error_has_no_information_ratio() -> None:
    tiny_tev = np.finfo(float).eps

    assert np.isnan(information_ratio(-0.1, tiny_tev))
    assert np.isnan(break_even_eta(-0.1, tiny_tev))
