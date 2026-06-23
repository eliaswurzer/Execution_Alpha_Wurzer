from __future__ import annotations

import pandas as pd

from analysis.data import trade_conditions as tc
from analysis.data import taq_loader
from analysis.data.taq_loader import filter_valid_trades


def test_valid_correction_policy_is_strict() -> None:
    corrections = pd.Series(["00", "01", "", "02", None])

    mask = tc.valid_correction_mask(corrections)

    assert mask.tolist() == [True, True, False, False, False]


def test_preprocessing_preserves_open_and_close_markers() -> None:
    cond = pd.Series(["O", "Q", "6", "M", "Z", "B"])

    bad = tc.bad_sale_condition_mask(cond, "preprocessing")

    assert bad.tolist() == [False, False, False, False, True, True]


def test_evaluation_filter_is_stricter_but_keeps_closing_auction() -> None:
    cond = pd.Series(["O", "6", "M", "U", "Z"])

    bad = tc.bad_sale_condition_mask(cond, "evaluation")

    assert bad.tolist() == [True, False, False, False, True]


def test_loader_uses_central_evaluation_policy() -> None:
    df = pd.DataFrame({
        "time": pd.date_range("2018-01-02 09:30:00", periods=5, freq="s"),
        "symbol": ["TEST"] * 5,
        "exchange": ["Q"] * 5,
        "sale_condition": ["", "O", "6", "M", "Z"],
        "volume": [100] * 5,
        "price": [10.0] * 5,
        "correction": ["00", "00", "00", "01", "00"],
    })

    out = filter_valid_trades(df)

    assert out["sale_condition"].tolist() == ["", "6", "M"]


def test_trade_qc_status_accepts_expected_policy(monkeypatch) -> None:
    date = pd.Timestamp("2018-01-02").date()
    monkeypatch.setattr(
        taq_loader,
        "read_trade_qc_summary",
        lambda _date: {
            "trade_condition_policy_version": tc.POLICY_VERSION,
            "trade_filter_policy": "preprocessing",
        },
    )

    ok, msg = taq_loader.trade_qc_policy_status(date)

    assert ok
    assert msg == "ok"


def test_trade_qc_status_rejects_missing_manifest(monkeypatch) -> None:
    date = pd.Timestamp("2018-01-02").date()
    monkeypatch.setattr(taq_loader, "read_trade_qc_summary", lambda _date: None)

    ok, msg = taq_loader.trade_qc_policy_status(date)

    assert not ok
    assert "missing trade QC summary" in msg
