from __future__ import annotations

import datetime as dt
import time

import pandas as pd
import pytest

from pathlib import Path

from analysis.fill_model.adverse_selection import build_as_panel
from analysis.fill_model.state_vector import build_event_panel
from analysis.runners import calibrate_fill_model as calib


def _nbbo() -> pd.DataFrame:
    times = pd.date_range("2018-01-02 09:30:00", "2018-01-02 16:00:00", freq="30s")
    return pd.DataFrame({
        "time": times,
        "symbol": "TEST",
        "best_bid": 100.0,
        "best_bid_size": 1000,
        "best_offer": 101.0,
        "best_offer_size": 1200,
        "mid": 100.5,
        "half_spread": 0.5,
        "rel_spread": 1.0 / 100.5,
    })


def test_event_panel_dense_grid_expected_row_count() -> None:
    nbbo = _nbbo()
    trades = pd.DataFrame({
        "time": [pd.Timestamp("2018-01-02 10:30:10")],
        "price": [99.9],
        "volume": [100],
        "sale_condition": [""],
    })

    panel = build_event_panel(
        nbbo,
        trades,
        "TEST",
        dt.date(2018, 1, 2),
        sample_every_seconds=3600,
        offset_grid_bps=(0.0,),
    )

    assert len(panel) == 12
    assert set(panel["side"]) == {"BUY", "SELL"}


def test_event_panel_buy_and_sell_fill_rules() -> None:
    nbbo = _nbbo()
    trades = pd.DataFrame({
        "time": [
            pd.Timestamp("2018-01-02 10:30:10"),
            pd.Timestamp("2018-01-02 11:30:10"),
        ],
        "price": [99.9, 101.1],
        "volume": [100, 100],
        "sale_condition": ["", ""],
    })

    panel = build_event_panel(
        nbbo,
        trades,
        "TEST",
        dt.date(2018, 1, 2),
        sample_every_seconds=3600,
        offset_grid_bps=(0.0,),
    )

    buy_1030 = panel[(panel["t0"] == pd.Timestamp("2018-01-02 10:30:00")) & (panel["side"] == "BUY")]
    sell_1130 = panel[(panel["t0"] == pd.Timestamp("2018-01-02 11:30:00")) & (panel["side"] == "SELL")]
    assert int(buy_1030["event"].iloc[0]) == 1
    assert int(sell_1130["event"].iloc[0]) == 1


def test_as_panel_uses_array_window_fill_rules() -> None:
    nbbo = _nbbo()
    trades = pd.DataFrame({
        "time": [pd.Timestamp("2018-01-02 10:30:10")],
        "price": [99.9],
        "volume": [100],
    })

    panel = build_as_panel(
        nbbo,
        trades,
        horizon_seconds=30,
        sample_every_seconds=3600,
    )

    buy = panel[(panel["t0"] == pd.Timestamp("2018-01-02 10:30:00")) & (panel["side"] == "BUY")]
    sell = panel[(panel["t0"] == pd.Timestamp("2018-01-02 10:30:00")) & (panel["side"] == "SELL")]
    assert int(buy["fill"].iloc[0]) == 1
    assert int(sell["fill"].iloc[0]) == 0


def test_downcast_event_panel_has_compact_dtypes() -> None:
    panel = pd.DataFrame({
        "symbol": ["TEST"],
        "date": [dt.date(2018, 1, 2)],
        "side": ["BUY"],
        "t0": [pd.Timestamp("2018-01-02 10:00:00")],
        "limit_price": [100.0],
        "duration": [30.0],
        "event": [1],
        "q0": [500.0],
        "D0": [1000.0],
        "ofi_z": [0.1],
        "sigma": [0.01],
        "limit_offset_bps": [0.0],
        "half_spread_bps": [5.0],
    })

    out = calib._downcast_event_panel(panel)

    assert str(out["event"].dtype) == "int8"
    assert str(out["duration"].dtype) == "float32"
    assert str(out["side"].dtype) == "category"


def test_memory_guard_status_when_ram_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        calib,
        "_wait_for_memory",
        lambda **_: (False, "ram_percent=99.0;available_gb=0.1"),
    )

    status = calib._process_symbol_day(
        dt.date(2018, 1, 2),
        "TEST",
        Path("."),
        memory_wait_seconds=0,
    )

    assert status["status"] == "failed"
    assert status["reason"] == "memory_guard"


def test_process_symbol_day_missing_parquet_is_allowed_skip(monkeypatch, artifact_dir) -> None:
    monkeypatch.setattr(
        calib,
        "load_symbol_day",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )

    status = calib._process_symbol_day(dt.date(2018, 1, 2), "AAPL", artifact_dir)

    assert status["status"] == "skipped"
    assert status["reason"] == "missing_parquet"


def test_process_symbol_day_insufficient_quotes_skip(monkeypatch, artifact_dir) -> None:
    trades = pd.DataFrame({
        "time": [pd.Timestamp("2018-01-02 10:00:00")],
        "price": [100.0],
        "volume": [100],
        "sale_condition": [""],
    })
    monkeypatch.setattr(calib, "load_symbol_day", lambda *_args, **_kwargs: (trades, _nbbo()))
    monkeypatch.setattr(calib, "filter_valid_trades", lambda frame: frame)
    monkeypatch.setattr(calib, "filter_valid_quotes", lambda frame: frame.iloc[0:0])

    status = calib._process_symbol_day(dt.date(2018, 1, 2), "AAPL", artifact_dir)

    assert status["status"] == "skipped"
    assert status["reason"] == "insufficient_quotes"


def test_process_symbol_day_empty_after_trade_filter_skip(monkeypatch, artifact_dir) -> None:
    trades = pd.DataFrame({
        "time": [pd.Timestamp("2018-01-02 10:00:00")],
        "price": [100.0],
        "volume": [100],
        "sale_condition": [""],
    })
    monkeypatch.setattr(calib, "load_symbol_day", lambda *_args, **_kwargs: (trades, _nbbo()))
    monkeypatch.setattr(calib, "filter_valid_trades", lambda frame: frame.iloc[0:0])
    monkeypatch.setattr(calib, "filter_valid_quotes", lambda frame: frame)

    status = calib._process_symbol_day(dt.date(2018, 1, 2), "AAPL", artifact_dir)

    assert status["status"] == "skipped"
    assert status["reason"] == "empty_after_filter"


def test_process_symbol_day_dtype_error_status(monkeypatch, artifact_dir) -> None:
    monkeypatch.setattr(
        calib,
        "load_symbol_day",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("bad schema")),
    )

    status = calib._process_symbol_day(dt.date(2018, 1, 2), "AAPL", artifact_dir)

    assert status["status"] == "failed"
    assert status["reason"] == "dtype_error"
    assert "bad schema" in status["detail"]


def test_run_calibration_writes_failed_manifest_on_critical_status(monkeypatch, artifact_dir) -> None:
    day = dt.date(2018, 1, 2)
    monkeypatch.setattr(calib, "_select_dates", lambda start, end: [day])
    monkeypatch.setattr(
        calib,
        "_process_symbol_day",
        lambda *_args, **_kwargs: calib._status(
            day,
            "AAPL",
            status="failed",
            reason="dtype_error",
            started_at=time.perf_counter(),
            detail="synthetic failure",
        ),
    )

    with pytest.raises(RuntimeError, match="Calibration QC failed"):
        calib.run_calibration(
            ["AAPL"],
            start=day,
            end=day,
            out_dir=artifact_dir,
            workers=1,
        )

    manifest = pd.read_json(artifact_dir / "calibration_manifest.json", typ="series")
    assert manifest["status"] == "failed_qc"
    assert manifest["n_critical_failures"] == 1
