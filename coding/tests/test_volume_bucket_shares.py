"""Tests for the dollar-volume bucket-share statistics runner."""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from analysis import config as cfg
from analysis.runners.volume_bucket_shares import build_bucket_shares


def _make_db(path, rows) -> None:
    con = duckdb.connect(str(path))
    con.execute("""
        CREATE TABLE daily_volume (
            Ticker VARCHAR, Date DATE, Is_Witching_Day BOOLEAN,
            Pre_Market_Val DOUBLE, Open_Auction_Val DOUBLE,
            Morning_30m_Val DOUBLE, Mid_Day_Val DOUBLE,
            Afternoon_30m_Val DOUBLE, Close_Auction_Val DOUBLE,
            Post_Market_Val DOUBLE, Total_Daily_Val DOUBLE
        )
    """)
    con.executemany(
        "INSERT INTO daily_volume VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.close()


@pytest.mark.unit
def test_bucket_shares_filters_and_math(monkeypatch, artifact_dir) -> None:
    # Synthetic evaluation calendar: two regular days plus one documented
    # exclusion (2018-07-03 early close) that must be filtered out.
    taq_root = artifact_dir / "taq"
    for ds in ("20180102", "20180103", "20180703"):
        (taq_root / ds / "trades").mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(cfg.TAQ_PARQUET_DIR, 2018, taq_root)

    membership_root = artifact_dir / "membership"
    membership_root.mkdir(parents=True, exist_ok=True)
    (membership_root / "sp500_membership_intervals.csv").write_text(
        "index_id,symbol,effective_from,effective_to\n"
        "sp500,AAPL,2018-01-01,2019-12-31\n"
        "sp500,MSFT,2018-01-01,2018-01-02\n",  # member on day 1 only
        encoding="utf-8",
    )

    def row(ticker, date, witching, morning, midday, afternoon, close):
        total = morning + midday + afternoon + close
        return (ticker, date, witching, 0.0, 0.0, morning, midday,
                afternoon, close, 0.0, total)

    db_path = artifact_dir / "vol.duckdb"
    _make_db(db_path, [
        # AAPL both days: afternoon 20, midday 55, morning 15, close 10.
        row("AAPL", dt.date(2018, 1, 2), False, 15.0, 55.0, 20.0, 10.0),
        row("AAPL", dt.date(2018, 1, 3), True, 15.0, 55.0, 20.0, 10.0),
        # MSFT member only on Jan 2; its Jan 3 row must be dropped.
        row("MSFT", dt.date(2018, 1, 2), False, 10.0, 60.0, 20.0, 10.0),
        row("MSFT", dt.date(2018, 1, 3), True, 0.0, 0.0, 100.0, 0.0),
        # Non-member: always dropped.
        row("XXX", dt.date(2018, 1, 2), False, 0.0, 0.0, 100.0, 0.0),
        # Early-close day: excluded through the evaluation calendar.
        row("AAPL", dt.date(2018, 7, 3), False, 50.0, 50.0, 0.0, 0.0),
    ])

    out_dir = artifact_dir / "out"
    summary = build_bucket_shares(
        db_path, out_dir,
        start=dt.date(2018, 1, 2), end=dt.date(2018, 12, 31),
        universe="sp500", membership_root=membership_root,
    )

    # Surviving rows: AAPL Jan 2 + Jan 3, MSFT Jan 2 -> 3 symbol-days, 2 days.
    assert summary["symbol_days"] == 3
    assert summary["trading_days"] == 2
    assert summary["witching_symbol_days"] == 1
    # Volume-weighted afternoon share: (20+20+20) / (100+100+100) = 0.2.
    assert summary["afternoon_30m_share_vw"] == pytest.approx(0.2)
    # Close share VW: 30/300; EW identical here (all rows 10 percent).
    assert summary["close_auction_share_vw"] == pytest.approx(0.1)
    assert summary["close_auction_share_ew"] == pytest.approx(0.1)
    # Intensity ratio: 0.2 / ((170/300) / 11).
    midday_per_30m = (170.0 / 300.0) / 11.0
    assert summary["afternoon_to_midday_intensity_ratio"] == pytest.approx(
        0.2 / midday_per_30m,
    )
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["symbol_days"] == 3
    assert (out_dir / "bucket_shares.csv").exists()
